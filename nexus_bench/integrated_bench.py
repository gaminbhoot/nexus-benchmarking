"""Integrated NEXUS: explicit UAV + rover agents + central fusion, sequential vs concurrent.

UAV agent:   stream -> YOLO (or labelled fallback) -> track -> telemetry queue.
Rover agent: stream -> YOLO -> track + obstacle occupancy grid -> telemetry queue.
Central:     drain queues -> world-state merge -> task-allocation tick -> fan-out.
Compares per-agent FPS alone vs under contention, fusion tick health, queue drops.
"""
import os
import queue
import threading
import time
import numpy as np

from nexus_bench.monitor import Monitor
from nexus_bench.profiles import resolve_device
from nexus_bench.reid_embed import embed_crops, load_embedder
from nexus_bench.stats import summarize
from nexus_bench.tracking_bench import Tracker
from nexus_bench.vision_bench import _motion_detections, _moving_squares
from nexus_bench.yolo_util import load_weights, precision_kwargs

def _source(cfg, key, seed):
    import cv2
    vp = (cfg.get(key) or cfg.get("video") or "").strip()
    if vp and os.path.exists(vp):
        cap = cv2.VideoCapture(vp)
        def gen():
            while True:
                ok, f = cap.read()
                if not ok:
                    break
                yield cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            cap.release()
        return gen(), f"video:{vp}"
    n = cfg.get("agent_frames", 60)
    return iter(_moving_squares(n, seed=seed)), "synthetic (labelled)"

def _load_yolo(cfg):
    model, tag = load_weights((cfg.get("model") or "").strip(),
                              want_fp16=(cfg.get("precision") == "fp16"))
    pk = precision_kwargs(model, cfg.get("precision") == "fp16") if model else {}
    return model, pk, tag

def _agent(name, gen, width, do_obstacle, yolo, pk, model, dev, cfg, q, stats):
    import cv2
    tr = Tracker()
    lat, prev, n, drops = [], None, 0, 0
    for frame in gen:
        n += 1
        if n > cfg.get("agent_frames", 60):
            break
        t0 = time.perf_counter()
        small = cv2.resize(frame, (width, width * 3 // 4))
        gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
        if yolo is not None:
            r = yolo.predict(small, imgsz=width, device=dev, verbose=False, **pk)[0]
            boxes = []
            if r.boxes is not None:
                for b in r.boxes.xywh.cpu().numpy()[:12]:
                    x, y, w, h = b
                    boxes.append([x - w / 2, y - h / 2, w, h])
        else:
            boxes = _motion_detections(prev, gray) if prev is not None else []
        prev = gray
        crops = []
        for b in boxes[:8]:
            x, y, w, h = [max(0, int(v)) for v in b]
            c = small[y:y + h, x:x + w]
            crops.append(c if c.size else np.zeros((32, 32, 3), np.uint8))
        feats = list(embed_crops(model, crops, dev)[0]) if crops else []
        live = tr.step(boxes[:8], feats)
        obst = None
        if do_obstacle:  # rover: occupancy grid + obstacle cell count (real work)
            tiny = cv2.resize(gray, (80, 60))
            _, occ = cv2.threshold(tiny, 100, 1, cv2.THRESH_BINARY)
            obst = int((occ == 0).sum())
        pkt = {"agent": name, "frame": n, "tracks": len(live), "obstacle_cells": obst}
        try:
            q.put_nowait(pkt)
        except queue.Full:
            drops += 1
        lat.append((time.perf_counter() - t0) * 1000)
    stats.update({"frames": n, "drops": drops, "lat": lat})

def _central(qs, stats):
    """World-state merge + allocation tick. Returns tick stats."""
    world, ticks, tick_ts = {}, 0, []
    t_end = time.time() + stats.pop("_budget_s", 120)
    alive = True
    while alive and time.time() < t_end:
        got = False
        for q in qs:
            try:
                while True:
                    p = q.get_nowait()
                    got = True
                    w = world.setdefault(p["agent"], {})
                    w[p["frame"]] = (p["tracks"], p["obstacle_cells"])
            except queue.Empty:
                pass
        t0 = time.perf_counter()
        # allocation tick: rank agents by pending tracks, assign tasks (real, cheap).
        load = {a: sum(t for t, _ in w.values()) for a, w in world.items()}
        ranked = sorted(load.items(), key=lambda kv: kv[1])
        _ = {a: f"task_{i}" for i, (a, _) in enumerate(ranked)}
        ticks += 1
        tick_ts.append((time.perf_counter() - t0) * 1000)
        alive = any(t.is_alive() for t in stats.pop("_threads", []))
        if not got:
            time.sleep(0.005)
    return {"ticks": ticks, "tick_ms": summarize(tick_ts) if tick_ts else {},
            "world_keys": {a: len(w) for a, w in world.items()}}

def run(cfg):
    dev = resolve_device(cfg.get("device", "auto"))
    out = {"module": "integrated", "config": {**cfg, "resolved_device": dev},
           "tests": {}, "errors": []}
    with Monitor() as mon:
        try:
            yolo, pk, dtag = _load_yolo(cfg)
            out["config"]["detector"] = dtag
            model, etag = load_embedder(dev, (cfg.get("reid_model") or "").strip())
            out["config"]["embedder"] = etag
            agents = [("uav", "uav_video", 640, False, 11), ("rover", "rover_video", 480, True, 22)]
            # Warmup first so the alone baseline isn't penalized by cold kernels.
            wgen, _ = _source(cfg, "uav_video", 99)
            _agent("warmup", wgen, 640, False, yolo, pk, model, dev,
                   dict(cfg, agent_frames=3), queue.Queue(maxsize=64), {})
            # Sequential baseline: one agent at a time, no contention.
            alone = {}
            for name, vkey, width, obst, seed in agents:
                gen, stag = _source(cfg, vkey, seed)
                out["config"][f"{name}_source"] = stag
                q, stats = queue.Queue(maxsize=64), {}
                t0 = time.perf_counter()
                _agent(name, gen, width, obst, yolo, pk, model, dev, cfg, q, stats)
                wall = time.perf_counter() - t0
                s = stats
                alone[name] = {"fps": round(s["frames"] / wall, 1),
                               "median_ms": summarize(s["lat"])["median_ms"]}
            out["tests"]["alone_fps"] = {k: v["fps"] for k, v in alone.items()}
            # Concurrent: both agents + central fusion contend.
            qs, stats, threads, own_time = [], {}, [], {}
            def _timed(name, gen, width, obst, q, s):
                t0 = time.perf_counter()
                _agent(name, gen, width, obst, yolo, pk, model, dev, cfg, q, s)
                own_time[name] = time.perf_counter() - t0
            for name, vkey, width, obst, seed in agents:
                gen, _ = _source(cfg, vkey, seed)
                q = queue.Queue(maxsize=64)
                qs.append(q)
                s = {}
                stats[name] = s
                t = threading.Thread(target=_timed, daemon=True,
                                     args=(name, gen, width, obst, q, s))
                threads.append(t)
            for t in threads:
                t.start()
            central = _central(qs, {"_threads": threads,
                                    "_budget_s": cfg.get("duration_s", 60) + 120})
            for t in threads:
                t.join(timeout=30)
            conc = {}
            for name, _, _, _, _ in agents:
                s = stats[name]
                dur = own_time.get(name, 0)
                fps = round(s["frames"] / dur, 1) if dur else 0
                conc[name] = {"fps": fps,
                              "median_ms": summarize(s["lat"])["median_ms"],
                              "drops": s["drops"]}
                af = alone[name]["fps"]
                conc[name]["vs_alone_pct"] = round(100 * fps / af, 1) if af else None
            out["tests"]["concurrent_fps"] = {k: v["fps"] for k, v in conc.items()}
            out["tests"]["per_agent"] = conc
            out["tests"]["fusion"] = central
            out["tests"]["note"] = ("vs_alone_pct <100 shows contention cost of running the "
                                    "full stack on this machine.")
        except Exception as e:
            out["errors"].append(f"integrated: {e}")
        out["telemetry"] = mon.summary()
    return out
