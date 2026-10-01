"""Integrated NEXUS: explicit UAV + rover agents + central fusion.

Two modes (cfg integrated_mode):
  realtime    paced sources at target_fps into bounded queues for duration_s —
              the purchase-gate mode: input vs processed vs dropped, queue depth.
  throughput  as-fast-as-possible over agent_frames — max capacity only.

UAV agent:   stream -> YOLO (or labelled PARTIAL fallback) -> track -> telemetry queue.
Rover agent: stream -> YOLO -> track + obstacle occupancy grid -> telemetry queue.
Central:     drain queues while agents alive OR data remains OR window open ->
             world-state merge -> allocation tick. Thread refs are NEVER popped.
Status FULL only with real YOLO + real pixels; else PARTIAL (not gate-eligible).
"""
import queue
import threading
import time
import numpy as np

from nexus_bench import statuses as S
from nexus_bench.monitor import Monitor
from nexus_bench.profiles import resolve_device
from nexus_bench.reid_embed import embed_crops, load_embedder
from nexus_bench.stats import summarize
from nexus_bench.stream import PacedSource
from nexus_bench.tracking_bench import Tracker
from nexus_bench.vision_bench import _motion_detections
from nexus_bench.yolo_util import check_effective, load_weights, precision_kwargs

def _process_frame(frame, width, do_obstacle, yolo, pk, model, dev, cfg, tr, prev):
    import cv2
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
    crops = []
    for b in boxes[:8]:
        x, y, w, h = [max(0, int(v)) for v in b]
        c = small[y:y + h, x:x + w]
        crops.append(c if c.size else np.zeros((32, 32, 3), np.uint8))
    feats = list(embed_crops(model, crops, dev)[0]) if crops else []
    live = tr.step(boxes[:8], feats)
    obst = None
    if do_obstacle:
        tiny = cv2.resize(gray, (80, 60))
        _, occ = cv2.threshold(tiny, 100, 1, cv2.THRESH_BINARY)
        obst = int((occ == 0).sum())
    return live, obst, gray

def _agent_paced(name, width, do_obstacle, yolo, pk, model, dev, cfg, q, stats, target_fps):
    """Real-time consumer: paced source, bounded queue, drops counted."""
    src = PacedSource(cfg, target_fps=target_fps).start()
    tr, prev, lat = Tracker(), None, []
    n_out, t_end = 0, time.time() + cfg.get("duration_s", 30)
    max_q = 0
    while time.time() < t_end:
        item = src.get(timeout=2.0)
        if item is None:
            break
        if item == "retry":
            continue
        frame, capture_t, seq = item
        max_q = max(max_q, src.q.qsize())
        t0 = time.perf_counter()
        live, obst, prev = _process_frame(frame, width, do_obstacle, yolo, pk,
                                         model, dev, cfg, tr, prev)
        lat.append((time.perf_counter() - t0) * 1000)
        try:
            q.put_nowait({"agent": name, "frame": seq, "tracks": len(live),
                          "obstacle_cells": obst, "e2e_ms": (time.time() - capture_t) * 1000})
        except queue.Full:
            stats["drops"] = stats.get("drops", 0) + 1
        n_out += 1
    src.stop()
    stats.update({"frames_out": n_out, "frames_in": src.emitted,
                  "drops": stats.get("drops", 0) + (src.emitted - n_out) + src.dropped_producer,
                  "lat": lat, "max_q": max_q, "source": src.tag,
                  "real_pixels": src.real_pixels})

def _agent_fast(name, gen, width, do_obstacle, yolo, pk, model, dev, cfg, q, stats):
    tr, prev, lat = Tracker(), None, []
    n = 0
    t0 = time.perf_counter()
    for frame in gen:
        n += 1
        if n > cfg.get("agent_frames", 60):
            break
        b0 = time.perf_counter()
        live, obst, prev = _process_frame(frame, width, do_obstacle, yolo, pk,
                                          model, dev, cfg, tr, prev)
        lat.append((time.perf_counter() - b0) * 1000)
        try:
            q.put_nowait({"agent": name, "frame": n, "tracks": len(live),
                          "obstacle_cells": obst})
        except queue.Full:
            stats["drops"] = stats.get("drops", 0) + 1
    stats.update({"frames_out": n, "lat": lat, "wall_s": time.perf_counter() - t0,
                  "drops": stats.get("drops", 0)})

def _central(qs, threads, budget_s):
    """Drain while agents alive OR queues hold data OR window open. No popping."""
    world, ticks, tick_ts, max_q = {}, 0, [], 0
    t_end = time.time() + budget_s
    while time.time() < t_end:
        if not any(t.is_alive() for t in threads) and all(q.empty() for q in qs):
            break
        got = False
        for q in qs:
            max_q = max(max_q, q.qsize())
            try:
                while True:
                    p = q.get_nowait()
                    got = True
                    w = world.setdefault(p["agent"], {})
                    w[p["frame"]] = (p["tracks"], p["obstacle_cells"])
            except queue.Empty:
                pass
        t0 = time.perf_counter()
        load = {a: sum(t for t, _ in w.values()) for a, w in world.items()}
        _ = {a: f"task_{i}" for i, (a, _) in enumerate(sorted(load.items(), key=lambda kv: kv[1]))}
        ticks += 1
        tick_ts.append((time.perf_counter() - t0) * 1000)
        if not got:
            time.sleep(0.005)
    return {"ticks": ticks, "tick_ms": summarize(tick_ts) if tick_ts else {},
            "world_keys": {a: len(w) for a, w in world.items()}, "max_q": max_q}

def run(cfg):
    dev = resolve_device(cfg.get("device", "auto"))
    mode = cfg.get("integrated_mode", "realtime")
    target_fps = cfg.get("target_fps") or 15.0
    out = {"module": "integrated",
           "config": {**cfg, "resolved_device": dev, "mode": mode, "target_fps": target_fps},
           "tests": {}, "errors": [], "status": S.FULL}
    with Monitor() as mon:
        try:
            want_fp16 = cfg.get("precision") == "fp16"
            yolo, prec_rec = load_weights((cfg.get("model") or "").strip(), want_fp16=want_fp16)
            pk = precision_kwargs(want_fp16) if yolo else {}
            out["config"]["detector"] = prec_rec.get("error", f"yolo ({prec_rec['requested']})") \
                if yolo is None else f"yolo ({prec_rec['requested']})"
            out["config"]["precision"] = prec_rec
            model, etag = load_embedder(dev, (cfg.get("reid_model") or "").strip())
            out["config"]["embedder"] = etag
            agents = [("uav", 640, False), ("rover", 480, True)]
            full_evidence = yolo is not None
            verified = False

            def run_agent(name, width, obst, q, stats):
                nonlocal verified
                if mode == "realtime":
                    _agent_paced(name, width, obst, yolo, pk, model, dev, cfg, q, stats, target_fps)
                else:
                    from nexus_bench.vision_bench import _moving_squares
                    gen = iter(_moving_squares(cfg.get("agent_frames", 60),
                                               seed=cfg.get("seed", 0) + (0 if name == "uav" else 99)))
                    _agent_fast(name, gen, width, obst, yolo, pk, model, dev, cfg, q, stats)
                if yolo is not None and not verified:
                    verified = check_effective(yolo, prec_rec)
                    out["config"]["effective_precision"] = prec_rec["effective"]

            # Warmup (cold kernels must not bias the baseline).
            from nexus_bench.vision_bench import _moving_squares
            _agent_fast("warmup", iter(_moving_squares(3, seed=1)), 640, False,
                        yolo, pk, model, dev, dict(cfg, agent_frames=3),
                        queue.Queue(maxsize=64), {})
            if yolo is not None and not verified:
                verified = check_effective(yolo, prec_rec)
                out["config"]["effective_precision"] = prec_rec["effective"]
                if not verified:
                    out["status"] = S.FAILED
                    out["errors"].append(f"precision fail-closed: requested {prec_rec['requested']}, "
                                         f"effective {prec_rec['effective']}")
                    out["telemetry"] = mon.summary()
                    return out
            # Sequential baseline.
            alone = {}
            for name, width, obst in agents:
                q, s = queue.Queue(maxsize=256), {}
                t0 = time.perf_counter()
                run_agent(name, width, obst, q, s)
                wall = time.perf_counter() - t0
                full_evidence &= s.get("real_pixels", mode == "throughput")
                fps = round(s["frames_out"] / wall, 1) if wall else 0
                alone[name] = {"fps": fps, "median_ms": summarize(s["lat"])["median_ms"],
                               "drops": s.get("drops", 0)}
                if "source" in s:
                    out["config"][f"{name}_source"] = s["source"]
            out["tests"]["alone"] = alone
            # Concurrent mission.
            qs, stats, threads, own = [], {}, [], {}
            def _timed(name, width, obst, q, s):
                b0 = time.perf_counter()
                run_agent(name, width, obst, q, s)
                own[name] = time.perf_counter() - b0
            for name, width, obst in agents:
                q = queue.Queue(maxsize=8 if mode == "realtime" else 256)
                qs.append(q)
                s = {}
                stats[name] = s
                threads.append(threading.Thread(target=_timed, daemon=True,
                                                args=(name, width, obst, q, s)))
            for t in threads:
                t.start()
            central = _central(qs, threads, cfg.get("duration_s", 30) + 60)
            for t in threads:
                t.join(timeout=60)
            conc = {}
            for name, _, _ in agents:
                s = stats[name]
                dur = own.get(name, 0)
                fps = round(s["frames_out"] / dur, 1) if dur else 0
                lat = summarize(s["lat"])
                conc[name] = {"fps": fps, "median_ms": lat["median_ms"],
                              "p95_ms": lat["p95_ms"], "p99_ms": lat["p99_ms"],
                              "drops": s.get("drops", 0), "max_q": s.get("max_q"),
                              "frames_in": s.get("frames_in"), "frames_out": s["frames_out"]}
                af = alone[name]["fps"]
                conc[name]["vs_alone_pct"] = round(100 * fps / af, 1) if af else None
            out["tests"]["per_agent"] = conc
            out["tests"]["concurrent_fps"] = {k: v["fps"] for k, v in conc.items()}
            out["tests"]["fusion"] = central
            out["tests"]["note"] = ("realtime mode: bounded queues, paced inputs — drops/queue "
                                    "depth are the scaling signal. vs_alone_pct <100 = contention cost.")
            if yolo is None or not full_evidence:
                out["status"] = S.PARTIAL
                out["errors"].append("PARTIAL: fallback detector and/or synthetic pixels "
                                     "(informational, not gate-eligible)")
        except Exception as e:
            out["status"] = S.FAILED
            out["errors"].append(f"integrated: {e}")
        out["telemetry"] = mon.summary()
    return out
