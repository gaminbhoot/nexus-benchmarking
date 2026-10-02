"""Integrated NEXUS: explicit UAV + rover agents + central fusion.

Two modes (cfg integrated_mode):
  realtime    paced sources at target_fps into bounded queues for duration_s.
  throughput  as-fast-as-possible over agent_frames — max capacity only.

UAV agent uses --uav-video (-> --video fallback); rover uses --rover-video
(-> --video fallback). Each agent's actual source is recorded; two agents never
silently share one file while claiming distinct feeds (the config shows it).

Concurrency via ModelContexts (see concurrency.py): separate YOLO objects per
agent (real VRAM cost), forwards serialized — concurrent same-process forwards
segfault this stack (observed). The report states the contract verbatim.
Central drains while agents alive OR queues hold data OR window open.
Status FULL only with real YOLO + real pixels on BOTH agents.
"""
import queue
import threading
import time
import numpy as np

from nexus_bench import statuses as S
from nexus_bench.concurrency import ModelContexts
from nexus_bench.monitor import Monitor
from nexus_bench.profiles import resolve_device
from nexus_bench.reid_embed import embed_crops, load_embedder
from nexus_bench.stats import summarize
from nexus_bench.stream import PacedSource, drop_pct
from nexus_bench.tracking_bench import Tracker
from nexus_bench.vision_bench import _motion_detections
from nexus_bench.yolo_util import check_effective

def _process_frame(frame, width, do_obstacle, name, ctx, model, dev, tr, prev):
    import cv2
    small = cv2.resize(frame, (width, width * 3 // 4))
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    if ctx.models:
        r = ctx.predict(name, small, width, dev)
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

def _run_agent(name, width, obst, skeys, ctx, model, dev, cfg, q, stats,
               target_fps, duration_s):
    """One agent: paced source -> process -> telemetry queue, fully accounted."""
    src = PacedSource(cfg, target_fps=target_fps,
                      queue_size=8, source_keys=skeys, loop=True).start()
    stats["source"] = src.tag
    stats["source_kind"] = src.source_kind
    stats["real_pixels"] = src.real_pixels
    tr, prev, lat = Tracker(), None, []
    t_end = time.time() + duration_s
    while time.time() < t_end:
        item = src.get(timeout=2.0)
        if item is None:
            break
        if item == "retry":
            continue
        frame, meta = item
        t0 = time.perf_counter()
        try:
            live, obst, prev = _process_frame(frame, width, obst, name, ctx,
                                              model, dev, tr, prev)
        except Exception:
            src.acct.processing_failures += 1
            continue
        lat.append((time.perf_counter() - t0) * 1000)
        try:
            q.put_nowait({"agent": name, "frame": meta["seq"], "tracks": len(live),
                          "obstacle_cells": obst,
                          "e2e_ms": (time.time() - meta["t_capture"]) * 1000})
        except queue.Full:
            src.acct.queue_drops += 1
        src.acct.processed += 1
        src.acct.delivered += 1
    src.stop()
    a = src.acct.as_dict()
    stats.update({"lat": lat, "acct": a,
                  "drop_pct": drop_pct(a["generated"], a["delivered"]),
                  "conservation": src.acct.check_conservation()})

def _run_fast(name, width, obst, ctx, model, dev, cfg, q, stats):
    from nexus_bench.vision_bench import _moving_squares
    gen = iter(_moving_squares(cfg.get("agent_frames", 60),
                               seed=cfg.get("seed", 0) + (0 if name == "uav" else 99)))
    tr, prev, lat = Tracker(), None, []
    n, drops, t0 = 0, 0, time.perf_counter()
    for frame in gen:
        n += 1
        if n > cfg.get("agent_frames", 60):
            break
        b0 = time.perf_counter()
        live, obst, prev = _process_frame(frame, width, obst, name, ctx,
                                          model, dev, tr, prev)
        lat.append((time.perf_counter() - b0) * 1000)
        try:
            q.put_nowait({"agent": name, "frame": n, "tracks": len(live),
                          "obstacle_cells": obst})
        except queue.Full:
            drops += 1
    stats.update({"lat": lat, "wall_s": time.perf_counter() - t0,
                  "frames_out": n, "drops": drops, "real_pixels": False,
                  "source": "synthetic (throughput mode)"})

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
    duration_s = cfg.get("duration_s", 30)
    concurrency = cfg.get("concurrency", "separate-contexts")
    out = {"module": "integrated",
           "config": {**cfg, "resolved_device": dev, "mode": mode,
                      "target_fps": target_fps,
                      "drop_definition": "drop_pct = 100*(generated-delivered)/generated"},
           "tests": {}, "errors": [], "status": S.FULL}
    with Monitor() as mon:
        try:
            want_fp16 = cfg.get("precision") == "fp16"
            mp = (cfg.get("model") or "").strip()
            ctx = ModelContexts(mp, want_fp16, mode=concurrency)
            try:
                ctx.configure(dev)
            except RuntimeError as e:
                out["status"] = S.FAILED
                out["errors"].append(str(e))
                out["telemetry"] = mon.summary()
                return out
            out["config"]["detector"] = ctx.prec_rec.get("error", f"yolo ({ctx.prec_rec['requested']})") \
                if not ctx.models else f"yolo ({ctx.prec_rec['requested']})"
            out["config"]["precision"] = ctx.prec_rec
            out["config"]["concurrency_model"] = ctx.mode
            model, etag = load_embedder(dev, (cfg.get("reid_model") or "").strip())
            out["config"]["embedder"] = etag
            agents = [("uav", 640, False, ("uav_video", "video")),
                      ("rover", 480, True, ("rover_video", "video"))]
            verified = False

            def run_agent(name, width, obst, skeys, q, stats):
                if mode == "realtime":
                    _run_agent(name, width, obst, skeys, ctx, model, dev, cfg, q,
                               stats, target_fps, duration_s)
                else:
                    _run_fast(name, width, obst, ctx, model, dev, cfg, q, stats)

            if ctx.models:  # warmup through the real contract, then verify
                _run_fast("warmup", 640, False, ctx, model, dev,
                          dict(cfg, agent_frames=3), queue.Queue(maxsize=64), {})
                probe = ctx.models.get("uav", next(iter(ctx.models.values())))
                verified = check_effective(probe, ctx.prec_rec)
                out["config"]["effective_precision"] = ctx.prec_rec["effective"]
                if want_fp16 and not verified:
                    out["status"] = S.FAILED
                    out["errors"].append(f"precision fail-closed: requested {ctx.prec_rec['requested']}, "
                                         f"effective {ctx.prec_rec['effective']}")
                    out["telemetry"] = mon.summary()
                    return out
            # Sequential baseline.
            alone = {}
            for name, width, obst, skeys in agents:
                q, s = queue.Queue(maxsize=256), {}
                t0 = time.perf_counter()
                run_agent(name, width, obst, skeys, q, s)
                wall = time.perf_counter() - t0
                if mode == "realtime":
                    a = s["acct"]
                    fps = round(a["delivered"] / wall, 1) if wall else 0
                    alone[name] = {"fps": fps, "median_ms": summarize(s["lat"])["median_ms"],
                                   "drops": a["delivered"] and a["generated"] - a["delivered"],
                                   "drop_pct": s["drop_pct"]}
                else:
                    fps = round(s["frames_out"] / wall, 1) if wall else 0
                    alone[name] = {"fps": fps, "median_ms": summarize(s["lat"])["median_ms"],
                                   "drops": s.get("drops", 0)}
                out["config"][f"{name}_source"] = s.get("source")
            out["tests"]["alone"] = alone
            # Concurrent mission.
            qs, stats, threads, own = [], {}, [], {}
            def _timed(name, width, obst, skeys, q, s):
                b0 = time.perf_counter()
                run_agent(name, width, obst, skeys, q, s)
                own[name] = time.perf_counter() - b0
            for name, width, obst, skeys in agents:
                q = queue.Queue(maxsize=8 if mode == "realtime" else 256)
                qs.append(q)
                s = {}
                stats[name] = s
                threads.append(threading.Thread(target=_timed, daemon=True,
                                                args=(name, width, obst, skeys, q, s)))
            for t in threads:
                t.start()
            central = _central(qs, threads, duration_s + 60)
            for t in threads:
                t.join(timeout=120)
            conc = {}
            for name, _, _, _ in agents:
                s = stats[name]
                if mode == "realtime":
                    a = s["acct"]
                    dur = own.get(name, 0)
                    fps = round(a["delivered"] / dur, 1) if dur else 0
                    lat = summarize(s["lat"])
                    conc[name] = {"fps": fps, "median_ms": lat["median_ms"],
                                  "p95_ms": lat["p95_ms"], "p99_ms": lat["p99_ms"],
                                  "generated": a["generated"], "delivered": a["delivered"],
                                  "drop_pct": s["drop_pct"], "conservation": s["conservation"]}
                else:
                    dur = own.get(name, 0)
                    fps = round(s["frames_out"] / dur, 1) if dur else 0
                    lat = summarize(s["lat"])
                    conc[name] = {"fps": fps, "median_ms": lat["median_ms"],
                                  "p95_ms": lat["p95_ms"], "p99_ms": lat["p99_ms"],
                                  "drops": s.get("drops", 0), "frames_out": s["frames_out"]}
                af = alone[name]["fps"]
                conc[name]["vs_alone_pct"] = round(100 * fps / af, 1) if af else None
            out["tests"]["per_agent"] = conc
            out["tests"]["concurrent_fps"] = {k: v["fps"] for k, v in conc.items()}
            out["tests"]["fusion"] = central
            out["tests"]["note"] = ("realtime: paced inputs, bounded queues, conservation-"
                                    "accounted drops. vs_alone_pct <100 = contention cost.")
            real = all(stats[n].get("real_pixels", False) for n, _, _, _ in agents)
            if not ctx.models or not real:
                out["status"] = S.PARTIAL
                out["errors"].append("PARTIAL: fallback detector and/or synthetic pixels "
                                     "(informational, not gate-eligible)")
        except Exception as e:
            out["status"] = S.FAILED
            out["errors"].append(f"integrated: {e}")
        out["telemetry"] = mon.summary()
    return out
