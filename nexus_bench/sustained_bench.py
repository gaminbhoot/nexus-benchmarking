"""Sustained NEXUS qualification: the representative dual-agent workload, run long.

Workload (documented exactly):
  UAV agent:   paced stream -> resize -> YOLO detect -> Re-ID embed ->
               DeepSORT-style associate -> telemetry packet -> fusion queue
  Rover agent: paced stream -> resize -> YOLO detect -> Re-ID embed ->
               associate -> occupancy-grid obstacle cells -> telemetry -> fusion queue
  Central:     drain queues -> world-state merge -> allocation tick -> fan-out

Concurrency model (cfg concurrency): "separate-contexts" (default, the intended
NEXUS deployment: one YOLO predictor per agent thread) or "serialized" (one
shared model behind a lock). The report states which was benchmarked.

KNOWN LIMITATION (explicit, not silent): no depth-estimation network is
implemented yet; the rover uses an occupancy-grid proxy. Depth remains NOT TESTED.

Windows (default 60 s) report evolution, never just an aggregate: throughput,
p95, VRAM, temp, utilization, power, queue depth, drops. Memory growth is
measured first-vs-last window (leak flag on sustained growth).
OOM kinds: WORKLOAD_OOM here (gate veto) vs CAPACITY_PROBE_OOM in memory_bench
(exploratory, no veto).
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

LIMITATIONS = [
    "rover depth network: NOT TESTED (occupancy-grid proxy in the rover agent)",
    "mission planner: NOT TESTED (allocation tick is a load-ranked assignment proxy)",
]

def _windowize(events, t0, window_s, duration_s):
    """events: [(t_wall, value)]. Returns per-window index -> list of values."""
    wins = {}
    for t, v in events:
        i = min(int((t - t0) // window_s), int(duration_s // window_s))
        wins.setdefault(i, []).append(v)
    return wins

def _series_stats(windows, duration_s, window_s, agg):
    n = max(1, int(duration_s // window_s))
    return [round(agg(windows.get(i, [])), 3) if windows.get(i) else None for i in range(n)]

def _growth(first_med, last_med, leak_threshold):
    growth = (last_med or 0) - (first_med or 0)
    return {"first": first_med, "last": last_med, "growth": round(growth, 2),
            "leak_suspected": bool(growth > leak_threshold)}

def _agent(name, width, do_obstacle, ctx, embed_model, dev, cfg, tele_q, stats,
           target_fps, source_keys, duration_s):
    import cv2
    src = PacedSource(cfg, target_fps=target_fps, queue_size=8,
                      source_keys=source_keys).start()
    stats["source"] = src.tag
    stats["source_kind"] = src.source_kind
    stats["real_pixels"] = src.real_pixels
    tr, prev = Tracker(), None
    events, qdepth, ooms = [], [], 0
    t_end = time.time() + duration_s
    while time.time() < t_end:
        item = src.get(timeout=2.0)
        if item is None:
            break
        if item == "retry":
            continue
        frame, meta = item
        qdepth.append((time.time(), src.q.qsize()))
        try:
            small = cv2.resize(frame, (width, width * 3 // 4))
            gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
            r = ctx.predict(name, small, width, dev)
            boxes = []
            if r.boxes is not None:
                for b in r.boxes.xywh.cpu().numpy()[:12]:
                    x, y, w, h = b
                    boxes.append([x - w / 2, y - h / 2, w, h])
            crops = []
            for b in boxes[:8]:
                x, y, w, h = [max(0, int(v)) for v in b]
                c = small[y:y + h, x:x + w]
                crops.append(c if c.size else np.zeros((32, 32, 3), np.uint8))
            feats = list(embed_crops(embed_model, crops, dev)[0]) if crops else []
            live = tr.step(boxes[:8], feats)
            obst = None
            if do_obstacle:
                tiny = cv2.resize(gray, (80, 60))
                _, occ = cv2.threshold(tiny, 100, 1, cv2.THRESH_BINARY)
                obst = int((occ == 0).sum())
            t_output = time.time()
            events.append((t_output, (t_output - meta["t_capture"]) * 1000))
            try:
                tele_q.put_nowait({"agent": name, "frame": meta["seq"],
                                   "tracks": len(live), "obstacle_cells": obst,
                                   "e2e_ms": (t_output - meta["t_capture"]) * 1000})
            except queue.Full:
                src.acct.queue_drops += 1
            src.acct.processed += 1
            src.acct.delivered += 1
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                ooms += 1  # WORKLOAD_OOM: gate veto (distinct from capacity probes)
                try:
                    import torch
                    if dev == "cuda":
                        torch.cuda.empty_cache()
                except Exception:
                    pass
            else:
                src.acct.processing_failures += 1
        except Exception:
            src.acct.processing_failures += 1
    src.stop()
    stats.update({"events": events, "qdepth": qdepth, "acct": src.acct.as_dict(),
                  "oom_events": ooms, "conservation": src.acct.check_conservation()})

def run(cfg):
    dev = resolve_device(cfg.get("device", "auto"))
    duration_s = cfg.get("sustained_duration_s", 600)
    window_s = cfg.get("sustained_window_s", 60)
    target_fps = cfg.get("target_fps") or 15.0
    concurrency = cfg.get("concurrency", "separate-contexts")
    req_fps = cfg.get("req_fps", 15) or 15
    deadline_ms = 1000.0 / req_fps
    out = {"module": "sustained",
           "config": {**cfg, "resolved_device": dev, "duration_s": duration_s,
                      "window_s": window_s, "target_fps": target_fps,
                      "concurrency_model": concurrency,
                      "deadline_ms": round(deadline_ms, 2),
                      "limitations": LIMITATIONS,
                      "drop_definition": "drop_pct = 100*(generated-delivered)/generated"},
           "tests": {}, "errors": [], "status": S.FULL}
    mp = (cfg.get("model") or "").strip()
    if not mp:
        out["status"] = S.UNSUPPORTED
        out["errors"].append("sustained needs --model: the representative workload cannot "
                             "run without YOLO, and no synthetic substitute is used.")
        with Monitor() as mon:
            out["telemetry"] = mon.summary()
        return out
    with Monitor(interval_s=1.0) as mon:
        try:
            want_fp16 = cfg.get("precision") == "fp16"
            ctx = ModelContexts(mp, want_fp16, mode=concurrency)
            try:
                ctx.configure(dev)
            except RuntimeError as e:
                out["status"] = S.FAILED
                out["errors"].append(str(e))
                out["telemetry"] = mon.summary()
                return out
            if not ctx.models:
                out["status"] = S.FAILED
                out["errors"].append(f"model unloadable: {ctx.prec_rec.get('error')}")
                out["telemetry"] = mon.summary()
                return out
            out["config"]["precision"] = ctx.prec_rec
            out["config"]["concurrency_model"] = ctx.mode + (
                " (separate YOLO objects; forwards serialized by a shared lock — "
                "concurrent same-process forwards segfault this stack, observed via "
                "faulthandler)" if ctx.mode == "separate-contexts" else
                (" (single shared model behind a lock — central scheduler)" if ctx.mode == "serialized"
                 else " (separate objects, lock-free — CUDA only)"))
            embed_model, etag = load_embedder(dev, (cfg.get("reid_model") or "").strip())
            out["config"]["embedder"] = etag
            agents = [("uav", 640, False, ("uav_video", "video")),
                      ("rover", 480, True, ("rover_video", "video"))]
            tele_q = queue.Queue(maxsize=256)
            stats, threads, own = {}, [], {}
            def _timed(name, width, obst, skeys):
                b0 = time.perf_counter()
                s = {}
                stats[name] = s
                _agent(name, width, obst, ctx, embed_model, dev, cfg, tele_q, s,
                       target_fps, skeys, duration_s)
                own[name] = time.perf_counter() - b0
            fusion_events, fusion_ticks = [], [0]
            def _fusion():
                world = {}
                while any(t.is_alive() for t in threads) or not tele_q.empty():
                    try:
                        p = tele_q.get(timeout=1.0)
                    except queue.Empty:
                        continue
                    t0 = time.perf_counter()
                    w = world.setdefault(p["agent"], {})
                    w[p["frame"]] = (p["tracks"], p["obstacle_cells"])
                    load = {a: sum(t for t, _ in wv.values()) for a, wv in world.items()}
                    _ = {a: f"task_{i}" for i, (a, _) in
                         enumerate(sorted(load.items(), key=lambda kv: kv[1]))}
                    fusion_events.append((time.time(), (time.perf_counter() - t0) * 1000))
                    fusion_ticks[0] += 1
            t0 = time.time()
            for name, width, obst, skeys in agents:
                threads.append(threading.Thread(target=_timed, daemon=True,
                                                args=(name, width, obst, skeys)))
            ft = threading.Thread(target=_fusion, daemon=True)
            for t in threads:
                t.start()
            ft.start()
            for t in threads:
                t.join(timeout=duration_s + 300)
            ft.join(timeout=30)
            # warmup through the real concurrency contract, then verify precision
            warm = np.zeros((cfg.get("imgsz", 640), cfg.get("imgsz", 640), 3), dtype=np.uint8)
            try:
                ctx.predict("uav", warm, cfg.get("imgsz", 640), dev)
            except Exception as e:
                out["status"] = S.FAILED
                out["errors"].append(f"warmup forward failed: {e}")
                out["telemetry"] = mon.summary()
                return out
            probe = ctx.models.get("uav", next(iter(ctx.models.values())))
            if not check_effective(probe, ctx.prec_rec):
                out["status"] = S.FAILED
                out["errors"].append(f"precision fail-closed: requested {ctx.prec_rec['requested']}, "
                                     f"effective {ctx.prec_rec['effective']}")
                out["telemetry"] = mon.summary()
                return out
            out["config"]["effective_precision"] = ctx.prec_rec["effective"]
            per_agent = {}
            for name, _, _, _ in agents:
                s = stats[name]
                a = s["acct"]
                wins = _windowize(s["events"], t0, window_s, duration_s)
                import statistics as _st
                n = max(1, int(duration_s // window_s))
                fps_w = [round(len(wins.get(i, [])) / window_s, 2) for i in range(n)]
                p95_w = [round(float(np.percentile(wins[i], 95)), 2) if wins.get(i) else None
                         for i in range(n)]
                lat_all = [v for w in wins.values() for v in w]
                d = summarize(lat_all) if lat_all else {"n": 0}
                d["windows_fps"] = fps_w
                d["windows_p95_ms"] = p95_w
                d["initial_fps"] = fps_w[0]
                steady = _st.median([f for f in fps_w[1:] if f is not None]) if len(fps_w) > 1 else fps_w[0]
                d["steady_fps"] = round(steady or 0, 2)
                d["min_fps"] = round(min(f for f in fps_w if f is not None) or 0, 2)
                d["final_fps"] = fps_w[-1]
                d["degradation_pct"] = round(100 * (fps_w[0] - steady) / fps_w[0], 1) if fps_w[0] else 0
                p95s = [p for p in p95_w if p is not None]
                d["initial_p95_ms"] = p95s[0] if p95s else None
                d["final_p95_ms"] = p95s[-1] if p95s else None
                d["max_p95_ms"] = round(max(p95s), 2) if p95s else None
                d["accounting"] = a
                d["drop_pct"] = drop_pct(a["generated"], a["delivered"])
                d["deadline_miss_pct"] = round(
                    100 * sum(1 for v in lat_all if v > deadline_ms) / len(lat_all), 2) if lat_all else 0
                d["oom_events"] = s["oom_events"]
                d["oom_kind"] = "WORKLOAD_OOM" if s["oom_events"] else None
                d["source"] = s["source"]
                d["real_pixels"] = s["real_pixels"]
                per_agent[name] = d
            qw = {}
            for name, _, _, _ in agents:
                qd = stats[name]["qdepth"]
                qwins = _windowize([(t, v) for t, v in qd], t0, window_s, duration_s)
                qw[name] = _series_stats(qwins, duration_s, window_s,
                                         lambda v: max(v) if v else 0)
            out["tests"]["per_agent"] = per_agent
            out["tests"]["queue_depth_max_per_window"] = qw
            out["tests"]["fusion"] = {"ticks": fusion_ticks[0],
                                      "tick_ms": summarize([v for _, v in fusion_events])}
            # resource evolution from monitor samples (VRAM/temp/power over time)
            evo = {}
            for key in ("vram_pct", "gpu_temp_c", "gpu_clock_mhz", "gpu_power_w",
                        "gpu_util_pct", "cpu_temp_c", "ram_pct", "proc_rss_mb"):
                pts = [(s["t"], s[key]) for s in mon.samples if key in s]
                if not pts:
                    evo[key] = None
                    continue
                t_start = pts[0][0]
                wins = _windowize(pts, t_start, window_s, duration_s)
                import statistics as _st
                med = _series_stats(wins, duration_s, window_s, _st.median)
                vals = [v for _, v in pts]
                first = _st.median(wins.get(0, vals[:1]))
                last_w = [v for i in sorted(wins) for v in [wins[i]]][-1]
                last = _st.median(last_w)
                evo[key] = {"initial": round(first, 2), "max": round(max(vals), 2),
                            "final": round(last, 2), "per_window": med}
            out["tests"]["resource_evolution"] = evo
            rss = evo.get("proc_rss_mb") or {}
            out["tests"]["memory_growth"] = _growth(
                rss.get("initial"), rss.get("final"), cfg.get("leak_threshold_mb", 300))
            if not all(s.get("real_pixels") for s in stats.values()):
                out["status"] = S.PARTIAL
                out["errors"].append("PARTIAL: a stream fell back to synthetic pixels "
                                     "(pass --uav-video/--rover-video for FULL)")
            if any(s["oom_events"] for s in stats.values()):
                out["errors"].append("WORKLOAD_OOM observed during sustained run (gate veto)")
        except Exception as e:
            out["status"] = S.FAILED
            out["errors"].append(f"sustained: {e}")
        out["telemetry"] = mon.summary()
    return out
