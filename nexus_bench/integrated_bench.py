"""Integrated NEXUS: simultaneous UAV+rover perception, tracking, fusion, mission sim."""
import threading
import time

from nexus_bench import cpu_bench, comms_bench, inference_bench, vision_bench
from nexus_bench.monitor import Monitor

def _timed(fn, cfg, store, key):
    t0 = time.perf_counter()
    try:
        store[key] = fn(cfg)
    except Exception as e:
        store[key] = {"module": key, "errors": [str(e)]}
    store[key + "_wall_s"] = round(time.perf_counter() - t0, 2)

def run(cfg):
    out = {"module": "integrated", "config": cfg, "tests": {}, "errors": []}
    with Monitor() as mon:
        # Baseline: sequential
        seq, t0 = {}, time.perf_counter()
        for name, fn in (("inference", inference_bench.run), ("vision", vision_bench.run),
                         ("cpu", cpu_bench.run), ("comms", comms_bench.run)):
            _timed(fn, cfg, seq, name)
        seq_wall = time.perf_counter() - t0
        # Combined: threads contend for CPU/GPU like the real stack
        conc, t0 = {}, time.perf_counter()
        threads = [threading.Thread(target=_timed, args=(fn, cfg, conc, name), daemon=True)
                   for name, fn in (("inference", inference_bench.run), ("vision", vision_bench.run),
                                     ("cpu", cpu_bench.run), ("comms", comms_bench.run))]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=cfg.get("duration_s", 60) + 120)
        conc_wall = time.perf_counter() - t0
        out["tests"]["sequential_wall_s"] = round(seq_wall, 2)
        out["tests"]["concurrent_wall_s"] = round(conc_wall, 2)
        out["tests"]["contention_ratio"] = round(conc_wall / seq_wall, 2) if seq_wall else None
        out["tests"]["mission_sim"] = {
            "uav_perception": conc.get("inference_wall_s"),
            "rover_tracking": conc.get("vision_wall_s"),
            "fusion_telemetry": conc.get("comms_wall_s"),
            "note": "ratio >1 means resource contention; >>1 flags a bottleneck",
        }
        out["subresults"] = {k: v for k, v in conc.items() if not k.endswith("_wall_s")}
        out["telemetry"] = mon.summary()
    return out
