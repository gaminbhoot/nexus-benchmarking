"""Cold start: fresh-process model load + CUDA init + first vs steady inference.

Runs inside the already-fresh worker process, so load_ms genuinely includes
process-level init. Reports load, first-inference, steady median separately.
"""
import time
import numpy as np

from nexus_bench import statuses as S
from nexus_bench.monitor import Monitor
from nexus_bench.profiles import resolve_device
from nexus_bench.stats import summarize

def run(cfg):
    import os
    dev = resolve_device(cfg.get("device", "auto"))
    mp = (cfg.get("model") or "").strip()
    imgsz = cfg.get("imgsz", 640)
    out = {"module": "coldstart", "config": {**cfg, "resolved_device": dev},
           "tests": {}, "errors": [], "status": S.FULL}
    if not mp or not os.path.exists(mp):
        out["status"] = S.UNSUPPORTED
        out["errors"].append("coldstart needs --model weights")
        with Monitor() as mon:
            out["telemetry"] = mon.summary()
        return out
    with Monitor() as mon:
        try:
            from ultralytics import YOLO
            t0 = time.perf_counter()
            model = YOLO(mp)
            load_ms = (time.perf_counter() - t0) * 1000
            frame = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
            t0 = time.perf_counter()
            model.predict(frame, imgsz=imgsz, device=dev, verbose=False)
            first_ms = (time.perf_counter() - t0) * 1000
            ts = []
            for _ in range(max(5, cfg.get("repeats", 10))):
                t0 = time.perf_counter()
                model.predict(frame, imgsz=imgsz, device=dev, verbose=False)
                ts.append((time.perf_counter() - t0) * 1000)
            d = summarize(ts)
            d["load_ms"] = round(load_ms, 1)
            d["first_inference_ms"] = round(first_ms, 1)
            d["cold_penalty_x"] = round(first_ms / d["median_ms"], 1) if d["median_ms"] else None
            out["tests"]["cold_vs_steady_ms"] = d
        except Exception as e:
            out["status"] = S.FAILED
            out["errors"].append(f"coldstart: {e}")
        out["telemetry"] = mon.summary()
    return out
