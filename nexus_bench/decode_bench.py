"""Video decode: CPU vs hardware decode path on real or generated clips.

With --video: FULL measurement of the actual NEXUS feed. Without: generates a
short H.264 clip (real codec path, labelled PARTIAL) so the decode pipeline is
still exercised. Reports decode FPS, per-frame ms, and the HW-acceleration
property OpenCV exposes (may be unavailable — reported, not assumed).
"""
import os
import time
import numpy as np

from nexus_bench import statuses as S
from nexus_bench.monitor import Monitor
from nexus_bench.stats import summarize

def _make_clip(path, n=90, w=640, h=480, fps=30):
    import cv2
    rng = np.random.default_rng(0)
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for _ in range(n):
        vw.write(rng.integers(0, 255, (h, w, 3), dtype=np.uint8))
    vw.release()
    return path

def run(cfg):
    import cv2
    out = {"module": "decode", "config": {**cfg}, "tests": {}, "errors": [],
           "status": S.FULL}
    with Monitor() as mon:
        try:
            vp = (cfg.get("video") or "").strip()
            tmp = None
            if vp and os.path.exists(vp):
                path = vp
            else:
                tmp = "/tmp/nexus_decode_probe.mp4"
                _make_clip(tmp)
                path = tmp
                out["status"] = S.PARTIAL
                out["errors"].append("PARTIAL: generated H.264 clip (pass --video for FULL)")
            cap = cv2.VideoCapture(path)
            try:
                hw = cap.get(cv2.CAP_PROP_HW_ACCELERATION)
            except Exception:
                hw = "unknown"
            out["config"]["hw_acceleration_prop"] = hw
            out["config"]["clip"] = path
            ts, n = [], 0
            t0 = time.perf_counter()
            while True:
                b0 = time.perf_counter()
                ok, _ = cap.read()
                if not ok:
                    break
                ts.append((time.perf_counter() - b0) * 1000)
                n += 1
            wall = time.perf_counter() - t0
            cap.release()
            d = summarize(ts)
            d["frames"] = n
            d["decode_fps"] = round(n / wall, 1) if wall else 0
            d["hw_acceleration"] = hw
            out["tests"]["decode_per_frame_ms"] = d
            if tmp:
                try:
                    os.remove(tmp)
                except Exception:
                    pass
        except Exception as e:
            out["status"] = S.FAILED
            out["errors"].append(f"decode: {e}")
        out["telemetry"] = mon.summary()
    return out
