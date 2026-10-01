"""CPU: single-thread, multi-thread, preprocess, decode, sustained."""
import base64
import io
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np

from nexus_bench.monitor import Monitor
from nexus_bench.safety import Guard
from nexus_bench.stats import summarize

def _jpeg_bytes(h=480, w=640, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (h, w, 3), dtype=np.uint8)

def _matmul(n=256, repeats=5):
    rng = np.random.default_rng(0)
    a = rng.random((n, n)); b = rng.random((n, n))
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter(); a @ b; ts.append((time.perf_counter() - t0) * 1000)
    return ts

def run(cfg):
    guard = Guard()
    out = {"module": "cpu", "config": cfg, "tests": {}, "errors": []}
    with Monitor() as mon:
        try:
            out["tests"]["single_thread_matmul_ms"] = summarize(_matmul(256, cfg.get("repeats", 20)))
        except Exception as e:
            out["errors"].append(f"single_thread: {e}")
        try:
            r = cfg.get("repeats", 20)
            with ThreadPoolExecutor() as ex:
                t0 = time.perf_counter()
                list(ex.map(lambda _: _matmul(128, 2), range(max(4, r // 5))))
                dt = (time.perf_counter() - t0) * 1000
            out["tests"]["multithread_pool_ms"] = {"total_ms": round(dt, 1)}
        except Exception as e:
            out["errors"].append(f"multithread: {e}")
        try:
            import cv2
            frame = _jpeg_bytes()
            ts = []
            for _ in range(cfg.get("repeats", 20)):
                t0 = time.perf_counter()
                small = cv2.resize(frame, (320, 240))
                _ = cv2.cvtColor(small, cv2.COLOR_RGB2BGR)
                _, buf = cv2.imencode(".jpg", small)
                _ = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                ts.append((time.perf_counter() - t0) * 1000)
            out["tests"]["preprocess_decode_ms"] = summarize(ts)
        except Exception as e:
            out["errors"].append(f"preprocess: {e}")
        try:
            t_end = time.time() + min(cfg.get("duration_s", 20), 15)
            it = 0; t0 = time.time()
            while time.time() < t_end and guard.ok(mon.samples[-1] if mon.samples else None):
                _matmul(128, 1); it += 1
            dt = time.time() - t0
            out["tests"]["sustained"] = {"iters": it, "seconds": round(dt, 2),
                                         "ips": round(it / dt, 1) if dt else 0}
            if guard.reason:
                out["errors"].append(f"sustained stopped: {guard.reason}")
        except Exception as e:
            out["errors"].append(f"sustained: {e}")
        out["telemetry"] = mon.summary()
    return out
