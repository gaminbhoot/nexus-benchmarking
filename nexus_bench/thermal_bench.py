"""Thermal stability: sustained load, degradation analysis, throttling flags."""
import time
import numpy as np

from nexus_bench.monitor import Monitor
from nexus_bench.profiles import resolve_device
from nexus_bench.safety import Guard, Limits

def run(cfg):
    dev = resolve_device(cfg.get("device", "auto"))
    duration = min(cfg.get("duration_s", 60), 300)
    guard = Guard(Limits(max_duration_s=duration + 30))
    out = {"module": "thermal", "config": {**cfg, "resolved_device": dev, "sustained_s": duration},
           "tests": {}, "errors": []}
    import torch
    a = torch.randn(512, 512)
    if dev in ("cuda", "mps"):
        try:
            a = a.to(dev)
        except Exception as e:
            out["errors"].append(f"device {dev} unavailable, CPU fallback: {e}")
            dev = "cpu"
    ips_marks, temps = [], []
    with Monitor(interval_s=1.0) as mon:
        t_end, it, t0 = time.time() + duration, 0, time.time()
        while time.time() < t_end:
            tele = mon.samples[-1] if mon.samples else None
            if not guard.ok(tele):
                out["errors"].append(f"aborted: {guard.reason}")
                break
            b = time.time()
            for _ in range(10):
                _ = a @ a
                if dev == "cuda":
                    torch.cuda.synchronize()
            it += 10
            ips_marks.append((time.time(), it / (time.time() - t0)))
            if tele and tele.get("gpu_temp_c"):
                temps.append(tele["gpu_temp_c"])
        wall = time.time() - t0
    half = len(ips_marks) // 2
    first = np.mean([m[1] for m in ips_marks[:half]]) if half else 0
    second = np.mean([m[1] for m in ips_marks[half:]]) if ips_marks[half:] else 0
    degr = (first - second) / first if first else 0
    out["tests"]["sustained"] = {"seconds": round(wall, 1), "iters": it,
                                 "avg_ips": round(it / wall, 1) if wall else 0,
                                 "first_half_ips": round(float(first), 1),
                                 "second_half_ips": round(float(second), 1),
                                 "degradation_pct": round(float(degr) * 100, 1)}
    out["tests"]["throttling"] = ("LIKELY (>10% slowdown — check cooling/power)" if degr > 0.10
                                  else "none detected" if ips_marks else "unknown (no telemetry)")
    if temps:
        out["tests"]["gpu_temp_c"] = {"max": round(max(temps), 1), "mean": round(sum(temps) / len(temps), 1)}
    else:
        out["tests"]["gpu_temp_c"] = "unavailable (no temp sensor on this host)"
    out["telemetry"] = mon.summary()
    return out
