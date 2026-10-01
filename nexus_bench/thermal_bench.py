"""Thermal stability: device-sized sustained load with evidence-based verdict.

Throttling is claimed ONLY when telemetry shows the conjunction:
high temp AND (clock drop while utilization stays high, or a driver throttle flag).
A bare slowdown is reported as a slowdown with its evidence, not as throttling.
"""
import time
import numpy as np

from nexus_bench.monitor import Monitor
from nexus_bench.profiles import resolve_device
from nexus_bench.safety import Guard, Limits

def _load(cfg, dev):
    """Sustained load sized to actually occupy the device. Returns (step_fn, tag)."""
    import os
    mp = (cfg.get("model") or "").strip()
    if mp and os.path.exists(mp):
        try:
            from ultralytics import YOLO
            y = YOLO(mp)
            frames = [np.zeros((640, 640, 3), dtype=np.uint8)]
            def step():
                y.predict(frames, imgsz=640, device=dev, verbose=False)
            return step, f"yolo ({mp}) sustained inference"
        except Exception:
            pass
    import torch
    n = 2048 if dev == "cuda" else 1024
    a = torch.randn(n, n)
    if dev in ("cuda", "mps"):
        try:
            a = a.to(dev)
        except Exception:
            dev = "cpu"
    def step():
        for _ in range(4):
            _ = a @ a
        if dev == "cuda":
            torch.cuda.synchronize()
    return step, f"gemm-{n} x4/iter on {dev}"

def run(cfg):
    import numpy as _np  # local alias; numpy already a hard dep
    dev = resolve_device(cfg.get("device", "auto"))
    duration = min(cfg.get("duration_s", 60), 600)
    guard = Guard(Limits(max_duration_s=duration + 30))
    out = {"module": "thermal",
           "config": {**cfg, "resolved_device": dev, "sustained_s": duration},
           "tests": {}, "errors": []}
    step, tag = _load(cfg, dev)
    out["config"]["load"] = tag
    series = []
    with Monitor(interval_s=1.0) as mon:
        it, t0 = 0, time.time()
        t_end = t0 + duration
        while time.time() < t_end:
            tele = mon.samples[-1] if mon.samples else None
            if not guard.ok(tele):
                out["errors"].append(f"aborted: {guard.reason}")
                break
            b = time.time()
            step()
            it += 1
            series.append({"t": round(time.time() - t0, 1),
                           "cum_ips": round(it / (time.time() - t0), 2),
                           **{k: (tele or {}).get(k) for k in
                              ("gpu_temp_c", "gpu_clock_mhz", "gpu_power_w",
                               "gpu_util_pct", "cpu_temp_c", "cpu_mhz")}})
        wall = time.time() - t0
    ips = [s["cum_ips"] for s in series]
    half = len(ips) // 2
    first = float(_np.mean(ips[:half])) if half else 0
    second = float(_np.mean(ips[half:])) if ips[half:] else 0
    degr = (first - second) / first if first else 0
    out["tests"]["sustained"] = {"seconds": round(wall, 1), "iters": it,
                                 "avg_ips": round(it / wall, 2) if wall else 0,
                                 "first_half_ips": round(first, 2),
                                 "second_half_ips": round(second, 2),
                                 "degradation_pct": round(degr * 100, 1)}
    out["tests"]["series"] = series[::max(1, len(series) // 60)]  # cap stored points
    # --- evidence-based verdict ---
    temps = [s["gpu_temp_c"] for s in series if s.get("gpu_temp_c") is not None]
    clocks = [s["gpu_clock_mhz"] for s in series if s.get("gpu_clock_mhz") is not None]
    utils = [s["gpu_util_pct"] for s in series if s.get("gpu_util_pct") is not None]
    flags = [s.get("throttle_flags") for s in series if s.get("throttle_flags")]
    verdict, evidence = "unknown", "insufficient telemetry (no GPU temp/clock sensors on this host)"
    if temps and clocks:
        early_c, late_c = _np.mean(clocks[:5]), _np.mean(clocks[-5:])
        clock_drop = (early_c - late_c) / early_c if early_c else 0
        hot = max(temps) >= (cfg.get("throttle_temp_c", 83) - 5)
        busy = (sum(utils) / len(utils) > 50) if utils else False
        ev = (f"max_temp={max(temps):.0f}C clock {early_c:.0f}->{late_c:.0f}MHz "
              f"({clock_drop * 100:.0f}% drop) util_high={busy} flags={set(flags) if flags else None}")
        if hot and clock_drop > 0.10 and busy:
            verdict, evidence = "likely", ev + " — temp high with clock drop under load"
        elif degr > 0.10:
            verdict, evidence = "slowdown without thermal evidence", ev + " — check power caps/scheduling, not temperature"
        else:
            verdict, evidence = "none detected", ev
    elif degr > 0.10:
        verdict, evidence = "slowdown, cause unknown", "no temp/clock telemetry to attribute it"
    elif series:
        verdict, evidence = "none detected", "no performance degradation over the window"
    out["tests"]["throttling"] = verdict
    out["tests"]["throttling_evidence"] = evidence
    out["telemetry"] = mon.summary()
    return out
