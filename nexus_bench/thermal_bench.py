"""Thermal stability: windowed sustained load with evidence-based verdict.

Load: requested YOLO model (fail-closed: load failure = FAILED, never a silent
GEMM substitution) or an explicitly-labelled GEMM stress when no model is given.
Throughput is measured in fixed 5 s windows (burst vs steady-state); throttling
is claimed ONLY on temp + clock-drop-under-load + flags evidence.
"""
import time
import numpy as np

from nexus_bench import statuses as S
from nexus_bench.monitor import Monitor
from nexus_bench.profiles import resolve_device
from nexus_bench.safety import Guard, Limits
from nexus_bench.stats import summarize
from nexus_bench.yolo_util import check_effective, load_weights, precision_kwargs

WINDOW_S = 5.0

def _load(cfg, dev):
    import os
    mp = (cfg.get("model") or "").strip()
    imgsz = cfg.get("imgsz", 640)  # deployment resolution from the profile, never hardcoded
    if mp:
        want_fp16 = cfg.get("precision") == "fp16"
        yolo, prec_rec = load_weights(mp, want_fp16=want_fp16)
        if yolo is None:
            raise RuntimeError(f"requested model unusable: {prec_rec.get('error')}")
        pk = precision_kwargs(want_fp16)
        frames = [np.zeros((imgsz, imgsz, 3), dtype=np.uint8)]
        yolo.predict(frames, imgsz=imgsz, device=dev, verbose=False, **pk)
        if not check_effective(yolo, prec_rec):
            raise RuntimeError(f"precision fail-closed: requested {prec_rec['requested']}, "
                               f"effective {prec_rec['effective']}")
        def step():
            yolo.predict(frames, imgsz=imgsz, device=dev, verbose=False, **pk)
        return step, f"yolo ({mp} @ {prec_rec['effective']})"
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
    return step, f"gemm-{n} x4/iter on {dev} (explicit stress, no model requested)"

def run(cfg):
    dev = resolve_device(cfg.get("device", "auto"))
    duration = min(cfg.get("duration_s", 60), 1200)
    guard = Guard(Limits(max_duration_s=duration + 60))
    out = {"module": "thermal",
           "config": {**cfg, "resolved_device": dev, "sustained_s": duration,
                      "window_s": WINDOW_S},
           "tests": {}, "errors": [], "status": S.FULL}
    try:
        step, tag = _load(cfg, dev)
    except RuntimeError as e:
        out["status"] = S.FAILED
        out["errors"].append(str(e))
        with Monitor() as mon:
            out["telemetry"] = mon.summary()
        return out
    out["config"]["load"] = tag
    windows, cur, w0 = [], 0, time.time()
    win_telemetry = []
    with Monitor(interval_s=1.0) as mon:
        it, t0 = 0, time.time()
        t_end = t0 + duration
        while time.time() < t_end:
            tele = mon.samples[-1] if mon.samples else None
            if S.stop_requested():
                out["errors"].append("stopped by user (STOP file or Ctrl+C)")
                out["status"] = S.ABORTED
                break
            if not guard.ok(tele):
                out["errors"].append(f"aborted: {guard.reason}")
                out["status"] = S.ABORTED
                break
            step()
            it += 1
            now = time.time()
            if now - w0 >= WINDOW_S:
                windows.append(round((it - cur) / (now - w0), 2))
                cur, w0 = it, now
                snap = dict(tele or {})
                win_telemetry.append({k: snap.get(k) for k in
                                      ("gpu_temp_c", "gpu_clock_mhz", "gpu_power_w",
                                       "gpu_util_pct", "cpu_temp_c")})
        wall = time.time() - t0
    if windows:
        import statistics as _st
        # Temporal order PRESERVED: burst = first window, steady = the rest.
        burst = windows[0]
        rest = windows[1:] or windows
        steady_med = _st.median(rest)
        out["tests"]["windows_ips"] = [round(w, 2) for w in windows]
        out["tests"]["steady"] = {
            "burst_ips": round(burst, 2), "steady_median_ips": round(steady_med, 2),
            "min_window_ips": round(min(windows), 2),
            "final_window_ips": round(windows[-1], 2),
            "burst_steady_ratio": round(burst / steady_med, 3) if steady_med else None,
            "degradation_pct": round(100 * (burst - steady_med) / burst, 1) if burst else 0}
        out["tests"]["window_telemetry"] = win_telemetry
    out["tests"]["sustained"] = {"seconds": round(wall, 1), "iters": it,
                                 "avg_ips": round(it / wall, 2) if wall else 0}
    temps = [w.get("gpu_temp_c") for w in win_telemetry if w.get("gpu_temp_c") is not None]
    clocks = [w.get("gpu_clock_mhz") for w in win_telemetry if w.get("gpu_clock_mhz") is not None]
    utils = [w.get("gpu_util_pct") for w in win_telemetry if w.get("gpu_util_pct") is not None]
    verdict, evidence = "unknown", "insufficient telemetry (no GPU temp/clock sensors)"
    if temps and clocks and len(windows) >= 2:
        early_c = sum(clocks[:2]) / 2
        late_c = sum(clocks[-2:]) / 2
        clock_drop = (early_c - late_c) / early_c if early_c else 0
        hot = max(temps) >= (cfg.get("throttle_temp_c", 83) - 5)
        busy = (sum(utils) / len(utils) > 50) if utils else False
        ev = (f"max_temp={max(temps):.0f}C clock {early_c:.0f}->{late_c:.0f}MHz "
              f"({clock_drop * 100:.0f}% drop) util_high={busy} windows={len(windows)}")
        degr = out["tests"]["steady"]["degradation_pct"] / 100
        if hot and clock_drop > 0.10 and busy:
            verdict, evidence = "likely", ev + " — temp high with clock drop under load"
        elif degr > 0.10:
            verdict, evidence = "slowdown_cause_unknown", ev + " — check power/scheduling"
        else:
            verdict, evidence = "none_detected", ev
    elif windows and out["tests"]["steady"]["degradation_pct"] > 10:
        verdict, evidence = "slowdown_cause_unknown", "no temp/clock telemetry to attribute it"
    elif windows:
        verdict, evidence = "none_detected", "no windowed degradation"
    out["tests"]["throttling"] = verdict
    out["tests"]["throttling_evidence"] = evidence
    out["telemetry"] = mon.summary()
    return out
