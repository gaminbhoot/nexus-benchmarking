"""Laptop power section: measured vs unavailable vs not-applicable, never invented.

Collects AC/battery state, GPU power + limit (NVML), CPU/system power only where
exposed. The report states coverage explicitly; power is QUALIFIED only when the
gate's required signals were all measured (see gate power policy).
"""
import subprocess

def collect(dev="auto"):
    sec = {"coverage": {}, "signals": {}}

    def put(name, value, how):
        sec["signals"][name] = {"value": value, "how": how}  # measured|unavailable|n/a

    try:
        import psutil
        bat = psutil.sensors_battery()
        if bat is not None:
            put("ac_connected", bool(bat.power_plugged), "measured")
            put("battery_pct", round(bat.percent, 1) if bat.percent is not None else None,
                "measured" if bat.percent is not None else "unavailable")
            put("battery_state", "charging/discharging"
                if bat.power_plugged is False else ("on AC" if bat.power_plugged else None),
                "measured" if bat.power_plugged is not None else "unavailable")
        else:
            put("ac_connected", None, "unavailable")
            put("battery_pct", None, "unavailable")
            put("battery_state", None, "unavailable")
    except Exception:
        put("ac_connected", None, "unavailable")
        put("battery_pct", None, "unavailable")
        put("battery_state", None, "unavailable")

    nv = _nvidia_power()
    put("gpu_power_w", nv.get("power_w"), "measured" if nv.get("power_w") is not None else "unavailable")
    put("gpu_power_limit_w", nv.get("power_limit_w"),
        "measured" if nv.get("power_limit_w") is not None else "unavailable")
    put("cpu_power_w", None, "unavailable (no portable CPU-power sensor probed)")
    put("system_power_w", None, "unavailable (no system power sensor probed)")
    put("performance_mode", _perf_mode(), "measured" if _perf_mode() else "unavailable")

    measured = [k for k, v in sec["signals"].items() if v["how"] == "measured"]
    sec["coverage"] = {"measured_signals": measured,
                       "power_qualified": False,
                       "note": ("power is QUALIFIED only when every gate-required power signal "
                                "is measured; GPU-only power never qualifies whole-system power")}
    return sec

def _nvidia_power():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=power.draw,power.limit",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10)
        if out.returncode == 0 and out.stdout.strip():
            p, lim = [x.strip() for x in out.stdout.strip().splitlines()[0].split(",")[:2]]
            def num(x):
                try:
                    return float(x)
                except ValueError:
                    return None
            return {"power_w": num(p), "power_limit_w": num(lim)}
    except Exception:
        pass
    return {}

def _perf_mode():
    # Best-effort OS power-mode detection; absence is reported, not guessed.
    import platform
    try:
        if platform.system() == "Darwin":
            return "unknown (macOS lowpowermode not probed)"
        import os
        gov = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"
        if os.path.exists(gov):
            with open(gov, encoding="utf-8") as f:
                return f"cpufreq:{f.read().strip()}"
    except Exception:
        pass
    return None
