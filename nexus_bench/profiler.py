"""System profiler: detect CPU/GPU/RAM/VRAM/storage/OS/drivers. Never hardcodes hardware."""
import os
import platform
import shutil
import subprocess

def _try(fn, default="unavailable"):
    try:
        return fn()
    except Exception:
        return default

def _cpu_brand():
    try:
        if os.path.exists("/proc/cpuinfo"):
            with open("/proc/cpuinfo", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
    except Exception:
        pass
    if platform.system() == "Darwin":
        try:
            out = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                 capture_output=True, text=True, timeout=5)
            if out.returncode == 0 and out.stdout.strip():
                return out.stdout.strip()
        except Exception:
            pass
    return platform.processor() or "unavailable"

def _smi(fields):
    try:
        out = subprocess.run(["nvidia-smi", f"--query-gpu={fields}",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10)
        if out.returncode == 0 and out.stdout.strip():
            return [p.strip() for p in out.stdout.strip().splitlines()[0].split(",")]
    except Exception:
        pass
    return None

def _system_model():
    """Exact machine identity where the OS exposes it; 'unavailable' otherwise."""
    if platform.system() == "Darwin":
        try:
            out = subprocess.run(["sysctl", "-n", "hw.model"],
                                 capture_output=True, text=True, timeout=5)
            if out.returncode == 0 and out.stdout.strip():
                return out.stdout.strip()
        except Exception:
            pass
    try:
        if os.path.exists("/sys/devices/virtual/dmi/id/product_name"):
            with open("/sys/devices/virtual/dmi/id/product_name", encoding="utf-8") as f:
                prod = f.read().strip()
            with open("/sys/devices/virtual/dmi/id/sys_vendor", encoding="utf-8") as f:
                vend = f.read().strip()
            return f"{vend} {prod}".strip()
    except Exception:
        pass
    return "unavailable"

def profile():
    import psutil
    info = {
        "os": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "kernel": _try(lambda: platform.version()[:120]),
        "system_model": _system_model(),
        "python": platform.python_version(),
        "cpu": {},
        "memory": {},
        "storage": {},
        "gpu": {},
        "accelerators": {},
    }
    freq = None
    try:
        f = psutil.cpu_freq()
        freq = round(f.current, 1) if f and f.current else "unavailable"
    except Exception:
        freq = "unavailable"
    info["cpu"] = {"brand": _cpu_brand(),
                   "physical_cores": _try(lambda: psutil.cpu_count(logical=False)),
                   "logical_cores": _try(lambda: psutil.cpu_count(logical=True)),
                   "freq_mhz": freq}
    vm = psutil.virtual_memory()
    info["memory"] = {"ram_total_gb": round(vm.total / 1e9, 2),
                      "swap_total_gb": round(psutil.swap_memory().total / 1e9, 2)}
    du = shutil.disk_usage(os.getcwd())
    info["storage"] = {"total_gb": round(du.total / 1e9, 1),
                       "free_gb": round(du.free / 1e9, 1)}
    parts = _smi("name,driver_version,memory.total,compute_cap,power.limit")
    if parts and len(parts) >= 4:
        info["gpu"] = {"vendor": "nvidia", "name": parts[0], "driver": parts[1],
                       "vram_total": parts[2] + " MiB", "compute_cap": parts[3]}
        info["gpu"]["power_limit_w"] = parts[4] if len(parts) > 4 else "unavailable"
        cur = _smi("power.draw,clocks.gr,clocks.mem")
        if cur and len(cur) >= 3:
            info["gpu"]["power_now_w"] = cur[0]
            info["gpu"]["clock_gr_mhz"] = cur[1]
            info["gpu"]["clock_mem_mhz"] = cur[2]
    else:
        info["gpu"] = {"vendor": "unavailable (no nvidia-smi)",
                       "note": "non-NVIDIA or driverless host — GPU fields unavailable, not zero"}
    import torch
    info["accelerators"]["torch"] = torch.__version__
    info["accelerators"]["cuda_available"] = torch.cuda.is_available()
    info["accelerators"]["cuda_runtime"] = _try(lambda: torch.version.cuda or "unavailable")
    info["accelerators"]["cudnn"] = _try(
        lambda: str(torch.backends.cudnn.version() or "unavailable")
        if torch.backends.cudnn.is_available() else "unavailable")
    try:
        import tensorrt  # noqa
        info["accelerators"]["tensorrt"] = getattr(tensorrt, "__version__", "installed")
    except Exception:
        info["accelerators"]["tensorrt"] = "missing"
    info["accelerators"]["mps_available"] = _try(torch.backends.mps.is_available, False)
    if torch.cuda.is_available():
        try:
            info["accelerators"]["cuda_device"] = torch.cuda.get_device_name(0)
        except Exception:
            info["accelerators"]["cuda_device"] = "unavailable"
    info["packages"] = {}
    for pkg in ("ultralytics", "numpy", "pandas", "matplotlib", "yaml"):
        try:
            m = __import__(pkg)
            info["packages"][pkg] = getattr(m, "__version__", "installed")
        except Exception:
            info["packages"][pkg] = "missing"
    try:
        import cv2
        info["packages"]["opencv"] = cv2.__version__
    except Exception:
        info["packages"]["opencv"] = "missing"
    return info
