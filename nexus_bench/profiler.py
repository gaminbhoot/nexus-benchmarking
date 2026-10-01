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

def _nvidia_smi():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,compute_cap",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
        if out.returncode == 0 and out.stdout.strip():
            parts = [p.strip() for p in out.stdout.strip().splitlines()[0].split(",")]
            return {"name": parts[0] if len(parts) > 0 else "unknown",
                    "driver": parts[1] if len(parts) > 1 else "unknown",
                    "vram_total": parts[2] if len(parts) > 2 else "unknown",
                    "compute_cap": parts[3] if len(parts) > 3 else "unknown"}
    except Exception:
        pass
    return None

def profile():
    import psutil
    info = {
        "os": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "python": platform.python_version(),
        "cpu": {},
        "memory": {},
        "storage": {},
        "gpu": {},
        "accelerators": {},
    }
    info["cpu"] = {
        "model": _try(lambda: platform.processor() or "unknown"),
        "physical_cores": _try(psutil.cpu_count(logical=False)),
        "logical_cores": _try(psutil.cpu_count(logical=True)),
        "freq_mhz": _try(lambda: round((psutil.cpu_freq() or {}).current or 0, 1)),
    }
    vm = psutil.virtual_memory()
    info["memory"] = {"ram_total_gb": round(vm.total / 1e9, 2),
                      "swap_total_gb": round(psutil.swap_memory().total / 1e9, 2)}
    du = shutil.disk_usage(os.getcwd())
    info["storage"] = {"total_gb": round(du.total / 1e9, 1),
                       "free_gb": round(du.free / 1e9, 1)}
    smi = _nvidia_smi()
    if smi:
        info["gpu"] = {"vendor": "nvidia", **smi}
    else:
        info["gpu"] = {"vendor": "unavailable (no nvidia-smi)", "note": "non-NVIDIA or driverless host"}
    import torch
    info["accelerators"]["torch"] = torch.__version__
    info["accelerators"]["cuda_available"] = torch.cuda.is_available()
    info["accelerators"]["cuda_version"] = _try(lambda: torch.version.cuda or "unavailable")
    info["accelerators"]["mps_available"] = _try(torch.backends.mps.is_available, False)
    if torch.cuda.is_available():
        try:
            info["accelerators"]["cuda_device"] = torch.cuda.get_device_name(0)
        except Exception:
            info["accelerators"]["cuda_device"] = "unavailable"
    info["packages"] = {}
    for pkg in ("ultralytics", "cv2", "numpy", "pandas", "matplotlib"):
        try:
            m = __import__(pkg)
            info["packages"][pkg] = getattr(m, "__version__", "installed")
        except Exception:
            info["packages"][pkg] = "missing"
    return info
