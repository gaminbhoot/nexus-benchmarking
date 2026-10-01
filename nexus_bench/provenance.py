"""Provenance manifest: every report carries its own reproducibility record."""
import hashlib
import json
import platform
import subprocess
import time

def _git():
    try:
        root = subprocess.run(["git", "rev-parse", "--show-toplevel"],
                              capture_output=True, text=True, timeout=10)
        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True, timeout=10)
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                               text=True, timeout=10)
        return {"sha": (sha.stdout.strip() if sha.returncode == 0 else "unknown"),
                "dirty": bool(dirty.stdout.strip()) if dirty.returncode == 0 else "unknown",
                "root": (root.stdout.strip() if root.returncode == 0 else "unknown")}
    except Exception:
        return {"sha": "unknown", "dirty": "unknown", "root": "unknown"}

def _sha256_file(path, limit_mb=512):
    try:
        h = hashlib.sha256()
        n = 0
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
                n += 1
                if n > limit_mb:
                    break
        return h.hexdigest()[:16]
    except Exception:
        return "unknown"

def manifest(cfg):
    import psutil
    mods = {}
    for pkg, imp in (("torch", "torch"), ("ultralytics", "ultralytics"),
                     ("opencv", "cv2"), ("numpy", "numpy"),
                     ("pandas", "pandas"), ("tensorrt", "tensorrt"),
                     ("onnxruntime", "onnxruntime")):
        try:
            m = __import__(imp)
            mods[pkg] = getattr(m, "__version__", "installed")
        except Exception:
            mods[pkg] = "missing"
    try:
        bat = psutil.sensors_battery()
        power = {"ac": bool(bat.power_plugged) if bat else "unknown",
                 "battery_pct": round(bat.percent, 1) if bat and bat.percent is not None else "unknown"}
    except Exception:
        power = {"ac": "unknown", "battery_pct": "unknown"}
    mp = (cfg.get("model") or "").strip()
    return {
        "benchmark": "nexus-benchmarking",
        "benchmark_version": __import__("nexus_bench").__version__
        if hasattr(__import__("nexus_bench"), "__version__") else "unknown",
        "git": _git(),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": platform.python_version(),
        "platform": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "packages": mods,
        "model_sha256": _sha256_file(mp) if mp else "none",
        "config_sha256": hashlib.sha256(
            json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:16],
        "power": power,
    }
