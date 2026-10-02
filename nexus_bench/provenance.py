"""Provenance manifest: every report carries its own reproducibility record."""
import hashlib
import json
import os
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

def _sha256_file(path, limit_mb=4096):
    """Full SHA-256 of the complete file (truncation would weaken identity)."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
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
    def _fp(key):
        p = (cfg.get(key) or "").strip()
        if not p:
            return "none"
        if os.path.isdir(p):
            h = hashlib.sha256()
            for f in sorted(os.listdir(p)):
                fp = os.path.join(p, f)
                if os.path.isfile(fp):
                    h.update(f.encode())
                    h.update(_sha256_file(fp).encode())
            return "dir:" + h.hexdigest()
        return _sha256_file(p)

    mp = (cfg.get("model") or "").strip()
    git = _git()
    try:
        from nexus_bench import release as R
        rel = R.read_release()
    except Exception:
        rel = None
    if rel:
        # Packaged distribution: release identity replaces git; a missing .git
        # is normal here and must never penalize a legitimate release.
        distribution = f"release {rel.get('qualification_version', '?')} " \
                       f"({rel.get('release_id', 'unidentified')})"
        source_state = "packaged release"
        git = {"sha": "n/a (packaged distribution)", "dirty": False,
               "root": "n/a (packaged distribution)"}
    else:
        distribution = "dev-git"
        source_state = ("UNRELEASED / DIRTY SOURCE — results not reproducible from "
                        "commit alone" if git.get("dirty") else "clean tree")
    return {
        "benchmark": "nexus-benchmarking",
        "benchmark_version": __import__("nexus_bench").__version__
        if hasattr(__import__("nexus_bench"), "__version__") else "unknown",
        "git": git,
        "distribution": distribution,
        "release": rel,
        "source_state": source_state,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": platform.python_version(),
        "platform": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "packages": mods,
        "fingerprints": {
            "model": _fp("model"),
            "reid_model": _fp("reid_model"),
            "video": _fp("video"),
            "uav_video": _fp("uav_video"),
            "rover_video": _fp("rover_video"),
            "image_dir": _fp("image_dir"),
            "val_data": _fp("val_data"),
        },
        "model_sha256": _sha256_file(mp) if mp else "none",
        "config_sha256": hashlib.sha256(
            json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest(),
        "gate_sha256": cfg.get("gate_cfg_sha256", "none"),
        "power": power,
    }
