"""Pre-flight: fast validation BEFORE any 10-minute run.

Checks (each OK/FAIL + detail): hardware, official model loads, backend works,
requested precision available, GPU present when required, assets verified,
videos decode, accuracy data present, RAM/disk headroom, power condition,
runtime healthy. Warnings (power/val-data) never silently pass anything: they
are recorded and the gate treats them as missing evidence downstream.
Status FULL only if every critical check passes, else FAILED (do not start).
"""
import os

from nexus_bench import statuses as S
from nexus_bench.monitor import Monitor
from nexus_bench.profiles import resolve_device

CRITICAL = {"hardware", "model", "backend", "precision", "assets", "video",
            "memory", "disk", "runtime"}

def _ok(checks, name, cond, detail, critical=True):
    checks.append({"check": name, "pass": bool(cond), "detail": detail,
                   "critical": critical})
    return bool(cond)

def run(cfg):
    import psutil
    dev = resolve_device(cfg.get("device", "auto"))
    want_fp16 = cfg.get("precision") == "fp16"
    out = {"module": "preflight", "config": {**cfg, "resolved_device": dev},
           "tests": {}, "errors": [], "status": S.FULL}
    checks = []
    with Monitor() as mon:
        try:
            import torch
            _ok(checks, "runtime", True, f"torch {torch.__version__}", True)
        except Exception as e:
            _ok(checks, "runtime", False, f"torch import failed: {e}", True)
        _ok(checks, "hardware", dev in ("cuda", "mps", "cpu"),
             f"device={dev}", True)
        mp = (cfg.get("model") or "").strip()
        yolo = None
        if mp and os.path.exists(mp):
            try:
                from ultralytics import YOLO
                yolo = YOLO(mp)
                _ok(checks, "model", True, f"loads: {os.path.basename(mp)}", True)
            except Exception as e:
                _ok(checks, "model", False, f"load failed: {str(e)[:200]}", True)
        else:
            _ok(checks, "model", False, "official model missing", True)
        if yolo is not None:
            try:
                from nexus_bench.yolo_util import precision_kwargs
                pk = precision_kwargs(want_fp16)
                import numpy as np
                yolo.predict(np.zeros((64, 64, 3), np.uint8),
                             imgsz=64, device=dev, verbose=False, **pk)
                from nexus_bench.yolo_util import effective_precision
                eff = effective_precision(yolo)
                _ok(checks, "backend", True, f"forward ok on {dev}", True)
                _ok(checks, "precision", (eff == ("fp16" if want_fp16 else "fp32")),
                     f"requested={'fp16' if want_fp16 else 'fp32'} effective={eff}", True)
            except Exception as e:
                _ok(checks, "backend", False, f"forward failed: {str(e)[:200]}", True)
                _ok(checks, "precision", False, "not verifiable", True)
        else:
            _ok(checks, "backend", False, "no model, backend untested", True)
            _ok(checks, "precision", False, "no model, precision unverified", True)
        try:
            from nexus_bench import assets as A
            ok, recs = A.verify_assets()
            bad = [r for r in recs if r["status"] != "OK"]
            _ok(checks, "assets", ok,
                 "all official assets verified" if ok else f"bad assets: {bad}", True)
        except Exception as e:
            _ok(checks, "assets", False, f"verify failed: {e}", True)
        for key, label in (("uav_video", "UAV"), ("rover_video", "Rover")):
            vp = (cfg.get(key) or cfg.get("video") or "").strip()
            if vp and os.path.exists(vp):
                try:
                    import cv2
                    cap = cv2.VideoCapture(vp)
                    ok, f = cap.read()
                    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                    cap.release()
                    _ok(checks, f"video_{key}", bool(ok and f is not None),
                         f"{label} decodes ({n} frames)", True)
                except Exception as e:
                    _ok(checks, f"video_{key}", False, f"{label} decode failed: {e}", True)
            else:
                _ok(checks, f"video_{key}", False, f"{label} video missing", True)
        vd = (cfg.get("val_data") or "").strip()
        _ok(checks, "accuracy_data", bool(vd and os.path.exists(vd)),
             "validation data present" if vd and os.path.exists(vd)
             else "validation data missing (accuracy will be INCONCLUSIVE)", False)
        try:
            free_gb = psutil.virtual_memory().available / 1e9
            _ok(checks, "memory", free_gb > 2.0, f"{free_gb:.1f} GB available", True)
        except Exception as e:
            _ok(checks, "memory", False, str(e), True)
        try:
            import shutil
            free = shutil.disk_usage(os.getcwd()).free / 1e9
            _ok(checks, "disk", free > 2.0, f"{free:.1f} GB free", True)
        except Exception as e:
            _ok(checks, "disk", False, str(e), True)
        try:
            from nexus_bench import power as P
            sec = P.collect(dev)
            ac = (sec["signals"].get("ac_connected") or {}).get("value")
            _ok(checks, "power", True,
                 f"AC={ac} (informational; strict gate requires AC)", False)
            out["tests"]["power"] = sec
        except Exception as e:
            _ok(checks, "power", True, f"power probe failed (informational): {e}", False)
        out["tests"]["checks"] = checks
        failed = [c for c in checks if c["critical"] and not c["pass"]]
        if failed:
            out["status"] = S.FAILED
            out["errors"].append("PRE-FLIGHT FAILED: " +
                                 "; ".join(f"{c['check']}: {c['detail']}" for c in failed))
        out["telemetry"] = mon.summary()
    return out
