"""Explicit YOLO precision adapter. No inference probing, fail-closed semantics.

Modern ultralytics (>=8.3): predict/val take `quantize=` with 16/'fp16' for FP16,
32/'fp32'/None for FP32. Legacy `half=True` is a deprecated alias kept only for
old installs. The adapter selects the flag WITHOUT running inference, and after
warmup the caller MUST verify via `effective_precision()` — a requested fp16 that
is not actually in effect fails the measurement closed instead of reporting
fp32 numbers under an fp16 label.
"""
import numpy as np

MODERN_MIN = (8, 3, 0)

def ultralytics_version():
    try:
        import ultralytics
        parts = []
        for p in ultralytics.__version__.split("."):
            try:
                parts.append(int("".join(c for c in p if c.isdigit()) or 0))
            except ValueError:
                break
        return tuple(parts[:3])
    except Exception:
        return None

def deployment_engine(dev="auto"):
    """Exact execution engine identity: framework + provider + runtime versions.
    PyTorch CUDA and ONNX CUDA both run 'on cuda' — this string tells them apart."""
    import torch
    parts = [f"pytorch-{torch.__version__}"]
    try:
        if dev == "cuda" and torch.cuda.is_available():
            parts.append(f"provider=CUDA(cuda{torch.version.cuda or '?'},"
                         f"cudnn={torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'n/a'})")
        else:
            parts.append(f"provider={dev}")
    except Exception:
        parts.append(f"provider={dev}?")
    try:
        import tensorrt
        parts.append(f"tensorrt={getattr(tensorrt, '__version__', '?')}")
    except Exception:
        pass
    return " ".join(parts)

def precision_kwargs(want_fp16):
    """Flag for this install's ultralytics. Raises RuntimeError if unknown."""
    if not want_fp16:
        return {}
    ver = ultralytics_version()
    if ver is None:
        raise RuntimeError("ultralytics version undetectable; precision UNSUPPORTED")
    if ver >= MODERN_MIN:
        return {"quantize": "fp16"}
    return {"half": True}  # legacy alias path only

def effective_precision(model):
    """Post-warmup ground truth from the live predictor args. Never inferred."""
    try:
        args = model.predictor.args
        q = args.get("quantize", None) if hasattr(args, "get") else getattr(args, "quantize", None)
        if q in (16, "fp16"):
            return "fp16"
        if q in (32, "fp32", None):
            if hasattr(args, "get") and args.get("half", False):
                return "fp16"
            return "fp32"
        return f"other({q})"
    except Exception:
        return "unknown"

def load_weights(path, want_fp16=False):
    """Returns (model_or_None, record_dict). Record carries requested/effective
    (effective filled after warmup), backend version. Never raises."""
    rec = {"requested": "fp16" if want_fp16 else "fp32",
           "effective": "unverified", "backend": "ultralytics",
           "ultralytics_version": ".".join(map(str, ultralytics_version() or ())) or "unknown",
           "flag": None}
    if not path:
        return None, {**rec, "error": "no model path (pass --model for YOLO)"}
    import os
    if not os.path.exists(path):
        return None, {**rec, "error": f"model not found: {path}"}
    try:
        from ultralytics import YOLO
        model = YOLO(path)
    except Exception as e:
        return None, {**rec, "error": f"yolo load failed: {e}"}
    try:
        rec["flag"] = precision_kwargs(want_fp16)
    except RuntimeError as e:
        return None, {**rec, "error": str(e)}
    return model, rec

def check_effective(model, rec):
    """Fill rec['effective'] post-warmup. Returns True iff requested precision holds."""
    rec["effective"] = effective_precision(model)
    return rec["effective"] == rec["requested"]
