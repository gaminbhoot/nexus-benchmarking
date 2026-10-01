"""Shared YOLO helpers: version-robust precision flag + single load path."""
import numpy as np

_FLAG = None

def precision_kwargs(model, want_fp16):
    """FP16 flag that works across ultralytics versions (`quantize` in new,
    `half` in old). Probed once per process; weights cast is handled by callers."""
    global _FLAG
    if not want_fp16:
        return {}
    if _FLAG is None:
        try:
            model.predict(np.zeros((32, 32, 3), np.uint8), verbose=False, quantize=True)
            _FLAG = "quantize"
        except TypeError:
            _FLAG = "half"
    return {_FLAG: True}

def load_weights(path, want_fp16=False):
    """Returns (model_or_None, tag). Never raises."""
    if not path:
        return None, "motion_detector_fallback (labelled; pass --model for YOLO)"
    import os
    if not os.path.exists(path):
        return None, f"model not found: {path}"
    try:
        from ultralytics import YOLO
        model = YOLO(path)
        flag = "fp32"
        if want_fp16:
            try:
                model.model.half()
                flag = "fp16 (weights cast + precision flag)"
            except Exception as e:
                try:
                    model.model.float()
                except Exception:
                    pass
                flag = f"fp16 unsupported, fp32 ({str(e)[:100]})"
        return model, f"yolo ({path}, {flag})"
    except Exception as e:
        return None, f"yolo load failed: {e}"
