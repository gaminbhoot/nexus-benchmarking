"""Deployment backends on the SAME workload: PyTorch fp32/fp16, ONNX CUDA, TensorRT.

Each backend runs only where its dependency exists; anything else is NOT_RUN
with a reason — never fake numbers, never conflated with PyTorch results.
INT8 tensor allocation is reported as allocator capability, NOT inference.
"""
import os
import time
import numpy as np

from nexus_bench import statuses as S
from nexus_bench.monitor import Monitor
from nexus_bench.profiles import resolve_device
from nexus_bench.stats import summarize
from nexus_bench.yolo_util import check_effective, load_weights

def _have(pkg):
    try:
        __import__(pkg)
        return True
    except Exception:
        return False

def _bench_torch(model, frames, imgsz, dev, pk, repeats, warmup):
    for _ in range(warmup):
        model.predict(frames, imgsz=imgsz, device=dev, verbose=False, **pk)
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        model.predict(frames, imgsz=imgsz, device=dev, verbose=False, **pk)
        ts.append((time.perf_counter() - t0) * 1000)
    return summarize(ts)

def run(cfg):
    import torch
    dev = resolve_device(cfg.get("device", "auto"))
    mp = (cfg.get("model") or "").strip()
    imgsz = cfg.get("imgsz", 640)
    out = {"module": "backends", "config": {**cfg, "resolved_device": dev},
           "tests": {}, "errors": [], "status": S.FULL}
    with Monitor() as mon:
        frames = [np.zeros((imgsz, imgsz, 3), dtype=np.uint8)]
        # --- PyTorch fp32 (baseline; always attempted when weights exist) ---
        if mp and os.path.exists(mp):
            try:
                from ultralytics import YOLO
                m32 = YOLO(mp)
                w, r = cfg.get("warmup", 3), max(5, cfg.get("repeats", 20))
                out["tests"]["torch_fp32_ms"] = _bench_torch(m32, frames, imgsz, dev, {}, r, w)
            except Exception as e:
                out["status"] = S.FAILED
                out["errors"].append(f"torch_fp32: {e}")
                out["telemetry"] = mon.summary()
                return out
            # --- PyTorch fp16 via explicit adapter, verified, fail-closed ---
            try:
                from nexus_bench.yolo_util import precision_kwargs
                m16, rec = load_weights(mp, want_fp16=True)
                pk = precision_kwargs(True)
                _bench_torch(m16, frames, imgsz, dev, pk, w, w)
                if check_effective(m16, rec):
                    out["tests"]["torch_fp16_ms"] = _bench_torch(m16, frames, imgsz, dev, pk, r, 1)
                    out["config"]["fp16_effective"] = True
                else:
                    out["errors"].append(f"torch_fp16 withheld: effective={rec['effective']}")
                    out["config"]["fp16_effective"] = False
            except Exception as e:
                out["errors"].append(f"torch_fp16 unavailable: {e}")
            # --- ONNX Runtime CUDA (export once, same frames) ---
            if _have("onnxruntime"):
                try:
                    import onnxruntime as ort
                    exp = m32.export(format="onnx", imgsz=imgsz, verbose=False)
                    sess = ort.InferenceSession(str(exp),
                                                providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
                    inp = sess.get_inputs()[0]
                    x = np.zeros((1, 3, imgsz, imgsz), dtype=np.float32)
                    for _ in range(w):
                        sess.run(None, {inp.name: x})
                    ts = []
                    for _ in range(r):
                        t0 = time.perf_counter()
                        sess.run(None, {inp.name: x})
                        ts.append((time.perf_counter() - t0) * 1000)
                    d = summarize(ts)
                    d["provider"] = sess.get_providers()[0] if sess.get_providers() else "unknown"
                    out["tests"]["onnx_ms"] = d
                except Exception as e:
                    out["errors"].append(f"onnx: {e}")
            else:
                out["tests"]["onnx"] = S.NOT_RUN
                out["errors"].append("NOT_RUN onnx: onnxruntime not installed")
            # --- TensorRT (export + engine inference; needs tensorrt + cuda) ---
            if _have("tensorrt") and dev == "cuda":
                for mode, kw in (("trt_fp16", {"half": True}), ("trt_int8", {"int8": True})):
                    try:
                        exp = m32.export(format="engine", imgsz=imgsz, verbose=False, **kw)
                        out["tests"][mode] = {"engine": str(exp),
                                              "note": "engine built; inference via exported YOLO wrapper"}
                        from ultralytics import YOLO as _Y
                        me = _Y(str(exp))
                        out["tests"][mode + "_ms"] = _bench_torch(me, frames, imgsz, dev, {}, r, w)
                    except Exception as e:
                        out["errors"].append(f"{mode} unavailable: {str(e)[:200]}")
                        if mode == "trt_fp16":
                            out["tests"][mode] = S.NOT_RUN
            else:
                out["tests"]["tensorrt"] = S.NOT_RUN
                out["errors"].append("NOT_RUN tensorrt: needs tensorrt package + CUDA device")
        else:
            out["status"] = S.UNSUPPORTED
            out["errors"].append("no YOLO weights; backend matrix needs a model")
        # --- allocator capability (explicitly NOT inference) ---
        try:
            if dev == "cuda" and torch.cuda.is_available():
                q = torch.randint(-128, 127, (512, 512), dtype=torch.int8, device="cuda")
                out["tests"]["int8_tensor_alloc"] = True
                del q
            else:
                out["tests"]["int8_tensor_alloc"] = "n/a (non-CUDA)"
        except Exception as e:
            out["tests"]["int8_tensor_alloc"] = f"failed: {e}"
        out["telemetry"] = mon.summary()
    return out
