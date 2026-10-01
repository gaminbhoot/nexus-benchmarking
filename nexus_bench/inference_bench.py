"""AI inference: YOLO (Ultralytics) at imgsz/batch/precision grid. Synthetic fallback if no weights."""
import os
import time
import numpy as np

from nexus_bench.monitor import Monitor
from nexus_bench.profiles import resolve_device
from nexus_bench.stats import summarize

def _synthetic_input(h, w, batch=1):
    rng = np.random.default_rng(0)
    return [(rng.integers(0, 255, (h, w, 3), dtype=np.uint8)) for _ in range(batch)]

def run(cfg):
    dev = resolve_device(cfg.get("device", "auto"))
    model_path = cfg.get("model") or ""
    imgszs = [cfg.get("imgsz", 640)] if cfg.get("imgsz") else [320, 640]
    batches = [cfg.get("batch", 1)]
    prec = cfg.get("precision", "fp32")
    out = {"module": "inference", "config": {**cfg, "resolved_device": dev},
           "tests": {}, "errors": []}
    with Monitor() as mon:
        if model_path and os.path.exists(model_path):
            try:
                from ultralytics import YOLO
                model = YOLO(model_path)
                for imgsz in imgszs:
                    for bs in batches:
                        frames = _synthetic_input(imgsz, imgsz, bs)
                        for _ in range(cfg.get("warmup", 3)):
                            model.predict(frames, imgsz=imgsz, device=dev, verbose=False)
                        pre, infer, post = [], [], []
                        for f in frames * max(1, cfg.get("repeats", 20) // max(1, bs)):
                            t0 = time.perf_counter()
                            r = model.predict(f, imgsz=imgsz, device=dev, verbose=False)
                            infer.append((time.perf_counter() - t0) * 1000)
                            try:
                                s = r[0].speed
                                pre.append(s.get("preprocess", 0)); post.append(s.get("postprocess", 0))
                            except Exception:
                                pass
                        key = f"yolo_imgsz{imgsz}_b{bs}_{prec}"
                        d = summarize(infer)
                        if pre:
                            d["preprocess_mean_ms"] = round(sum(pre) / len(pre), 2)
                            d["postprocess_mean_ms"] = round(sum(post) / len(post), 2)
                        out["tests"][key] = d
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    out["errors"].append(f"OOM at {model_path} (recovered, try smaller imgsz/batch): {e}")
                    try:
                        import torch
                        if dev == "cuda":
                            torch.cuda.empty_cache()
                    except Exception:
                        pass
                else:
                    out["errors"].append(str(e))
            except Exception as e:
                out["errors"].append(f"yolo failed, synthetic fallback: {e}")
        if not out["tests"]:
            try:  # torch conv proxy so the grid still yields numbers without weights
                import torch
                import torch.nn as nn
                c = nn.Conv2d(3, 16, 3, padding=1).to(dev).eval()
                for imgsz in imgszs:
                    x = torch.randn(1, 3, imgsz // 4, imgsz // 4, device=dev)
                    for _ in range(cfg.get("warmup", 3)):
                        c(x)
                    ts = []
                    for _ in range(cfg.get("repeats", 20)):
                        t0 = time.perf_counter()
                        with torch.no_grad():
                            c(x)
                        if dev == "cuda":
                            torch.cuda.synchronize()
                        ts.append((time.perf_counter() - t0) * 1000)
                    out["tests"][f"synthetic_conv_proxy_imgsz{imgsz}"] = summarize(ts)
                out["errors"].append("no model weights provided/found; synthetic proxy used")
            except Exception as e:
                out["errors"].append(f"synthetic fallback failed: {e}")
        out["telemetry"] = mon.summary()
    return out
