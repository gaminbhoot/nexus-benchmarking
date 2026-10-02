"""AI inference: real YOLO (Ultralytics) over an imgsz x batch x precision grid.

Honesty rule: if no valid weights are found, this module reports UNSUPPORTED and
produces no latency numbers. It never substitutes a conv proxy for YOLO results.
"""
import os
import time
import numpy as np

from nexus_bench import statuses as S
from nexus_bench.monitor import Monitor
from nexus_bench.profiles import resolve_device
from nexus_bench.stats import summarize
from nexus_bench.yolo_util import check_effective, deployment_engine, load_weights, precision_kwargs

def _synthetic_batch(h, w, batch, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.integers(0, 255, (h, w, 3), dtype=np.uint8) for _ in range(batch)]

def _grid(cfg):
    imgszs = cfg.get("imgsz_list") or [cfg.get("imgsz", 640)]
    batches = cfg.get("batch_list") or [cfg.get("batch", 1)]
    return imgszs, batches

def run(cfg):
    dev = resolve_device(cfg.get("device", "auto"))
    model_path = (cfg.get("model") or "").strip()
    prec = cfg.get("precision", "fp32")
    imgszs, batches = _grid(cfg)
    out = {"module": "inference",
           "config": {**cfg, "resolved_device": dev, "imgsz_list": imgszs,
                      "batch_list": batches,
                      "deployment_engine": deployment_engine(dev)},
           "tests": {}, "errors": [], "status": S.FULL}
    if not model_path or not os.path.exists(model_path):
        out["status"] = S.UNSUPPORTED
        out["errors"].append(
            "no YOLO weights found (pass --model path/to/weights.pt); "
            "no inference numbers reported. Synthetic proxies are not a "
            "substitute for YOLO — see README methodology.")
        with Monitor() as mon:
            out["telemetry"] = mon.summary()
        return out
    with Monitor() as mon:
        model, prec_rec = load_weights(model_path, want_fp16=(prec == "fp16"))
        out["config"]["precision"] = prec_rec
        if model is None:
            out["status"] = S.UNSUPPORTED
            out["errors"].append(f"{prec_rec.get('error')}; no inference numbers reported. "
                                 "Synthetic proxies are not a substitute for YOLO.")
            out["telemetry"] = mon.summary()
            return out
        try:
            pk = precision_kwargs(prec == "fp16")
        except RuntimeError as e:
            out["status"] = S.FAILED
            out["errors"].append(f"precision adapter: {e}")
            out["telemetry"] = mon.summary()
            return out
        out["status"] = S.FULL
        w, r = cfg.get("warmup", 3), cfg.get("repeats", 20)
        for imgsz in imgszs:
            for bs in batches:
                key = f"yolo_imgsz{imgsz}_b{bs}_{prec}"
                try:
                    frames = _synthetic_batch(imgsz, imgsz, bs)
                    for _ in range(w):  # warmup with the SAME batched call shape
                        model.predict(frames, imgsz=imgsz, device=dev,
                                      verbose=False, **pk)
                    if not check_effective(model, prec_rec):
                        out["errors"].append(
                            f"{key} FAILEDclosed: requested {prec_rec['requested']}, "
                            f"effective {prec_rec['effective']} — numbers withheld")
                        continue
                    out["config"].setdefault("effective_precision", prec_rec["effective"])
                    lat, pre, post, ndet = [], [], [], []
                    iters = max(1, r // max(1, bs))
                    for i in range(iters):
                        t0 = time.perf_counter()
                        # ONE call with a list of bs images = a true batch of bs.
                        res = model.predict(frames, imgsz=imgsz, device=dev,
                                            verbose=False, **pk)
                        dt = (time.perf_counter() - t0) * 1000
                        lat.append(dt)
                        ndet.append(sum(len(x.boxes) if x.boxes is not None else 0
                                        for x in res))
                        try:
                            s = res[0].speed
                            pre.append(s.get("preprocess", 0))
                            post.append(s.get("postprocess", 0))
                        except Exception:
                            pass
                    d = summarize(lat)
                    # Per-image cost AND batch throughput: both are meaningful.
                    d["batch_size"] = bs
                    d["per_image_ms"] = round(d["mean_ms"] / bs, 3) if d["mean_ms"] else None
                    d["batch_ips"] = round(1000.0 / d["mean_ms"] * bs, 1) if d["mean_ms"] else None
                    if pre:
                        d["preprocess_mean_ms"] = round(sum(pre) / len(pre), 2)
                        d["postprocess_mean_ms"] = round(sum(post) / len(post), 2)
                    d["mean_detections_per_batch"] = round(sum(ndet) / len(ndet), 1)
                    out["tests"][key] = d
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        out["errors"].append(
                            f"OOM at {key} (recovered; smaller imgsz/batch fits — "
                            f"this IS the VRAM boundary signal): {str(e)[:200]}")
                        try:
                            import torch
                            if dev == "cuda":
                                torch.cuda.empty_cache()
                        except Exception:
                            pass
                    else:
                        out["errors"].append(f"{key}: {e}")
                except Exception as e:
                    out["errors"].append(f"{key}: {e}")
        out["telemetry"] = mon.summary()
    return out
