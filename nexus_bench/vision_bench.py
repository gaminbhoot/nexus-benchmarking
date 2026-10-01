"""Computer vision: real detect->track on frames + optical flow.

Detection source is reported honestly: YOLO weights when provided, otherwise a
motion-based foreground detector (`motion_detector_fallback` — measures the
tracking pipeline, not detection quality). Association always runs the
DeepSORT-style tracker with real embedding inference.
"""
import time
import numpy as np

from nexus_bench.monitor import Monitor
from nexus_bench.profiles import resolve_device
from nexus_bench.reid_embed import embed_crops, load_embedder
from nexus_bench.stats import summarize
from nexus_bench.tracking_bench import Tracker, _crop_for
from nexus_bench.yolo_util import load_weights, precision_kwargs

def _moving_squares(n=20, h=480, w=640, k=4, seed=0):
    rng = np.random.default_rng(seed)
    P = rng.uniform([20, 20], [w - 100, h - 100], (k, 2))
    V = rng.uniform(-4, 4, (k, 2))
    S = rng.uniform(40, 70, k)
    frames = []
    for f in range(n):
        img = np.zeros((h, w, 3), dtype=np.uint8)
        for i in range(k):
            x, y = int(P[i][0] + V[i][0] * f) % (w - 80), int(P[i][1] + V[i][1] * f) % (h - 80)
            img[y:y + int(S[i]), x:x + int(S[i])] = 200
        frames.append(img)
    return frames

def _motion_detections(prev_gray, gray):
    import cv2
    diff = cv2.absdiff(prev_gray, gray)
    _, th = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    th = cv2.dilate(th, None, iterations=2)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w * h > 400:
            boxes.append([float(x), float(y), float(w), float(h)])
    return boxes[:16]

def run(cfg):
    import cv2
    dev = resolve_device(cfg.get("device", "auto"))
    n = max(10, cfg.get("repeats", 20))
    frames = _moving_squares(n, seed=cfg.get("seed", 0))
    out = {"module": "vision", "config": {**cfg, "resolved_device": dev},
           "tests": {}, "errors": []}
    with Monitor() as mon:
        # --- detection source ---
        yolo, dtag = load_weights((cfg.get("model") or "").strip(),
                                  want_fp16=(cfg.get("precision") == "fp16"))
        pk = precision_kwargs(yolo, cfg.get("precision") == "fp16") if yolo else {}
        if yolo is None:
            # Informational only (stays in config, NOT errors): the tracking
            # pipeline still ran for real; only the detection source is synthetic.
            out["config"]["detector"] = ("motion_detector_fallback (tracks pipeline cost only, "
                                         "not detection quality)")
        else:
            out["config"]["detector"] = dtag
        try:
            model, tag = load_embedder(dev, (cfg.get("reid_model") or "").strip())
            out["config"]["embedder"] = tag
            tr = Tracker()
            det_ts, assoc_ts, emb_ts = [], [], []
            prev = cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY)
            gid = 0
            for f in frames[1:]:
                gray = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
                t0 = time.perf_counter()
                if yolo is not None:
                    r = yolo.predict(f, imgsz=cfg.get("imgsz", 640),
                                     device=dev, verbose=False, **pk)[0]
                    boxes = []
                    if r.boxes is not None:
                        for b in r.boxes.xywh.cpu().numpy():
                            x, y, w, h = b
                            boxes.append([x - w / 2, y - h / 2, w, h])
                else:
                    boxes = _motion_detections(prev, gray)
                det_ts.append((time.perf_counter() - t0) * 1000)
                crops = []
                for b in boxes[:8]:
                    x, y, w, h = [int(v) for v in b]
                    crop = f[max(y, 0):y + h, max(x, 0):x + w]
                    crops.append(crop if crop.size else _crop_for(None, gid))
                    gid += 1
                t0 = time.perf_counter()
                feats = list(embed_crops(model, crops, dev)[0]) if crops else []
                emb_ts.append((time.perf_counter() - t0) * 1000)
                t0 = time.perf_counter()
                live = tr.step(boxes[:8], feats)
                assoc_ts.append((time.perf_counter() - t0) * 1000)
                prev = gray
            d = summarize(det_ts); d["source"] = out["config"]["detector"]
            out["tests"]["detection_per_frame_ms"] = d
            out["tests"]["embedding_per_frame_ms"] = summarize(emb_ts)
            a = summarize(assoc_ts); a["confirmed_tracks"] = len(live)
            out["tests"]["association_per_frame_ms"] = a
        except Exception as e:
            out["errors"].append(f"detect_track: {e}")
        try:
            ts = []
            prev = cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY)
            for f in frames[1:]:
                gray = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
                t0 = time.perf_counter()
                _ = cv2.calcOpticalFlowFarneback(prev, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                ts.append((time.perf_counter() - t0) * 1000)
                prev = gray
            out["tests"]["optical_flow_ms"] = summarize(ts)
        except Exception as e:
            out["errors"].append(f"optflow: {e}")
        out["telemetry"] = mon.summary()
    return out
