"""End-to-end NEXUS perception pipeline on a REAL frame stream.

Chain: source read -> decode -> resize -> detection -> embedding -> association
-> telemetry encode -> output. Reports throughput FPS, end-to-end latency
(median/p95/p99), deadline-miss % at --req-fps, and dropped frames under a
real-time consumption model (a slow stage drops later frames, like production).

Sources: --video FILE, --image-dir DIR, or seeded synthetic stream (labelled).
"""
import os
import time
import numpy as np

from nexus_bench.monitor import Monitor
from nexus_bench.profiles import resolve_device
from nexus_bench.reid_embed import embed_crops, load_embedder
from nexus_bench.stats import summarize
from nexus_bench.tracking_bench import Tracker
from nexus_bench.vision_bench import _motion_detections
from nexus_bench.yolo_util import load_weights, precision_kwargs

def _open_source(cfg):
    import cv2
    vp = (cfg.get("video") or "").strip()
    dp = (cfg.get("image_dir") or "").strip()
    if vp and os.path.exists(vp):
        cap = cv2.VideoCapture(vp)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        tag = f"video:{vp}@{fps:.1f}fps"
        def gen():
            while True:
                ok, f = cap.read()
                if not ok:
                    break
                yield cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            cap.release()
        return gen(), tag, fps
    if dp and os.path.isdir(dp):
        files = sorted(p for p in os.listdir(dp)
                       if p.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")))
        tag = f"image_dir:{dp} ({len(files)} files, replayed @30fps)"
        def gen():
            for f in files:
                img = cv2.imread(os.path.join(dp, f))
                if img is not None:
                    yield cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return gen(), tag, 30.0
    from nexus_bench.vision_bench import _moving_squares
    n = max(30, cfg.get("repeats", 20) * 2)
    frames = _moving_squares(n, seed=cfg.get("seed", 0))
    return iter(frames), "synthetic_moving_squares (labelled; use --video for real pixels)", 30.0

def run(cfg):
    import cv2
    dev = resolve_device(cfg.get("device", "auto"))
    req_fps = cfg.get("req_fps", 15) or 15
    deadline_ms = 1000.0 / req_fps
    out = {"module": "pipeline",
           "config": {**cfg, "resolved_device": dev, "deadline_ms": round(deadline_ms, 2)},
           "tests": {}, "errors": []}
    with Monitor() as mon:
        try:
            gen, tag, src_fps = _open_source(cfg)
            out["config"]["source"] = tag
            out["config"]["source_fps"] = round(src_fps, 1)
            yolo, dtag = load_weights((cfg.get("model") or "").strip(),
                                        want_fp16=(cfg.get("precision") == "fp16"))
            pk = precision_kwargs(yolo, cfg.get("precision") == "fp16") if yolo else {}
            # Fallback notice lives in config, not errors: everything else ran for real.
            out["config"]["detector"] = dtag if yolo is not None else \
                "motion_detector_fallback (pipeline cost only)"
            model, etag = load_embedder(dev, (cfg.get("reid_model") or "").strip())
            out["config"]["embedder"] = etag
            tr = Tracker()
            e2e, by_stage = [], {"read": [], "resize": [], "detect": [],
                                 "embed": [], "associate": [], "telemetry": []}
            n_frames, n_drops, n_miss = 0, 0, 0
            t_start = time.perf_counter()
            prev_gray = None
            for frame in gen:
                n_frames += 1
                pts = (n_frames - 1) * 1000.0 / src_fps
                elapsed_ms = (time.perf_counter() - t_start) * 1000
                if elapsed_ms - pts > deadline_ms:
                    n_drops += 1  # real-time consumer would have moved on
                    continue
                f0 = time.perf_counter()
                by_stage["read"].append((time.perf_counter() - f0) * 1000)  # already decoded above
                t0 = time.perf_counter()
                small = cv2.resize(frame, (cfg.get("imgsz", 640) // 2 * 2, 480))
                gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
                by_stage["resize"].append((time.perf_counter() - t0) * 1000)
                t0 = time.perf_counter()
                if yolo is not None:
                    r = yolo.predict(small, imgsz=cfg.get("imgsz", 640), device=dev,
                                     verbose=False, **pk)[0]
                    boxes = []
                    if r.boxes is not None:
                        for b in r.boxes.xywh.cpu().numpy()[:16]:
                            x, y, w, h = b
                            boxes.append([x - w / 2, y - h / 2, w, h])
                else:
                    boxes = _motion_detections(prev_gray, gray) if prev_gray is not None else []
                prev_gray = gray
                by_stage["detect"].append((time.perf_counter() - t0) * 1000)
                t0 = time.perf_counter()
                crops = []
                for b in boxes[:8]:
                    x, y, w, h = [max(0, int(v)) for v in b]
                    c = small[y:y + h, x:x + w]
                    crops.append(c if c.size else np.zeros((32, 32, 3), np.uint8))
                feats = list(embed_crops(model, crops, dev)[0]) if crops else []
                by_stage["embed"].append((time.perf_counter() - t0) * 1000)
                t0 = time.perf_counter()
                live = tr.step(boxes[:8], feats)
                by_stage["associate"].append((time.perf_counter() - t0) * 1000)
                t0 = time.perf_counter()
                pkt = {"n": n_frames, "tracks": [(i, [round(v, 1) for v in b]) for i, b in live]}
                _ = str(pkt).encode()
                by_stage["telemetry"].append((time.perf_counter() - t0) * 1000)
                lat = (time.perf_counter() - f0) * 1000
                e2e.append(lat)
                if lat > deadline_ms:
                    n_miss += 1
            wall_s = time.perf_counter() - t_start
            proc = len(e2e)
            d = summarize(e2e)
            d["frames_total"] = n_frames
            d["frames_processed"] = proc
            d["frames_dropped"] = n_drops
            d["drop_pct"] = round(100 * n_drops / n_frames, 2) if n_frames else 0
            d["deadline_miss_pct"] = round(100 * n_miss / proc, 2) if proc else 0
            d["throughput_fps"] = round(proc / wall_s, 2) if wall_s else 0
            d["stage_means_ms"] = {k: round(sum(v) / len(v), 3) for k, v in by_stage.items() if v}
            out["tests"]["end_to_end_ms"] = d
        except Exception as e:
            out["errors"].append(f"pipeline: {e}")
        out["telemetry"] = mon.summary()
    return out
