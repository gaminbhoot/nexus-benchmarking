"""End-to-end NEXUS perception pipeline on a paced frame stream.

Chain: paced acquisition (own thread, own latency) -> queue wait -> resize ->
detection -> embedding -> association -> telemetry encode -> output.
Reports input FPS vs processed FPS, e2e latency (capture->output, median/p95/p99),
deadline-miss %, drops, per-stage means.

Status: FULL only when detector is real YOLO at the verified requested precision
AND pixels are real (video/dir). Anything else is PARTIAL (never gate-eligible).
"""
import time
import numpy as np

from nexus_bench import statuses as S
from nexus_bench.monitor import Monitor
from nexus_bench.profiles import resolve_device
from nexus_bench.reid_embed import embed_crops, load_embedder
from nexus_bench.stats import summarize
from nexus_bench.stream import PacedSource
from nexus_bench.tracking_bench import Tracker
from nexus_bench.vision_bench import _motion_detections
from nexus_bench.yolo_util import check_effective, load_weights, precision_kwargs

def run(cfg):
    import cv2
    dev = resolve_device(cfg.get("device", "auto"))
    req_fps = cfg.get("req_fps", 15) or 15
    deadline_ms = 1000.0 / req_fps
    target_fps = cfg.get("target_fps") or min(30.0, req_fps * 2)
    out = {"module": "pipeline",
           "config": {**cfg, "resolved_device": dev, "deadline_ms": round(deadline_ms, 2),
                      "target_fps": target_fps},
           "tests": {}, "errors": [], "status": S.FULL}
    with Monitor() as mon:
        try:
            want_fp16 = cfg.get("precision") == "fp16"
            yolo, prec_rec = load_weights((cfg.get("model") or "").strip(), want_fp16=want_fp16)
            pk = precision_kwargs(want_fp16) if yolo else {}
            out["config"]["detector"] = prec_rec.get("error", f"yolo ({prec_rec['requested']})") \
                if yolo is None else f"yolo ({prec_rec['requested']})"
            model, etag = load_embedder(dev, (cfg.get("reid_model") or "").strip())
            out["config"]["embedder"] = etag
            src = PacedSource(cfg, target_fps=target_fps).start()
            out["config"]["source"] = src.tag
            tr = Tracker()
            e2e, by_stage, wait_ms = [], {"resize": [], "detect": [], "embed": [],
                                          "associate": [], "telemetry": []}, []
            n_out, n_miss, t_end = 0, 0, time.time() + cfg.get("duration_s", 30)
            prev_gray, verified, frames_seen = None, False, 0
            while time.time() < t_end:
                item = src.get(timeout=2.0)
                if item is None:
                    break
                if item == "retry":
                    continue
                frame, capture_t, seq = item
                frames_seen += 1
                wait_ms.append((time.time() - capture_t) * 1000)  # queue wait at pickup
                t0 = time.perf_counter()
                small = cv2.resize(frame, (cfg.get("imgsz", 640) // 2 * 2, 480))
                gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
                by_stage["resize"].append((time.perf_counter() - t0) * 1000)
                t0 = time.perf_counter()
                if yolo is not None:
                    r = yolo.predict(small, imgsz=cfg.get("imgsz", 640), device=dev,
                                     verbose=False, **pk)[0]
                    if not verified:
                        ok = check_effective(yolo, prec_rec)
                        out["config"]["effective_precision"] = prec_rec["effective"]
                        if not ok:
                            out["status"] = S.FAILED
                            out["errors"].append(
                                f"precision fail-closed: requested {prec_rec['requested']}, "
                                f"effective {prec_rec['effective']} — no fp16 numbers reported")
                            break
                        verified = True
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
                pkt = {"n": seq, "tracks": [(i, [round(v, 1) for v in b]) for i, b in live]}
                _ = str(pkt).encode()
                by_stage["telemetry"].append((time.perf_counter() - t0) * 1000)
                lat = (time.time() - capture_t) * 1000  # capture -> output
                e2e.append(lat)
                n_out += 1
                if lat > deadline_ms:
                    n_miss += 1
            wall_s = src.emitted / target_fps if src.emitted else 0
            src.stop()
            if out["status"] != S.FAILED:
                if yolo is None:
                    out["status"] = S.PARTIAL
                    out["errors"].append("PARTIAL: motion fallback, YOLO not benchmarked "
                                         "(informational only, not gate-eligible)")
                elif not src.real_pixels:
                    out["status"] = S.PARTIAL
                    out["errors"].append("PARTIAL: synthetic pixels (use --video/--image-dir "
                                         "for FULL)")
            d = summarize(e2e)
            d["input_fps"] = round(src.emitted / wall_s, 2) if wall_s else 0
            d["frames_emitted"] = src.emitted
            d["frames_processed"] = n_out
            d["frames_dropped"] = src.emitted - n_out + src.dropped_producer
            d["drop_pct"] = round(100 * d["frames_dropped"] / src.emitted, 2) if src.emitted else 0
            d["deadline_miss_pct"] = round(100 * n_miss / n_out, 2) if n_out else 0
            d["throughput_fps"] = round(n_out / wall_s, 2) if wall_s else 0
            d["stage_means_ms"] = {k: round(sum(v) / len(v), 3) for k, v in by_stage.items() if v}
            import statistics as _st
            d["acquire_mean_ms"] = round(_st.fmean(src.acquire_ms), 3) if src.acquire_ms else None
            d["queue_wait_mean_ms"] = round(_st.fmean(wait_ms), 3) if wait_ms else None
            out["tests"]["end_to_end_ms"] = d
        except Exception as e:
            out["status"] = S.FAILED
            out["errors"].append(f"pipeline: {e}")
        out["telemetry"] = mon.summary()
    return out
