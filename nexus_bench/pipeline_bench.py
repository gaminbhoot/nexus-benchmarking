"""End-to-end NEXUS perception pipeline on a paced frame stream.

Chain with timestamps: t_acquire_start -> t_decode_done (=t_capture) ->
t_enqueue -> t_dequeue -> resize/detect/embed/associate/telemetry -> t_output.
Reports acquisition latency, queue wait, processing latency, total
input->output latency, input vs processed FPS, deadline-miss %, and drops under
the single conservation-accounted definition (see stream.drop_pct).

Status: FULL only when detector is real YOLO at the verified requested precision
AND pixels are real (video/dir). Anything else is PARTIAL (never gate-eligible).
"""
import statistics as _st
import time
import numpy as np

from nexus_bench import statuses as S
from nexus_bench.monitor import Monitor
from nexus_bench.profiles import resolve_device
from nexus_bench.reid_embed import embed_crops, load_embedder
from nexus_bench.stats import summarize
from nexus_bench.stream import PacedSource, drop_pct
from nexus_bench.tracking_bench import Tracker
from nexus_bench.vision_bench import _motion_detections
from nexus_bench.yolo_util import check_effective, load_weights, precision_kwargs

def _detect(yolo, small, imgsz, dev, pk):
    r = yolo.predict(small, imgsz=imgsz, device=dev, verbose=False, **pk)[0]
    boxes = []
    if r.boxes is not None:
        for b in r.boxes.xywh.cpu().numpy()[:16]:
            x, y, w, h = b
            boxes.append([x - w / 2, y - h / 2, w, h])
    return boxes

def run(cfg):
    import cv2
    dev = resolve_device(cfg.get("device", "auto"))
    req_fps = cfg.get("req_fps", 15) or 15
    deadline_ms = 1000.0 / req_fps
    target_fps = cfg.get("target_fps") or min(30.0, req_fps * 2)
    out = {"module": "pipeline",
           "config": {**cfg, "resolved_device": dev, "deadline_ms": round(deadline_ms, 2),
                      "target_fps": target_fps,
                      "drop_definition": "drop_pct = 100*(generated-delivered)/generated"},
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
            src = PacedSource(cfg, target_fps=target_fps,
                              source_keys=tuple(cfg.get("source_keys") or ("video",)),
                              loop=True).start()
            out["config"]["source"] = src.tag
            out["config"]["source_kind"] = src.source_kind
            tr = Tracker()
            e2e, wait_ms, proc_ms = [], [], []
            by_stage = {"resize": [], "detect": [], "embed": [], "associate": [], "telemetry": []}
            n_miss, t_end = 0, time.time() + cfg.get("duration_s", 30)
            prev_gray, verified = None, False
            while time.time() < t_end:
                item = src.get(timeout=2.0)
                if item is None:
                    break
                if item == "retry":
                    continue
                frame, meta = item
                t_proc0 = time.perf_counter()
                wait_ms.append((meta["t_dequeue"] - meta["t_enqueue"]) * 1000)
                try:
                    t0 = time.perf_counter()
                    small = cv2.resize(frame, (cfg.get("imgsz", 640) // 2 * 2, 480))
                    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
                    by_stage["resize"].append((time.perf_counter() - t0) * 1000)
                    t0 = time.perf_counter()
                    if yolo is not None:
                        boxes = _detect(yolo, small, cfg.get("imgsz", 640), dev, pk)
                        if not verified:
                            verified = check_effective(yolo, prec_rec)
                            out["config"]["effective_precision"] = prec_rec["effective"]
                            if not verified:
                                out["status"] = S.FAILED
                                out["errors"].append(
                                    f"precision fail-closed: requested {prec_rec['requested']}, "
                                    f"effective {prec_rec['effective']} — numbers withheld")
                                break
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
                    pkt = {"n": meta["seq"],
                           "tracks": [(i, [round(v, 1) for v in b]) for i, b in live]}
                    _ = str(pkt).encode()
                    by_stage["telemetry"].append((time.perf_counter() - t0) * 1000)
                    lat = (time.time() - meta["t_capture"]) * 1000
                    e2e.append(lat)
                    proc_ms.append((time.perf_counter() - t_proc0) * 1000)
                    src.acct.processed += 1
                    src.acct.delivered += 1
                    if lat > deadline_ms:
                        n_miss += 1
                except Exception:
                    src.acct.processing_failures += 1
            src.stop()
            # Measured wall time: producer start to consumer end — never inferred
            # from configured FPS (that would hide overload by construction).
            wall_measured = time.time() - (src.t_start_wall or time.time())
            conserved, books = src.acct.check_conservation()
            if not conserved:
                out["errors"].append(f"accounting violation (investigate): {books}")
            if out["status"] != S.FAILED:
                if yolo is None:
                    out["status"] = S.PARTIAL
                    out["errors"].append("PARTIAL: motion fallback, YOLO not benchmarked "
                                         "(informational only, not gate-eligible)")
                elif not src.real_pixels:
                    out["status"] = S.PARTIAL
                    out["errors"].append("PARTIAL: synthetic pixels (use --video/--image-dir "
                                         "for FULL)")
            a = src.acct.as_dict()
            d = summarize(e2e)
            d["accounting"] = a
            d["conservation_ok"] = conserved
            wall_paced = a["generated"] / target_fps if a["generated"] else 0
            d["input_fps"] = round(a["generated"] / wall_measured, 2) if wall_measured else 0
            d["throughput_fps"] = round(a["delivered"] / wall_measured, 2) if wall_measured else 0
            d["wall_measured_s"] = round(wall_measured, 2)
            d["wall_paced_s"] = round(wall_paced, 2)
            d["drop_pct"] = drop_pct(a["generated"], a["delivered"])
            d["deadline_miss_pct"] = round(100 * n_miss / a["delivered"], 2) if a["delivered"] else 0
            d["stage_means_ms"] = {k: round(sum(v) / len(v), 3) for k, v in by_stage.items() if v}
            d["acquire_mean_ms"] = round(_st.fmean(src.acquire_ms), 3) if src.acquire_ms else None
            d["queue_wait_mean_ms"] = round(_st.fmean(wait_ms), 3) if wait_ms else None
            d["processing_mean_ms"] = round(_st.fmean(proc_ms), 3) if proc_ms else None
            d["replay_count"] = src.replay_count
            d["source_duration_s"] = src.source_duration_s
            out["tests"]["end_to_end_ms"] = d
        except Exception as e:
            out["status"] = S.FAILED
            out["errors"].append(f"pipeline: {e}")
        out["telemetry"] = mon.summary()
    return out
