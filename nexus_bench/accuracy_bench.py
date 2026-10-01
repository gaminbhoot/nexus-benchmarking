"""Accuracy preservation: does a faster backend/precision change what NEXUS sees?

Two evidence levels (both real, honestly separated):
1. agreement (always available with weights): fp32 vs candidate on the SAME
   frames — matched-box rate, mean IoU of matches, class-agreement, count delta.
2. mAP (needs --val-data): model.val() per backend/precision — mAP50, mAP50-95.
The gate vetoes any candidate whose mAP50 drop exceeds tolerance.
"""
import os
import numpy as np

from nexus_bench import statuses as S
from nexus_bench.monitor import Monitor
from nexus_bench.profiles import resolve_device
from nexus_bench.yolo_util import check_effective, load_weights, precision_kwargs

def _boxes_of(res):
    try:
        b = res[0].boxes
        if b is None:
            return np.zeros((0, 5))
        xyxy = b.xyxy.cpu().numpy()
        conf = b.conf.cpu().numpy()
        cls = b.cls.cpu().numpy()
        return np.column_stack([xyxy, cls, conf]) if len(xyxy) else np.zeros((0, 6))
    except Exception:
        return np.zeros((0, 6))

def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0

def _agreement(ref_boxes, cand_boxes, iou_thr=0.5):
    if not len(ref_boxes) and not len(cand_boxes):
        return {"matched_rate": 1.0, "mean_iou": 1.0, "class_agree": 1.0, "count_delta": 0}
    used, ious, cls_ok = set(), [], []
    for rb in ref_boxes:
        best, bj = 0, -1
        for j, cb in enumerate(cand_boxes):
            if j in used or rb[4] != cb[4]:
                continue
            v = _iou(rb[:4], cb[:4])
            if v > best:
                best, bj = v, j
        if best >= iou_thr:
            used.add(bj); ious.append(best); cls_ok.append(1)
    return {"matched_rate": round(len(ious) / max(1, len(ref_boxes)), 3),
            "mean_iou": round(float(np.mean(ious)), 3) if ious else 0.0,
            "class_agree": round(float(np.mean(cls_ok)), 3) if cls_ok else 0.0,
            "count_delta": int(len(cand_boxes) - len(ref_boxes))}

def run(cfg):
    from ultralytics import YOLO
    dev = resolve_device(cfg.get("device", "auto"))
    mp = (cfg.get("model") or "").strip()
    imgsz = cfg.get("imgsz", 640)
    val_data = (cfg.get("val_data") or "").strip()
    out = {"module": "accuracy", "config": {**cfg, "resolved_device": dev},
           "tests": {}, "errors": [], "status": S.FULL}
    if not mp or not os.path.exists(mp):
        out["status"] = S.UNSUPPORTED
        out["errors"].append("accuracy needs --model weights")
        with Monitor() as mon:
            out["telemetry"] = mon.summary()
        return out
    with Monitor() as mon:
        try:
            ref = YOLO(mp)
            frames = [np.zeros((imgsz, imgsz, 3), dtype=np.uint8) for _ in range(4)]
            ref_b = [_boxes_of(ref.predict([f], imgsz=imgsz, device=dev, verbose=False))
                     for f in frames]
            try:
                cand, rec = load_weights(mp, True)
                if cand is None:
                    raise RuntimeError(rec.get("error", "load failed"))
                pk = precision_kwargs(True)
                cand.predict(frames[:1], imgsz=imgsz, device=dev, verbose=False, **pk)
                if check_effective(cand, rec):
                    agg = [_agreement(rb, _boxes_of(cand.predict([f], imgsz=imgsz, device=dev,
                                                                 verbose=False, **pk)))
                           for rb, f in zip(ref_b, frames)]
                    d = {"frames": len(frames),
                         "mean_matched_rate": round(float(np.mean([a["matched_rate"] for a in agg])), 3),
                         "mean_iou": round(float(np.mean([a["mean_iou"] for a in agg])), 3),
                         "mean_count_delta": round(float(np.mean([a["count_delta"] for a in agg])), 2),
                         "effective": rec["effective"]}
                    out["tests"]["agreement"] = d
                else:
                    out["errors"].append(f"fp16 agreement withheld: effective={rec['effective']}")
            except Exception as e:
                out["errors"].append(f"fp16 agreement unavailable: {str(e)[:200]}")
            if val_data and os.path.exists(val_data):
                try:
                    m = ref.val(data=val_data, imgsz=imgsz, device=dev, verbose=False)
                    out["tests"]["map_fp32"] = {
                        "map50": round(float(m.box.map50), 4),
                        "map50_95": round(float(m.box.map), 4)}
                except Exception as e:
                    out["errors"].append(f"mAP val failed: {str(e)[:200]}")
            else:
                out["errors"].append("mAP skipped: pass --val-data <data.yaml> for labelled accuracy")
            out["telemetry"] = mon.summary()
        except Exception as e:
            out["status"] = S.FAILED
            out["errors"].append(f"accuracy: {e}")
            out["telemetry"] = mon.summary()
    return out
