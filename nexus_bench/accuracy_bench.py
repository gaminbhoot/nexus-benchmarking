"""Accuracy preservation: every deployment candidate vs the FP32 baseline.

For each candidate on the SAME validation data (--val-data):
  baseline mAP50 / mAP50-95 / precision / recall / per-class AP
  candidate mAP50 / mAP50-95 / precision / recall / per-class AP
  absolute + relative mAP50 and mAP50-95 drops, prediction agreement.
Candidates: fp16 (explicitly cast + dtype-verified, fail-closed) plus any
`candidate_engines: {name: path}` (TensorRT/ONNX artifacts, incl. INT8).
Agreement frames are real deployment pixels (--video > --image-dir >
synthetic content). Black frames are never used. Missing candidate mAP evidence
-> no map_comparison -> gate INCONCLUSIVE, never a pass.
"""
import os
import numpy as np

from nexus_bench import statuses as S
from nexus_bench.monitor import Monitor
from nexus_bench.profiles import resolve_device
from nexus_bench.yolo_util import load_weights

def _boxes_of(res):
    try:
        b = res[0].boxes
        if b is None:
            return np.zeros((0, 6))
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

def _agreement_frames(cfg, n=6):
    """Real pixels only. Chain documents itself; black frames never used."""
    import cv2
    vp = (cfg.get("video") or cfg.get("uav_video") or "").strip()
    if vp and os.path.exists(vp):
        cap = cv2.VideoCapture(vp)
        frames = []
        while len(frames) < n:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        cap.release()
        if frames:
            return frames, f"video:{vp}"
    dp = (cfg.get("image_dir") or "").strip()
    if dp and os.path.isdir(dp):
        files = sorted(p for p in os.listdir(dp)
                       if p.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")))[:n]
        frames = []
        for f in files:
            img = cv2.imread(os.path.join(dp, f))
            if img is not None:
                frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if frames:
            return frames, f"image_dir:{dp}"
    from nexus_bench.vision_bench import _moving_squares
    return _moving_squares(n, seed=cfg.get("seed", 0)), "synthetic content (labelled)"

def _val_map(model, val_data, imgsz, dev):
    m = model.val(data=val_data, imgsz=imgsz, device=dev, verbose=False)
    per_class = {}
    try:
        names = m.names if isinstance(m.names, dict) else {i: n for i, n in enumerate(m.names)}
        maps = list(m.box.maps)
        for i, ap in enumerate(maps):
            per_class[str(names.get(i, i))] = round(float(ap), 4)
    except Exception:
        pass
    return {"map50": round(float(m.box.map50), 4),
            "map50_95": round(float(m.box.map), 4),
            "precision": round(float(m.box.mp), 4),
            "recall": round(float(m.box.mr), 4),
            "per_class_ap50_95": per_class}

def _drop(base, cand):
    return {"abs": round(base - cand, 4),
            "rel": round((base - cand) / base, 4) if base else None}

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
            base = YOLO(mp)
            base_maps, cand_maps = None, {}
            if val_data and os.path.exists(val_data):
                try:
                    base_maps = _val_map(base, val_data, imgsz, dev)
                    out["tests"]["baseline_fp32"] = base_maps
                except Exception as e:
                    out["errors"].append(f"baseline mAP failed: {str(e)[:200]}")
            else:
                out["errors"].append("mAP skipped: pass --val-data <data.yaml> for labelled accuracy")
            # fp16 candidate: explicit cast + dtype verification (fail-closed).
            try:
                c16, rec = load_weights(mp, want_fp16=True)
                if c16 is None:
                    raise RuntimeError(rec.get("error", "load failed"))
                c16.model.float()
                c16.model.half()
                import torch
                dt = next(c16.model.parameters()).dtype
                if dt != torch.float16:
                    raise RuntimeError(f"cast unverified (dtype={dt})")
                out["config"]["fp16_dtype"] = "float16 (verified)"
                if base_maps:
                    try:
                        m16 = _val_map(c16, val_data, imgsz, dev)
                        d50, d5095 = _drop(base_maps["map50"], m16["map50"]), _drop(base_maps["map50_95"], m16["map50_95"])
                        cand_maps["fp16"] = {"map": m16, "map50_drop_abs": d50["abs"],
                                             "map50_drop_rel": d50["rel"],
                                             "map5095_drop_abs": d5095["abs"],
                                             "map5095_drop_rel": d5095["rel"]}
                    except Exception as e:
                        out["errors"].append(f"fp16 mAP failed: {str(e)[:200]}")
            except Exception as e:
                out["errors"].append(f"fp16 candidate withheld: {str(e)[:200]}")
            # engine candidates (TensorRT INT8/FP16, ONNX): same data, same metrics.
            for name, epath in (cfg.get("candidate_engines") or {}).items():
                try:
                    if not (epath and os.path.exists(epath)):
                        raise RuntimeError(f"engine file missing: {epath}")
                    me = YOLO(str(epath))
                    if base_maps:
                        mm = _val_map(me, val_data, imgsz, dev)
                        d50 = _drop(base_maps["map50"], mm["map50"])
                        d5095 = _drop(base_maps["map50_95"], mm["map50_95"])
                        cand_maps[name] = {"map": mm, "engine": str(epath),
                                           "map50_drop_abs": d50["abs"], "map50_drop_rel": d50["rel"],
                                           "map5095_drop_abs": d5095["abs"],
                                           "map5095_drop_rel": d5095["rel"]}
                    else:
                        out["errors"].append(f"{name}: no baseline mAP to compare against")
                except Exception as e:
                    out["errors"].append(f"{name} candidate failed: {str(e)[:200]}")
            out["tests"]["comparisons"] = cand_maps
            if cand_maps:
                worst = max(cand_maps.items(), key=lambda kv: kv[1].get("map50_drop_abs", -1))
                out["tests"]["map_comparison"] = {"candidate": worst[0], **worst[1]}
            else:
                out["errors"].append("no candidate-vs-baseline mAP comparison produced "
                                     "(gate INCONCLUSIVE until --val-data + a candidate exist)")
            # agreement on real frames (labels not needed).
            try:
                frames, ftag = _agreement_frames(cfg)
                out["config"]["agreement_source"] = ftag
                ref_b = [_boxes_of(base.predict([f], imgsz=imgsz, device=dev, verbose=False))
                         for f in frames]
                c16a, _ = load_weights(mp, want_fp16=True)
                if c16a is not None:
                    from nexus_bench.yolo_util import effective_precision, precision_kwargs
                    pk = precision_kwargs(True)
                    c16a.predict(frames[:1], imgsz=imgsz, device=dev, verbose=False, **pk)
                    if effective_precision(c16a) == "fp16":
                        agg = [_agreement(rb, _boxes_of(c16a.predict(
                            [f], imgsz=imgsz, device=dev, verbose=False, **pk)))
                            for rb, f in zip(ref_b, frames)]
                        out["tests"]["agreement"] = {
                            "frames": len(frames), "source": ftag,
                            "mean_matched_rate": round(float(np.mean([a["matched_rate"] for a in agg])), 3),
                            "mean_iou": round(float(np.mean([a["mean_iou"] for a in agg])), 3)}
            except Exception as e:
                out["errors"].append(f"agreement unavailable: {str(e)[:200]}")
            out["telemetry"] = mon.summary()
        except Exception as e:
            out["status"] = S.FAILED
            out["errors"].append(f"accuracy: {e}")
            out["telemetry"] = mon.summary()
    return out
