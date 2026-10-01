"""Test matrix: resolution x batch x target-FPS, honestly.

Batch dimension runs through inference_bench, which places all N images in ONE
predict call — batch=2 is a real batch-2 inference (verified per_image scaling
in CI). Each target_fps maps to its own deadline (15->66.7ms, 30->33.3ms):
a cell passes a column only against that column's deadline, on per-image
latency for batches and batch throughput for images/sec.

Each cell runs in a FRESH worker process. OOM at batch b prunes larger batches
at the same imgsz. Overall FULL only if every executed cell is FULL.
"""
import copy

from nexus_bench import statuses as S
from nexus_bench.monitor import Monitor

def deadline_for_fps(fps):
    return round(1000.0 / fps, 2)

def run(cfg):
    from nexus_bench.cli import _run_worker  # local import: cli never imports matrix
    out = {"module": "matrix",
           "config": {k: cfg.get(k) for k in
                      ("imgsz_list", "batch_list", "target_fps_list", "model",
                       "precision", "device", "req_fps")},
           "tests": {}, "errors": [], "status": S.FULL}
    imgszs = cfg.get("imgsz_list") or [cfg.get("imgsz", 640)]
    batches = sorted(cfg.get("batch_list") or [cfg.get("batch", 1)])
    fps_list = [float(f) for f in (cfg.get("target_fps_list")
                                   or [cfg.get("target_fps") or 15.0])]
    timeout = cfg.get("worker_timeout_s") or 600.0
    with Monitor() as mon:
        pruned, oom_at = 0, {}
        for iz in imgszs:
            for bs in batches:
                if oom_at.get(iz, 0) and bs >= oom_at[iz]:
                    pruned += 1
                    continue
                cell = copy.deepcopy(cfg)
                cell.update({"imgsz": iz, "imgsz_list": [iz], "batch_list": [bs],
                             "duration_s": 5})
                try:
                    res, _ = _run_worker("nexus_bench.inference_bench", cell,
                                         timeout, cfg.get("blas_threads"))
                except Exception as e:
                    out["errors"].append(f"cell iz{iz} b{bs}: supervisor: {e}")
                    continue
                prec = cfg.get("precision", "fp32")
                key = f"yolo_imgsz{iz}_b{bs}_{prec}"
                m = (res.get("tests") or {}).get(key, {})
                oom = any("out of memory" in str(e).lower() for e in res.get("errors", []))
                if oom and not m:
                    oom_at[iz] = bs if not oom_at.get(iz) else min(oom_at[iz], bs)
                    out["errors"].append(f"OOM boundary: iz{iz} b{bs} — larger batches pruned")
                cell_res = {
                    "status": res.get("status"),
                    "actual_batch": m.get("batch_size", bs if m else None),
                    "resolution": iz, "precision": prec,
                    "backend": (res.get("config") or {}).get("resolved_device"),
                    "latency_per_batch_ms": m.get("mean_ms"),
                    "latency_per_image_ms": m.get("per_image_ms"),
                    "images_per_sec": m.get("batch_ips"),
                    "vram_note": "see memory/sustained peaks (per-cell VRAM isolated per worker)",
                    "oom": oom,
                    "deadlines": {str(f): {
                        "deadline_ms": deadline_for_fps(f),
                        "per_image_ok": (m.get("per_image_ms") or 1e9) <= deadline_for_fps(f),
                        "throughput_ok": (m.get("batch_ips") or 0) >= f * bs} if m else None
                        for f in fps_list}}
                out["tests"][f"iz{iz}_b{bs}"] = cell_res
                if res.get("status") != S.FULL:
                    out["status"] = S.PARTIAL
        out["tests"]["pruned_cells"] = pruned
        out["tests"]["oom_boundary"] = oom_at or "none found in sweep"
        if out["status"] == S.PARTIAL:
            out["errors"].append("PARTIAL: at least one cell was not FULL evidence")
        out["telemetry"] = mon.summary()
    return out
