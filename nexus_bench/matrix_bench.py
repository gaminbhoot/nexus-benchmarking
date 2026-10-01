"""Test matrix: resolution x batch x target-FPS sweep with adaptive pruning.

Each cell runs the pipeline module in a FRESH worker process (no CUDA-state
contamination between cells). If a cell OOMs at batch b, larger batches at the
same imgsz are pruned (not run). Overall status FULL only if every executed
cell is FULL — one fallback poisons the matrix to PARTIAL.
"""
import copy

from nexus_bench import statuses as S
from nexus_bench.monitor import Monitor

def run(cfg):
    from nexus_bench.cli import _run_worker  # local import: cli never imports matrix
    out = {"module": "matrix",
           "config": {k: cfg.get(k) for k in
                      ("imgsz_list", "batch_list", "target_fps_list", "model",
                       "precision", "device", "req_fps")},
           "tests": {}, "errors": [], "status": S.FULL}
    imgszs = cfg.get("imgsz_list") or [cfg.get("imgsz", 640)]
    batches = sorted(cfg.get("batch_list") or [cfg.get("batch", 1)])
    fps_list = cfg.get("target_fps_list") or [cfg.get("target_fps") or 15.0]
    timeout = cfg.get("worker_timeout_s") or 300.0
    with Monitor() as mon:
        pruned, oom_at = 0, {}
        for iz in imgszs:
            for bs in batches:
                if oom_at.get(iz, 0) and bs >= oom_at[iz]:
                    pruned += 1
                    continue
                for fps in fps_list:
                    cell = copy.deepcopy(cfg)
                    cell.update({"imgsz": iz, "batch": bs, "target_fps": float(fps),
                                 "duration_s": min(cfg.get("duration_s", 30), 12)})
                    try:
                        res, _ = _run_worker("nexus_bench.pipeline_bench", cell,
                                             timeout, cfg.get("blas_threads"))
                    except Exception as e:
                        out["errors"].append(f"cell iz{iz} b{bs} f{fps}: supervisor: {e}")
                        continue
                    t = (res.get("tests") or {}).get("end_to_end_ms", {})
                    key = f"iz{iz}_b{bs}_f{fps}"
                    out["tests"][key] = {
                        "status": res.get("status"),
                        "fps": t.get("throughput_fps"), "p95_ms": t.get("p95_ms"),
                        "p99_ms": t.get("p99_ms"),
                        "miss_pct": t.get("deadline_miss_pct"),
                        "drop_pct": t.get("drop_pct")}
                    if any("out of memory" in str(e).lower() for e in res.get("errors", [])):
                        oom_at[iz] = bs if not oom_at.get(iz) else min(oom_at[iz], bs)
                        out["errors"].append(f"OOM boundary: iz{iz} b{bs} — larger batches pruned")
                    if res.get("status") != S.FULL:
                        out["status"] = S.PARTIAL
        out["tests"]["pruned_cells"] = pruned
        out["tests"]["oom_boundary"] = oom_at or "none found in sweep"
        if out["status"] == S.PARTIAL:
            out["errors"].append("PARTIAL: at least one cell was not FULL evidence")
        out["telemetry"] = mon.summary()
    return out
