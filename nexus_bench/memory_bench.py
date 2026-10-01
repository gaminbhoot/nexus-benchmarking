"""RAM/VRAM: stepped ladder with real ops, model footprint, OOM threshold.

On CUDA the ladder climbs toward 3.5 GB with allocate+compute+copy at each step,
recording per-step latency so the knee (slowdown before OOM) is visible. With
--model, the actual YOLO footprint (before/after load + one inference) is the
headline number — that remainder is what NEXUS really has to work with.
"""
import os
import time
import numpy as np

from nexus_bench.monitor import Monitor
from nexus_bench.profiles import resolve_device

LADDER_MB = [256, 512, 1024, 1536, 2048, 2560, 3072, 3328, 3584]

def run(cfg):
    import psutil
    dev = resolve_device(cfg.get("device", "auto"))
    out = {"module": "memory", "config": {**cfg, "resolved_device": dev},
           "tests": {}, "errors": []}
    with Monitor() as mon:
        try:
            for mb in (16, 64, 256):
                n = mb * 1024 * 1024 // 8
                t0 = time.perf_counter()
                a = np.ones(n)
                a += 1  # touch the pages; allocation alone proves nothing
                dt = time.perf_counter() - t0
                out["tests"][f"ram_alloc_touch_{mb}mb_gbps"] = round((mb * 2 / 1e3) / dt, 2) if dt else None
                del a
            out["tests"]["ram_total_gb"] = round(psutil.virtual_memory().total / 1e9, 2)
        except Exception as e:
            out["errors"].append(f"ram: {e}")
        try:
            import torch
            if dev == "cuda" and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                total_mb = torch.cuda.get_device_properties(0).total_memory / 1e6
                out["tests"]["vram_total_mb"] = round(total_mb, 1)
                try:  # driver truth, independent of the PyTorch allocator
                    free_b, total_b = torch.cuda.mem_get_info()
                    out["tests"]["driver_free_mb"] = round(free_b / 1e6, 1)
                    out["tests"]["driver_used_mb"] = round((total_b - free_b) / 1e6, 1)
                except Exception as e:
                    out["errors"].append(f"mem_get_info: {e}")
                steps, held, oom_at = [], [], None
                for mb in LADDER_MB:
                    if mb > total_mb * 0.95:
                        break
                    try:
                        t0 = time.perf_counter()
                        t = torch.empty(mb * 1024 * 1024 // 4, device="cuda")
                        t.uniform_()          # touch
                        u = t.clone().sum()   # copy + reduce (real traffic)
                        torch.cuda.synchronize()
                        steps.append({"mb": mb, "alloc_touch_ms": round((time.perf_counter() - t0) * 1000, 1)})
                        held.append(t)
                        del u
                    except RuntimeError:
                        oom_at = mb
                        out["errors"].append(
                            f"VRAM OOM at ladder step {mb}MB (recovered) — usable ceiling found")
                        torch.cuda.empty_cache()
                        break
                out["tests"]["vram_ladder"] = steps
                out["tests"]["vram_oom_threshold_mb"] = oom_at or f">{steps[-1]['mb'] if steps else '?'} (no OOM in ladder)"
                for t in held:
                    del t
                torch.cuda.empty_cache()
                peak_alloc = torch.cuda.max_memory_allocated() / 1e6
                peak_res = torch.cuda.max_memory_reserved() / 1e6
                out["tests"]["vram_peak_alloc_mb"] = round(peak_alloc, 1)
                out["tests"]["vram_peak_reserved_mb"] = round(peak_res, 1)
                out["tests"]["vram_fragmentation_gap_mb"] = round(peak_res - peak_alloc, 1)
                try:
                    st = torch.cuda.memory_stats()
                    out["tests"]["allocator"] = {
                        "allocated_mb": round(st.get("allocated_bytes.all.current", 0) / 1e6, 1),
                        "reserved_mb": round(st.get("reserved_bytes.all.current", 0) / 1e6, 1),
                        "active_blocks": st.get("active_bytes.all.current", "n/a"),
                        "note": "allocated=tensor memory, reserved=allocator cache (incl. fragmentation)"}
                except Exception as e:
                    out["errors"].append(f"memory_stats: {e}")
                mp = (cfg.get("model") or "").strip()
                if mp and os.path.exists(mp):
                    try:
                        torch.cuda.reset_peak_memory_stats()
                        torch.cuda.empty_cache()
                        free_before = torch.cuda.mem_get_info()[0] / 1e6
                        base_alloc = torch.cuda.memory_allocated() / 1e6
                        from ultralytics import YOLO
                        y = YOLO(mp)
                        _ = y.predict(np.zeros((640, 640, 3), np.uint8),
                                      imgsz=cfg.get("imgsz", 640), device="cuda", verbose=False)
                        peak = torch.cuda.max_memory_allocated() / 1e6
                        del y
                        torch.cuda.empty_cache()
                        free_after = torch.cuda.mem_get_info()[0] / 1e6
                        out["tests"]["model_memory_protocol_mb"] = {
                            "free_before": round(free_before, 1),
                            "peak_during_inference": round(peak, 1),
                            # working set above the pre-existing baseline, not total-minus-current
                            "model_working_set": round(peak - base_alloc, 1),
                            "free_after": round(free_after, 1),
                            "safe_headroom": round(free_after, 1)}
                    except RuntimeError as e:
                        out["errors"].append(f"model footprint OOM — model does not fit: {str(e)[:200]}")
                        torch.cuda.empty_cache()
                    except Exception as e:
                        out["errors"].append(f"model footprint: {e}")
                else:
                    out["tests"]["footprint_note"] = "pass --model to measure the actual YOLO VRAM footprint"
            else:
                out["tests"]["vram_note"] = "VRAM ladder probed on CUDA only"
        except Exception as e:
            out["errors"].append(f"vram: {e}")
        out["telemetry"] = mon.summary()
        try:
            rss = [s.get("proc_rss_mb", 0) for s in mon.samples]
            if rss:
                out["tests"]["proc_peak_rss_mb"] = round(max(rss), 1)
        except Exception:
            pass
    return out
