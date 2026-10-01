"""RAM/VRAM: alloc throughput, pressure steps, peak tracking, OOM recovery."""
import time
import numpy as np

from nexus_bench.monitor import Monitor
from nexus_bench.profiles import resolve_device

def run(cfg):
    import psutil
    dev = resolve_device(cfg.get("device", "auto"))
    out = {"module": "memory", "config": {**cfg, "resolved_device": dev}, "tests": {}, "errors": []}
    with Monitor() as mon:
        try:
            sizes_mb = [16, 64, 256]
            for mb in sizes_mb:
                n = mb * 1024 * 1024 // 8
                t0 = time.perf_counter()
                a = np.ones(n)
                dt = time.perf_counter() - t0
                out["tests"][f"ram_alloc_{mb}mb_gbps"] = round((mb / 1e3) / dt, 2) if dt else None
                del a
            out["tests"]["ram_total_gb"] = round(psutil.virtual_memory().total / 1e9, 2)
        except Exception as e:
            out["errors"].append(f"ram: {e}")
        try:
            import torch
            if dev == "cuda" and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                held = []
                for mb in (256, 512, 1024):
                    try:
                        held.append(torch.empty(mb * 1024 * 1024 // 4, device="cuda"))
                    except RuntimeError as e:
                        out["errors"].append(f"vram pressure stopped at {mb}MB (recovered): {e}")
                        torch.cuda.empty_cache()
                        break
                out["tests"]["vram_peak_mb"] = round(torch.cuda.max_memory_allocated() / 1e6, 1)
                del held
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            else:
                out["tests"]["vram_note"] = "VRAM pressure probed on CUDA only"
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
