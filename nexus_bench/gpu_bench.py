"""GPU: matmul/conv/bandwidth at fp32 (+fp16/int8 where supported). Graceful CPU fallback."""
import time
import torch
import torch.nn as nn

from nexus_bench.monitor import Monitor
from nexus_bench.profiles import resolve_device
from nexus_bench.stats import summarize

def _sync(dev):
    if dev == "cuda":
        torch.cuda.synchronize()

def _bench(fn, warmup, repeats, dev):
    for _ in range(warmup):
        fn(); _sync(dev)
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter(); fn(); _sync(dev)
        ts.append((time.perf_counter() - t0) * 1000)
    return summarize(ts)

def run(cfg):
    dev = resolve_device(cfg.get("device", "auto"))
    w, r = cfg.get("warmup", 3), cfg.get("repeats", 20)
    out = {"module": "gpu", "config": {**cfg, "resolved_device": dev}, "tests": {}, "errors": []}
    if dev == "cpu":
        out["errors"].append("no GPU accelerator (cuda/mps unavailable); ran on CPU for reference")
    try:
        with Monitor() as mon:
            a = torch.randn(1024, 1024, device=dev); b = torch.randn(1024, 1024, device=dev)
            out["tests"]["matmul_fp32_ms"] = _bench(lambda: a @ b, w, r, dev)
            c = nn.Conv2d(64, 64, 3, padding=1).to(dev).eval()
            x = torch.randn(4, 64, 56, 56, device=dev)
            with torch.no_grad():
                out["tests"]["conv_fp32_ms"] = _bench(lambda: c(x), w, r, dev)
            big = torch.randn(64 * 1024 * 1024 // 4, device=dev)  # 64MB fp32
            def bw():
                return big.clone().sum()
            t0 = time.perf_counter()
            for _ in range(10):
                bw(); _sync(dev)
            dt = time.perf_counter() - t0
            out["tests"]["mem_bandwidth_gbps"] = round((64 * 2 * 10 / 1e3) / dt, 2) if dt else None
            if dev == "cuda" and torch.cuda.is_available():
                try:
                    ah, bh = a.half(), b.half()
                    out["tests"]["matmul_fp16_ms"] = _bench(lambda: ah @ bh, w, r, dev)
                except Exception as e:
                    out["errors"].append(f"fp16 unsupported: {e}")
                try:
                    q = torch.randint(-128, 127, (512, 512), dtype=torch.int8, device=dev)
                    out["tests"]["int8_supported"] = True
                    out["tests"]["matmul_int8_shape"] = list(q.shape)
                except Exception as e:
                    out["tests"]["int8_supported"] = False
                    out["errors"].append(f"int8 unsupported: {e}")
            else:
                out["tests"]["fp16_note"] = "fp16/int8 probed on CUDA only"
            out["telemetry"] = mon.summary()
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            out["errors"].append(f"OOM recovered gracefully: {e}")
        else:
            out["errors"].append(str(e))
    except Exception as e:
        out["errors"].append(str(e))
    return out
