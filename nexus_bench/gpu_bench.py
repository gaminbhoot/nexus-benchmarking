"""GPU: measured GEMM throughput (explicit TFLOPS math), conv, directional bandwidth.

Results are labelled `measured_gemm_tflops` microbenchmarks — GEMM throughput,
not NEXUS performance. Graceful CPU reference fallback when no accelerator exists.
"""
import time
import torch
import torch.nn as nn

from nexus_bench.monitor import Monitor
from nexus_bench.profiles import resolve_device
from nexus_bench.stats import summarize

def _sync(dev):
    if dev == "cuda":
        torch.cuda.synchronize()

def _gemm_tflops(dev, n, dtype, warmup, repeats):
    """TFLOPS = 2*N^3 flops / elapsed. Synchronized, timed around the op only."""
    a = torch.randn(n, n, device=dev, dtype=dtype)
    b = torch.randn(n, n, device=dev, dtype=dtype)
    for _ in range(warmup):
        _ = a @ b
    _sync(dev)
    t0 = time.perf_counter()
    for _ in range(repeats):
        _ = a @ b
    _sync(dev)
    dt = (time.perf_counter() - t0) / repeats
    flops = 2 * n ** 3
    return {"n": n, "dtype": str(dtype).replace("torch.", ""),
            "mean_ms": round(dt * 1000, 3),
            "measured_gemm_tflops": round(flops / dt / 1e12, 3),
            "note": "synchronized GEMM microbenchmark; not a system score"}

def _bandwidth(dev, mb=256):
    """Directional bandwidth: H2D copy, D2D copy, D2H copy — timed separately."""
    out = {}
    n = mb * 1024 * 1024 // 4
    host = torch.randn(n)
    t0 = time.perf_counter()
    d = host.to(dev)
    _sync(dev)
    h2d = time.perf_counter() - t0
    out["h2d_gbps"] = round(mb / 1e3 / h2d, 2)
    t0 = time.perf_counter()
    e = d.clone()
    _sync(dev)
    d2d = time.perf_counter() - t0
    out["d2d_copy_gbps"] = round(mb / 1e3 / d2d, 2)
    t0 = time.perf_counter()
    _ = e.cpu()
    _sync(dev)
    out["d2h_gbps"] = round(mb / 1e3 / (time.perf_counter() - t0), 2)
    out["size_mb"] = mb
    del d, e
    return out

def run(cfg):
    dev = resolve_device(cfg.get("device", "auto"))
    w, r = cfg.get("warmup", 3), cfg.get("repeats", 20)
    out = {"module": "gpu", "config": {**cfg, "resolved_device": dev},
           "tests": {}, "errors": []}
    if dev == "cpu":
        out["errors"].append("no GPU accelerator (cuda/mps unavailable); CPU reference only")
    try:
        with Monitor() as mon:
            n = 2048 if dev == "cuda" else 1024
            out["tests"]["matmul_fp32"] = _gemm_tflops(dev, n, torch.float32, w, max(3, r // 4))
            c = nn.Conv2d(64, 64, 3, padding=1).to(dev).eval()
            x = torch.randn(4, 64, 56, 56, device=dev)
            with torch.no_grad():
                for _ in range(w):
                    c(x)
                _sync(dev)
                ts = []
                for _ in range(r):
                    t0 = time.perf_counter()
                    c(x)
                    _sync(dev)
                    ts.append((time.perf_counter() - t0) * 1000)
            out["tests"]["conv_fp32_ms"] = summarize(ts)
            out["tests"]["bandwidth"] = _bandwidth(dev)
            if dev == "cuda" and torch.cuda.is_available():
                try:
                    out["tests"]["matmul_fp16"] = _gemm_tflops(dev, n, torch.float16, w, max(3, r // 4))
                except Exception as e:
                    out["errors"].append(f"fp16 unsupported: {e}")
                try:
                    q = torch.randint(-128, 127, (512, 512), dtype=torch.int8, device=dev)
                    out["tests"]["int8_alloc_ok"] = True
                    out["tests"]["int8_shape"] = list(q.shape)
                    del q
                except Exception as e:
                    out["tests"]["int8_alloc_ok"] = False
                    out["errors"].append(f"int8 unsupported: {e}")
            else:
                out["tests"]["precision_note"] = "fp16/int8 throughput probed on CUDA only"
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
