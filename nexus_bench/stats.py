"""Latency/throughput statistics. Stdlib only."""
import statistics

def _pct(sorted_vals, q):
    if not sorted_vals:
        return None
    k = (len(sorted_vals) - 1) * q
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)

def summarize(samples_ms, throughput=None):
    """Return mean/median/p95/p99/stdev over latency samples (ms)."""
    vals = [float(v) for v in samples_ms if v is not None]
    if not vals:
        return {"n": 0, "mean_ms": None, "median_ms": None,
                "p95_ms": None, "p99_ms": None, "stdev_ms": None,
                "throughput_ips": throughput}
    s = sorted(vals)
    return {
        "n": len(vals),
        "mean_ms": statistics.fmean(vals),
        "median_ms": statistics.median(vals),
        "p95_ms": _pct(s, 0.95),
        "p99_ms": _pct(s, 0.99),
        "stdev_ms": statistics.stdev(vals) if len(vals) > 1 else 0.0,
        "throughput_ips": throughput if throughput is not None else 1000.0 / (statistics.fmean(vals) or 1e-9),
    }
