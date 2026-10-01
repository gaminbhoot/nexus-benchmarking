"""Stats + feasibility correctness: the numbers must mean what they claim."""
from nexus_bench import statuses as S
from nexus_bench.stats import summarize
from nexus_bench.report import _assess

def test_percentiles_ordered():
    s = summarize([10.0, 20.0, 30.0, 40.0])
    assert s["n"] == 4 and s["median_ms"] == 25.0
    assert s["median_ms"] <= s["p95_ms"] <= s["p99_ms"]

def test_empty_and_single():
    assert summarize([])["n"] == 0
    s = summarize([5.0])
    assert s["mean_ms"] == 5.0 and s["stdev_ms"] == 0.0

def test_throughput_override():
    assert summarize([10.0], throughput=42.0)["throughput_ips"] == 42.0

def _res(tests, **kw):
    return {"module": "x", "config": {"deadline_ms": 66.7}, "tests": tests,
            "errors": kw.get("errors", []), **({"status": kw["status"]} if "status" in kw else {})}

def test_median_ok_p95_bad_is_limited():
    v, why = _assess("p", _res({"t": {"median_ms": 45.0, "p95_ms": 180.0, "p99_ms": 200.0}}), 15)
    assert v == "limited", why  # the exact robotics case: fine median, bad tail

def test_clean_run_is_ok():
    v, _ = _assess("p", _res({"t": {"median_ms": 30.0, "p95_ms": 50.0,
                                    "deadline_miss_pct": 1.0, "drop_pct": 0.0}}), 15)
    assert v == "ok"

def test_oom_exceeds():
    v, _ = _assess("p", _res({}, errors=["CUDA out of memory at batch 4"]), 15)
    assert v == "exceeds limits"

def test_no_workload_is_unsupported():
    v, _ = _assess("p", _res({}, status=S.UNSUPPORTED), 15)
    assert v == "unsupported"

def test_partial_is_limited_not_ok():
    v, why = _assess("p", _res({"t": {"median_ms": 30.0, "p95_ms": 50.0}}, status=S.PARTIAL), 15)
    assert v == "limited" and any("PARTIAL" in r for r in why)

def test_heavy_misses_exceed():
    v, _ = _assess("p", _res({"t": {"deadline_miss_pct": 40.0, "drop_pct": 5.0}}), 15)
    assert v == "exceeds limits"

def test_low_fps_limited():
    v, _ = _assess("p", _res({"t": {"throughput_fps": 10.0}}), 15)
    assert v == "limited"
