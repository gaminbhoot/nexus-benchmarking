"""Gate engine: workload-specific criteria, headroom, fail-closed semantics."""
from nexus_bench import gate as G
from nexus_bench import statuses as S

def _full_pipeline(fps=20.0, p95=50.0, p99=80.0, miss=1.0, drop=0.0):
    return {"module": "pipeline", "status": S.FULL, "tests": {
        "end_to_end_ms": {"throughput_fps": fps, "median_ms": 40.0, "p95_ms": p95,
                          "p99_ms": p99, "deadline_miss_pct": miss, "drop_pct": drop}},
        "errors": []}

def _full_integrated(fps=12.0):
    return {"module": "integrated", "status": S.FULL, "tests": {
        "per_agent": {"uav": {"fps": fps, "drops": 0}, "rover": {"fps": fps, "drops": 0}},
        "fusion": {"ticks": 50}}, "errors": []}

def _full_memory(occ_pct=70.0):
    return {"module": "memory", "status": S.FULL,
            "tests": {"vram_total_mb": 4000.0, "vram_peak_reserved_mb": 40.0 * occ_pct},
            "errors": []}

def _thermal_ok():
    return {"module": "thermal", "status": S.FULL, "tests": {"throttling": "none detected"},
            "errors": []}

def _acc_ok():
    return {"module": "accuracy", "status": S.FULL,
            "tests": {"agreement": {"map50_drop_vs_fp32": 0.01}}, "errors": []}

def _base():
    return {"pipeline": _full_pipeline(), "integrated": _full_integrated(),
            "memory": _full_memory(), "thermal": _thermal_ok(), "accuracy": _acc_ok()}

def test_gate_pass_with_headroom():
    g = G.evaluate(_base())
    assert g["verdict"] == S.PASS_WITH_HEADROOM, g["checks"]

def test_gate_partial_evidence_is_not_pass():
    r = _base()
    r["pipeline"] = dict(r["pipeline"], status=S.PARTIAL)
    g = G.evaluate(r)
    assert g["verdict"] in (S.FAIL, S.INCONCLUSIVE)

def test_gate_missing_module_is_not_pass():
    r = _base()
    del r["integrated"]
    assert G.evaluate(r)["verdict"] in (S.FAIL, S.INCONCLUSIVE)

def test_gate_tail_violation_fails():
    r = _base()
    r["pipeline"] = _full_pipeline(p95=200.0)
    g = G.evaluate(r)
    assert g["verdict"] in (S.FAIL, S.INCONCLUSIVE)
    assert any(not c["pass"] and "p95" in c["check"] for c in g["checks"])

def test_gate_oom_vetoes():
    r = _base()
    r["memory"] = dict(r["memory"], errors=["VRAM OOM at ladder step 3584MB (recovered)"])
    g = G.evaluate(r)
    assert g["verdict"] == S.FAIL
    assert any(c["check"] == "resources.no_oom" and not c["pass"] for c in g["checks"])

def test_gate_throttle_vetoes():
    r = _base()
    r["thermal"] = {"module": "thermal", "status": S.FULL,
                    "tests": {"throttling": "likely"}, "errors": []}
    g = G.evaluate(r)
    assert g["verdict"] == S.FAIL

def test_gate_no_accuracy_is_not_pass():
    r = _base()
    del r["accuracy"]
    assert G.evaluate(r)["verdict"] in (S.FAIL, S.INCONCLUSIVE)

def test_gate_aborted_module_fails():
    r = _base()
    r["thermal"] = {"module": "thermal", "status": S.ABORTED, "tests": {}, "errors": ["killed"]}
    assert G.evaluate(r)["verdict"] == S.FAIL
