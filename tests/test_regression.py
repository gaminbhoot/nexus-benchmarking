"""Regression tests for every previously identified gate/fail-open issue."""
import json
import sys

import numpy as np

from nexus_bench import gate as G
from nexus_bench import provenance, statuses as S
from nexus_bench.cli import _load_profile, MODULES


def _passing_base():
    def full(mod, tests):
        return {"module": mod, "status": S.FULL, "tests": tests, "errors": []}
    return {
        "pipeline": full("pipeline", {"end_to_end_ms": {
            "throughput_fps": 20.0, "median_ms": 40.0, "p95_ms": 50.0, "p99_ms": 80.0,
            "deadline_miss_pct": 1.0, "drop_pct": 0.0}}),
        "integrated": full("integrated", {
            "per_agent": {"uav": {"fps": 12.0, "drops": 0}, "rover": {"fps": 12.0, "drops": 0}},
            "fusion": {"ticks": 50}}),
        "memory": full("memory", {"vram_total_mb": 4000.0, "vram_peak_reserved_mb": 2800.0}),
        "thermal": full("thermal", {"throttling": "none_detected"}),
        "accuracy": dict(full("accuracy", {"map_comparison": {
            "candidate": "fp16", "map50_drop_abs": 0.01, "map50_drop_rel": 0.02}}),
            config={"model": "m.pt", "precision": "fp32", "resolved_device": "cuda",
                    "imgsz": 640, "batch": 1}),
        "sustained": dict(full("sustained", {
            "per_agent": {
                "uav": {"steady_fps": 16.0, "final_fps": 16.0, "drop_pct": 1.0,
                        "degradation_pct": 5.0, "oom_events": 0, "min_window_fps": 12.0,
                        "final_window_fps": 12.0, "max_p95_ms": 55.0,
                        "max_window_p99_ms": 90.0, "max_window_miss_pct": 2.0,
                        "max_window_drop_pct": 1.0, "conservation_ok": True,
                        "actual_duration_s": 600, "requested_duration_s": 600,
                        "duration_complete": True},
                "rover": {"steady_fps": 16.0, "final_fps": 16.0, "drop_pct": 1.0,
                          "degradation_pct": 5.0, "oom_events": 0, "min_window_fps": 12.0,
                          "final_window_fps": 12.0, "max_p95_ms": 55.0,
                          "max_window_p99_ms": 90.0, "max_window_miss_pct": 2.0,
                          "max_window_drop_pct": 1.0, "conservation_ok": True,
                          "actual_duration_s": 600, "requested_duration_s": 600,
                          "duration_complete": True}},
            "memory_growth": {"growth": 10.0, "leak_suspected": False},
            "fusion": {"ticks": 1200, "tick_ms": {"p95_ms": 5.0}},
            "resource_evolution": {
                "vram_pct": {"initial": 60.0, "max": 70.0, "final": 65.0},
                "gpu_temp_c": {"initial": 60.0, "max": 75.0, "final": 70.0,
                               "per_window": [60.0, 70.0, 75.0]},
                "gpu_clock_mhz": {"per_window": [1500.0, 1500.0, 1495.0, 1495.0]}}}),
            config={"duration_s": 600, "model": "m.pt", "precision": "fp32",
                    "resolved_device": "cuda", "imgsz": 640, "batch": 1}),
        "_power": {"signals": {"ac_connected": {"value": True, "how": "measured"}}},
        "_provenance": {"git": {"sha": "abc", "dirty": False},
                        "fingerprints": {"model": "abc123"}},
    }


# --- invalid/incomplete result can never PASS ---
def test_garbage_result_never_passes():
    bad = {"pipeline": {"module": "pipeline", "status": S.FAILED, "tests": {}, "errors": ["boom"]},
           "integrated": {"module": "integrated", "status": S.PARTIAL, "tests": {}, "errors": []}}
    assert G.evaluate(bad)["verdict"] != S.PASS
    assert G.evaluate(bad)["verdict"] != S.PASS_WITH_HEADROOM
    assert G.evaluate({})["verdict"] in (S.FAIL, S.INCONCLUSIVE)


# --- thermal fail-closed ---
def test_thermal_unknown_is_inconclusive_not_pass():
    import copy
    r = _passing_base()
    r["thermal"] = {"module": "thermal", "status": S.FULL,
                    "tests": {"throttling": "unknown"}, "errors": []}
    # remove sustained thermal telemetry too: nothing certifiable anywhere
    r2 = copy.deepcopy(r)
    del r2["sustained"]["tests"]["resource_evolution"]["gpu_temp_c"]
    del r2["sustained"]["tests"]["resource_evolution"]["gpu_clock_mhz"]
    g = G.evaluate(r2)
    assert g["verdict"] == S.INCONCLUSIVE


def test_thermal_slowdown_unknown_is_inconclusive():
    import copy
    r = _passing_base()
    r["thermal"]["tests"] = {"throttling": "slowdown_cause_unknown"}
    r2 = copy.deepcopy(r)
    del r2["sustained"]["tests"]["resource_evolution"]["gpu_temp_c"]
    del r2["sustained"]["tests"]["resource_evolution"]["gpu_clock_mhz"]
    assert G.evaluate(r2)["verdict"] == S.INCONCLUSIVE


def test_sustained_thermal_overrides_unknown_short():
    # Sustained 10-minute telemetry with no throttle signs authoritatively
    # clears an unknown short probe (fixture evo: cool + flat clocks).
    r = _passing_base()
    r["thermal"] = {"module": "thermal", "status": S.FULL,
                    "tests": {"throttling": "unknown"}, "errors": []}
    g = G.evaluate(r)
    assert g["verdict"] in (S.PASS, S.PASS_WITH_HEADROOM)


# --- OOM kinds ---
def test_capacity_probe_oom_does_not_veto():
    r = _passing_base()
    r["memory"] = dict(r["memory"], errors=["CAPACITY_PROBE_OOM at ladder step 3584MB (recovered, exploratory only)"])
    g = G.evaluate(r)
    assert all(c["pass"] for c in g["checks"] if c["check"] == "resources.no_oom")


def test_workload_oom_vetoes():
    r = _passing_base()
    r["sustained"]["tests"]["per_agent"]["uav"]["oom_events"] = 2
    r["sustained"]["errors"] = ["WORKLOAD_OOM observed during sustained run (gate veto)"]
    g = G.evaluate(r)
    assert g["verdict"] == S.FAIL
    assert any(c["check"] == "resources.no_oom" and not c["pass"] for c in g["checks"])


# --- source separation ---
def test_uav_rover_sources_resolve_independently(tmp_path):
    from nexus_bench.stream import PacedSource
    u = tmp_path / "uav.mp4"
    v = tmp_path / "rover.mp4"
    u.write_bytes(b"fake-uav")
    v.write_bytes(b"fake-rover")
    cfg = {"uav_video": str(u), "rover_video": str(v), "video": "", "seed": 0, "agent_frames": 2}
    su = PacedSource(cfg, source_keys=("uav_video", "video"))
    sr = PacedSource(cfg, source_keys=("rover_video", "video"))
    assert su.source_used == "uav_video" and su.source_detail == str(u)
    assert sr.source_used == "rover_video" and sr.source_detail == str(v)
    assert su.source_detail != sr.source_detail  # never silently the same feed


def test_single_source_fallback_is_explicit(tmp_path):
    from nexus_bench.stream import PacedSource
    only = tmp_path / "only.mp4"
    only.write_bytes(b"x")
    cfg = {"uav_video": "", "rover_video": "", "video": str(only), "seed": 0, "agent_frames": 2}
    sr = PacedSource(cfg, source_keys=("rover_video", "video"))
    assert sr.source_used == "video"  # documented fallback chain, recorded per agent


# --- synthetic integrated stays non-gate-eligible ---
def test_synthetic_integrated_is_partial():
    import nexus_bench.integrated_bench as ib
    cfg = {"seed": 0, "warmup": 1, "repeats": 2, "duration_s": 2, "device": "cpu",
           "imgsz": 320, "batch": 1, "precision": "fp32", "model": "",
           "integrated_mode": "throughput", "agent_frames": 4, "req_fps": 15}
    r = ib.run(cfg)
    assert r["status"] == S.PARTIAL
    g = G.evaluate({"pipeline": r, "integrated": r})
    assert g["verdict"] in (S.FAIL, S.INCONCLUSIVE)


# --- real YOLO vision path with mocked model ---
class _FakeBoxes:
    @property
    def xywh(self):
        import torch
        return torch.tensor([[100.0, 100.0, 40.0, 40.0]])


class _FakeRes:
    boxes = _FakeBoxes()


class _FakeYOLO:
    instances = []

    def __init__(self, *a, **k):
        self.calls = []
        _FakeYOLO.instances.append(self)
        import types
        self.predictor = types.SimpleNamespace(args={"quantize": "fp16"})
        self.model = types.SimpleNamespace(
            parameters=lambda: iter([__import__("torch").zeros(1).half()]),
            float=lambda: None, half=lambda: None)

    def predict(self, *a, **k):
        self.calls.append(k)
        return [_FakeRes()]


def test_vision_real_yolo_path_uses_adapter(monkeypatch):
    import nexus_bench.vision_bench as vb
    import torch
    monkeypatch.setattr(vb, "load_weights", lambda p, want_fp16=False: (_FakeYOLO(), {
        "requested": "fp16", "effective": "unverified", "backend": "ultralytics",
        "ultralytics_version": "8.4.171", "flag": {"quantize": "fp16"}}))
    monkeypatch.setattr(vb, "precision_kwargs", lambda want: {"quantize": "fp16"} if want else {})
    monkeypatch.setattr(vb, "check_effective", lambda m, rec: True)
    cfg = {"seed": 0, "warmup": 1, "repeats": 3, "duration_s": 2, "device": "cpu",
           "imgsz": 320, "batch": 1, "precision": "fp16", "model": "x.pt"}
    r = vb.run(cfg)
    assert r["status"] == S.FULL
    assert _FakeYOLO.instances and _FakeYOLO.instances[-1].calls
    assert _FakeYOLO.instances[-1].calls[0].get("quantize") == "fp16"
    assert "detection_per_frame_ms" in r["tests"]
    _FakeYOLO.instances.clear()


# --- ONNX CPU fallback rejection ---
def test_onnx_cpu_fallback_rejected(monkeypatch):
    import types
    from nexus_bench import backends_bench as bb

    class FakeSess:
        def get_providers(self):
            return ["CPUExecutionProvider"]

    fake_ort = types.SimpleNamespace(
        InferenceSession=lambda *a, **k: FakeSess(), get_device=lambda: "CPU")
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    sess, why = bb.onnx_cuda_session("/tmp/never.onnx")
    assert sess is None and "CPU fallback rejected" in why


# --- matrix batch honesty + deadline mapping ---
def test_matrix_batches_are_real_and_deadlines_move(monkeypatch):
    import nexus_bench.cli as cli
    import nexus_bench.matrix_bench as mb
    seen = []

    def fake_worker(submod, cell, timeout, cblas):
        seen.append((cell["imgsz"], cell["batch_list"]))
        bs = cell["batch_list"][0]
        prec = cell.get("precision", "fp32")
        return ({"module": "inference", "status": S.FULL,
                 "config": {"resolved_device": "cuda"},
                 "tests": {f"yolo_imgsz{cell['imgsz']}_b{bs}_{prec}": {
                     "batch_size": bs, "mean_ms": 10.0 * bs,
                     "per_image_ms": 10.0, "batch_ips": 100.0 * bs}},
                 "errors": []}, {})

    monkeypatch.setattr(cli, "_run_worker", fake_worker)
    cfg = {"seed": 0, "imgsz_list": [320], "batch_list": [1, 2],
           "target_fps_list": [15.0, 30.0], "model": "m.pt", "precision": "fp32",
           "device": "cuda", "req_fps": 15, "duration_s": 5, "worker_timeout_s": 60,
           "blas_threads": None}
    r = mb.run(cfg)
    assert seen == [(320, [1]), (320, [2])]  # batch dimension really changes the workload
    c1, c2 = r["tests"]["iz320_b1"], r["tests"]["iz320_b2"]
    assert c1["actual_batch"] == 1 and c2["actual_batch"] == 2
    assert c1["deadlines"]["15.0"]["deadline_ms"] == 66.67
    assert c1["deadlines"]["30.0"]["deadline_ms"] == 33.33  # deadline follows FPS


# --- --runs worst-case aggregation ---
def test_runs_aggregate_worst_case():
    res = {
        "pipeline__run1": {"module": "pipeline", "status": S.FULL,
                           "tests": {"end_to_end_ms": {"throughput_fps": 20.0, "p95_ms": 50.0,
                                                        "deadline_miss_pct": 1.0, "drop_pct": 0.0}},
                           "errors": []},
        "pipeline__run2": {"module": "pipeline", "status": S.FULL,
                           "tests": {"end_to_end_ms": {"throughput_fps": 12.0, "p95_ms": 90.0,
                                                        "deadline_miss_pct": 4.0, "drop_pct": 2.0}},
                           "errors": []},
    }
    out = G.aggregate_runs(res)
    agg = out["pipeline"]
    assert agg["tests"]["end_to_end_ms"]["throughput_fps"] == 12.0  # worst fps
    assert agg["tests"]["end_to_end_ms"]["p95_ms"] == 90.0  # worst tail
    assert "pipeline__run1" in out and "pipeline__run2" in out  # individuals kept


# --- full_stack [all] really means all ---
def test_full_stack_all_expands():
    import os
    p = "profiles/full_stack.yaml"
    assert os.path.exists(p)
    from nexus_bench.profiles import get
    cfg, mods, _ = _load_profile(p, {})
    assert mods, "modules=[all] resolved to zero modules"
    assert set(mods) == {m for m in MODULES if m != "system"}


# --- hardware mismatch ---
def test_hardware_mismatch_fails():
    r = _passing_base()
    results = dict(r, _profile={"gpu": {"name": "Apple M4", "vram_total": "24 GB"}})
    gate = dict(G.DEFAULT_GATE, expected_hardware={"gpu_contains": "RTX 3050"})
    g = G.evaluate(results, gate)
    assert g["verdict"] == S.FAIL
    assert any("DOES NOT MATCH" in c["detail"] for c in g["checks"] if not c["pass"])


# --- dirty provenance ---
def test_dirty_tree_blocks_strict_gate():
    import copy
    r = _passing_base()
    results = dict(r, _provenance={"git": {"sha": "abc", "dirty": True}})
    gate = copy.deepcopy(G.DEFAULT_GATE)
    gate.setdefault("provenance", {})["forbid_dirty"] = True
    g = G.evaluate(results, gate)
    assert g["verdict"] == S.INCONCLUSIVE
    assert any(c["check"] == "provenance.clean_tree" and not c["pass"] for c in g["checks"])


def test_provenance_full_sha_and_inputs(tmp_path):
    f = tmp_path / "m.pt"
    f.write_bytes(b"model-bytes-123")
    m = provenance.manifest({"model": str(f), "video": "", "uav_video": "",
                             "rover_video": "", "image_dir": "", "val_data": "",
                             "reid_model": "", "seed": 0})
    assert len(m["fingerprints"]["model"]) == 64  # full SHA-256, never truncated
    assert m["fingerprints"]["uav_video"] == "none"
    assert "source_state" in m


# --- accuracy agreement + missing evidence ---
def test_accuracy_map_comparison_shape():
    from nexus_bench.accuracy_bench import _drop
    assert _drop(0.5, 0.48) == {"abs": 0.02, "rel": 0.04}


# --- deployment identity mismatch ---
def test_deployment_identity_mismatch_fails():
    import copy
    r = _passing_base()
    r2 = copy.deepcopy(r)
    r2["accuracy"]["config"]["precision"] = "int8"
    g = G.evaluate(r2)
    assert g["verdict"] in (S.FAIL, S.INCONCLUSIVE)
    assert any(c["check"] == "deployment.identity_match" and not c["pass"]
               for c in g["checks"])


# --- fusion contract missing ---
def test_missing_fusion_is_not_pass():
    import copy
    r = _passing_base()
    r2 = copy.deepcopy(r)
    del r2["sustained"]["tests"]["fusion"]
    r2["sustained"]["tests"]["per_agent"]["uav"]["steady_fps"] = 16.0
    g = G.evaluate(r2)
    assert g["verdict"] in (S.FAIL, S.INCONCLUSIVE)
    assert any(c["check"] == "sustained.fusion_rate" and not c["pass"]
               for c in g["checks"])


# --- release tamper detection ---
def test_release_tamper_detected(tmp_path):
    import json
    from nexus_bench import release as R
    (tmp_path / "profiles").mkdir()
    (tmp_path / "profiles" / "a.yaml").write_text("x: 1\n")
    (tmp_path / "nexus_bench").mkdir()
    (tmp_path / "nexus_bench" / "m.py").write_text("x=1\n")
    (tmp_path / "release").mkdir()
    (tmp_path / "release" / "release.json").write_text(json.dumps({
        "benchmark_source_sha256": "0" * 64, "gate_sha256": "1" * 64,
        "asset_manifest_sha256": "2" * 64}))
    ok, problems = R.verify_runtime(str(tmp_path))
    assert not ok and len(problems) >= 3


# --- qualify exit codes are authoritative ---
def test_qualify_exit_codes():
    from nexus_bench.qualify import VERDICT_EXIT
    assert VERDICT_EXIT == {"PASS": 0, "PASS_WITH_HEADROOM": 0, "FAIL": 2,
                            "INCONCLUSIVE": 3, "ABORTED": 4}


# --- report rows carry 4 states ---
def test_report_rows_four_states(tmp_path):
    from nexus_bench import report as R
    r = _passing_base()
    del r["accuracy"]["tests"]["map_comparison"]  # sole evidence gone
    results = dict(r, _profile={"cpu": {"brand": "X"}, "gpu": {"name": "G"},
                                "memory": {"ram_total_gb": 16}},
                   _config={"req_fps": 15, "duration_s": 600, "seed": 0},
                   _provenance={"git": {"sha": "a", "dirty": False},
                                "fingerprints": {"model": "abc123"}},
                   _power={"signals": {"ac_connected": {"value": True, "how": "measured"}}})
    results["_gate"] = G.evaluate({k: v for k, v in results.items() if "__run" not in k})
    assert results["_gate"]["verdict"] == S.INCONCLUSIVE  # missing, not failed hardware
    paths = R.write_all(results, tmp_path, req_fps=15)
    import pathlib
    html = pathlib.Path(paths["html"]).read_text()
    assert "INCONCLUSIVE" in html  # missing metric must not render red FAIL
    assert "Accuracy drop" in html
# --- nested sustained metrics steer worst-run selection ---
def test_runs_aggregate_nested_sustained():
    def sus_run(fps):
        return {"module": "sustained", "status": S.FULL,
                "tests": {"per_agent": {
                    "uav": {"steady_fps": fps, "min_window_fps": fps},
                    "rover": {"steady_fps": fps, "min_window_fps": fps}}},
                "errors": []}
    res = {"sustained__run1": sus_run(18.0), "sustained__run2": sus_run(9.0)}
    out = G.aggregate_runs(res)
    agg = out["sustained"]
    assert agg["tests"]["per_agent"]["uav"]["steady_fps"] == 9.0  # nested worst kept
    assert agg["tests"]["per_agent"]["uav"]["min_window_fps"] == 9.0

# --- Windows-safe file I/O: every text open declares UTF-8 ---
def test_all_text_opens_declare_encoding():
    """Regression for the Windows cp1252 crash (Ultralytics yaml ships a UTF-8
    emoji; locale-default decoding explodes). Binary opens excluded."""
    import pathlib
    import re
    bad = []
    for p in sorted(pathlib.Path("nexus_bench").glob("*.py")):
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if "open(" not in line or "Popen" in line or "os.path.abspath" in line:
                continue
            if re.search(r'"rb"|"wb"|\'rb\'', line):
                continue
            if "encoding=" not in line:
                bad.append(f"{p.name}:{i}: {line.strip()}")
    assert not bad, bad


def test_report_and_summary_generate(tmp_path):
    from nexus_bench import report as R
    r = _passing_base()
    for a in ("uav", "rover"):
        for k in ("steady_fps", "final_fps", "min_window_fps", "final_window_fps"):
            r["sustained"]["tests"]["per_agent"][a][k] = 20.0
    results = dict(r, _profile={"cpu": {"brand": "X"}, "gpu": {"name": "RTX 3050 Laptop GPU"},
                                "memory": {"ram_total_gb": 16}},
                   _config={"req_fps": 15, "duration_s": 600, "seed": 0},
                   _provenance={"git": {"sha": "abc", "dirty": False}},
                   _power={"signals": {"ac_connected": {"value": True, "how": "measured"}},
                           "coverage": {"note": "partial"}})
    results["_gate"] = G.evaluate({k: v for k, v in results.items() if "__run" not in k})
    paths = R.write_all(results, tmp_path, req_fps=15)
    import pathlib
    assert pathlib.Path(paths["html"]).exists() and pathlib.Path(paths["json"]).exists()
    html = pathlib.Path(paths["html"]).read_text()
    assert "PASS_WITH_HEADROOM" in html and "SUSTAINED" in html.upper()
    txt = R.plain_summary(results)
    assert "RESULT: PASS_WITH_HEADROOM" in txt and "WhatsApp" not in txt


# --- wizard config surface ---
def test_wizard_discovery_and_packaging(tmp_path, monkeypatch):
    from nexus_bench import wizard as W
    (tmp_path / "nexus.pt").write_bytes(b"m")
    (tmp_path / "uav.mp4").write_bytes(b"v")
    monkeypatch.chdir(tmp_path)
    found = W.discover()
    assert found["models"] == ["nexus.pt"] and found["videos"] == ["uav.mp4"]
    for name in ("r.html", "r.json", "r.csv", "e.log"):
        (tmp_path / name).write_text("x")
    folder, zipp = W.package_results(
        {"html": str(tmp_path / "r.html"), "json": str(tmp_path / "r.json"),
         "csv": str(tmp_path / "r.csv"), "log": str(tmp_path / "e.log")},
        {"_provenance": {"git": {"sha": "x"}}, "_gate": {"verdict": "FAIL", "checks": []},
         "_config": {"req_fps": 15, "duration_s": 1}})
    import os
    assert os.path.isdir(folder) and os.path.exists(zipp)
    assert os.path.exists(os.path.join(folder, "qualification_summary.txt"))
