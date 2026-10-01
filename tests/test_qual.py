"""Qualification machinery: isolation, supervisor, stream, states, provenance."""
import numpy as np

from nexus_bench import provenance, statuses as S
from nexus_bench.accuracy_bench import _agreement
from nexus_bench.cli import _run_worker
from nexus_bench.stream import PacedSource

CFG = {"seed": 0, "warmup": 1, "repeats": 2, "duration_s": 2, "device": "cpu",
       "imgsz": 320, "batch": 1, "precision": "fp32", "model": "", "reid_model": "",
       "video": "", "uav_video": "", "rover_video": "", "image_dir": "",
       "val_data": "", "imgsz_list": None, "batch_list": None, "agent_frames": 6,
       "target_fps": 30.0, "integrated_mode": "throughput", "throttle_temp_c": 83,
       "req_fps": 15, "blas_threads": None, "out": "reports"}

def test_worker_runs_module_isolated():
    res, tele = _run_worker("nexus_bench.comms_bench", CFG, timeout_s=120, cblas=None)
    assert res["module"] == "comms" and "tests" in res
    assert isinstance(tele, dict)

def test_supervisor_timeout_aborts():
    res, _ = _run_worker("nexus_bench.comms_bench", CFG, timeout_s=0, cblas=None)
    assert res["status"] == S.ABORTED  # wait(0) always times out -> killed

def test_paced_source_emits_and_tags_synthetic():
    src = PacedSource(CFG, target_fps=60.0).start()
    got = [src.get(timeout=3.0) for _ in range(3)]
    src.stop()
    assert all(g not in (None, "retry") for g in got)
    assert src.emitted >= 3 and not src.real_pixels

def test_pipeline_partial_without_model():
    import nexus_bench.pipeline_bench as p
    r = p.run(dict(CFG, duration_s=2, target_fps=30.0))
    assert r["status"] == S.PARTIAL  # fallback detector: informational, never gate-grade
    assert "end_to_end_ms" in r["tests"]

def test_thermal_fails_closed_on_bad_model(tmp_path):
    import nexus_bench.thermal_bench as t
    r = t.run(dict(CFG, model=str(tmp_path / "nope.pt"), duration_s=2))
    assert r["status"] == S.FAILED and r["tests"] == {}

def test_inference_unsupported_status_constant():
    import nexus_bench.inference_bench as inf
    assert inf.run(CFG)["status"] == S.UNSUPPORTED

def test_provenance_manifest_keys():
    m = provenance.manifest(CFG)
    assert {"benchmark_version", "git", "packages", "model_sha256",
            "config_sha256", "power", "timestamp_utc"} <= set(m)

def test_agreement_perfect_match():
    b = np.array([[10, 10, 50, 50, 0, 0.9]])
    a = _agreement(b, b.copy())
    assert a["matched_rate"] == 1.0 and a["mean_iou"] == 1.0 and a["count_delta"] == 0

def test_agreement_empty_is_perfect():
    a = _agreement(np.zeros((0, 6)), np.zeros((0, 6)))
    assert a["matched_rate"] == 1.0 and a["count_delta"] == 0

def test_agreement_miss_detected():
    b = np.array([[10, 10, 50, 50, 0, 0.9]])
    a = _agreement(b, np.zeros((0, 6)))
    assert a["matched_rate"] == 0.0 and a["count_delta"] == -1
