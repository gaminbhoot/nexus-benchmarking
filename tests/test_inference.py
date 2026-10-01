"""Inference honesty: no weights -> UNSUPPORTED with zero latency numbers."""
import nexus_bench.yolo_util as yu
from nexus_bench import statuses as S
from nexus_bench.inference_bench import _grid, run

CFG = {"seed": 0, "warmup": 1, "repeats": 2, "duration_s": 2, "device": "cpu",
       "imgsz": 320, "batch": 1, "precision": "fp32", "model": "",
       "video": "", "image_dir": "", "out": "reports"}

def test_missing_weights_reports_unsupported_not_numbers():
    r = run(CFG)
    assert r["status"] == S.UNSUPPORTED
    assert r["tests"] == {}  # the guarantee: no fake inference numbers, ever

def test_missing_file_same(tmp_path):
    cfg = dict(CFG, model=str(tmp_path / "nope.pt"))
    r = run(cfg)
    assert r["status"] == S.UNSUPPORTED and r["tests"] == {}

def test_grid_lists():
    cfg = dict(CFG, imgsz_list=[320, 640], batch_list=[1, 4])
    assert _grid(cfg) == ([320, 640], [1, 4])
    assert _grid(CFG) == ([320], [1])

def test_no_precision_flag_by_default():
    assert yu.precision_kwargs(False) == {}

def test_precision_flag_is_explicit_fp16():
    assert yu.precision_kwargs(True) == {"quantize": "fp16"}

def test_effective_precision_reads_predictor_args():
    class Args(dict):
        pass
    class Pred:
        args = {"quantize": "fp16"}
    class M:
        predictor = Pred()
    assert yu.effective_precision(M()) == "fp16"
    Pred.args = {"quantize": "fp32"}
    assert yu.effective_precision(M()) == "fp32"

def test_check_effective_fail_closed():
    rec = {"requested": "fp16", "effective": "unverified"}
    class Pred:
        args = {"quantize": "fp32"}
    class M:
        predictor = Pred()
    assert yu.check_effective(M(), rec) is False
    assert rec["effective"] == "fp32"
