"""Inference honesty: no weights -> UNSUPPORTED with zero latency numbers."""
import nexus_bench.yolo_util as yu
from nexus_bench.inference_bench import STATUS_UNSUPPORTED, _grid, run

CFG = {"seed": 0, "warmup": 1, "repeats": 2, "duration_s": 2, "device": "cpu",
       "imgsz": 320, "batch": 1, "precision": "fp32", "model": "",
       "video": "", "image_dir": "", "out": "reports"}

def test_missing_weights_reports_unsupported_not_numbers():
    r = run(CFG)
    assert r["status"] == STATUS_UNSUPPORTED
    assert r["tests"] == {}  # the guarantee: no fake inference numbers, ever

def test_missing_file_same(tmp_path):
    cfg = dict(CFG, model=str(tmp_path / "nope.pt"))
    r = run(cfg)
    assert r["status"] == STATUS_UNSUPPORTED and r["tests"] == {}

def test_grid_lists():
    cfg = dict(CFG, imgsz_list=[320, 640], batch_list=[1, 4])
    assert _grid(cfg) == ([320, 640], [1, 4])
    assert _grid(CFG) == ([320], [1])

def test_no_precision_flag_by_default():
    assert yu.precision_kwargs(object(), False) == {}

class _NewAPI:
    def predict(self, *a, **k):
        assert "quantize" in k  # new ultralytics path probed
        return []

class _OldAPI:
    def predict(self, *a, **k):
        if "quantize" in k:
            raise TypeError("unexpected keyword 'quantize'")
        return []

def test_precision_flag_probed_new_api():
    yu._FLAG = None
    assert yu.precision_kwargs(_NewAPI(), True) == {"quantize": True}

def test_precision_flag_falls_back_old_api():
    yu._FLAG = None
    assert yu.precision_kwargs(_OldAPI(), True) == {"half": True}
    yu._FLAG = None
