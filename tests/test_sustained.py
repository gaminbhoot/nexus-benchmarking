"""Sustained math + accounting identities (fast, no 10-minute runs here)."""
import numpy as np

from nexus_bench.stream import FrameAccount, PacedSource, drop_pct
from nexus_bench.sustained_bench import _growth, _series_stats, _trajectory, _window_drops

def test_conservation_identity():
    a = FrameAccount()
    a.generated = 100
    a.acquired = 100
    a.enqueued = 97
    a.dequeued = 95
    a.processed = 93
    a.delivered = 93
    a.producer_drops = 3
    a.queue_drops = 0
    a.processing_failures = 2
    a.finalize(remaining_in_queue=2)  # 95-93-2+2 = 2 unfinished
    ok, books = a.check_conservation()
    assert ok, books  # 100 = 93+3+0+2+2

def test_conservation_detects_double_count():
    a = FrameAccount()
    a.generated = 10
    a.delivered = 10
    a.producer_drops = 1  # impossible: 10 != 10+1
    a.finalize(0)
    ok, books = a.check_conservation()
    assert not ok and books["generated"] == 10 and books["accounted"] == 11

def test_drop_definition_single():
    assert drop_pct(1000, 950) == 5.0
    assert drop_pct(0, 0) == 0.0
    assert drop_pct(100, 100) == 0.0

def test_windowize_chronology():
    from nexus_bench.sustained_bench import _windowize
    t0 = 1000.0
    ev = [(t0 + 10, 1.0), (t0 + 70, 2.0), (t0 + 130, 3.0)]
    w = _windowize(ev, t0, 60, 180)
    assert w == {0: [1.0], 1: [2.0], 2: [3.0]}

def test_window_drops_use_real_timestamps_not_uniform():
    # Window 0 collapses (300/900 delivered), window 1 is healthy (890/900).
    # A uniform spread would report ~5.5% everywhere and hide window 0.
    t0 = 1000.0
    gen0 = [t0 + i * 60.0 / 900 for i in range(900)]
    gen = gen0 + [t + 60 for t in gen0]
    ev = [(t, 10.0) for t in gen0[:300]]  # 300 delivered in w0
    ev += [(t + 60, 10.0) for t in gen0[:890]]  # 890 delivered in w1
    drop_w, gen_w = _window_drops(gen, ev, t0, 60, 120)
    assert gen_w == [900, 900]
    assert drop_w[0] == round(100 * (900 - 300) / 900, 2)  # 66.67, not ~5.5
    assert drop_w[1] == round(100 * (900 - 890) / 900, 2)
    assert drop_w[0] > 50 > drop_w[1]

def test_series_stats_order_preserved():
    w = {0: [18.0], 1: [15.0], 2: [9.0]}
    s = _series_stats(w, 180, 60, lambda v: sum(v) / len(v))
    assert s == [18.0, 15.0, 9.0]  # a collapse to 9 FPS stays visible, never averaged away

def test_growth_detects_leak():
    g = _growth(1000.0, 1500.0, 300)
    assert g["growth"] == 500.0 and g["leak_suspected"] is True

def test_growth_no_false_alarm():
    g = _growth(1000.0, 1050.0, 300)
    assert g["leak_suspected"] is False

def test_sustained_needs_model_fast():
    import nexus_bench.sustained_bench as s
    from nexus_bench import statuses as S
    r = s.run({"seed": 0, "warmup": 1, "repeats": 1, "duration_s": 1, "device": "cpu",
               "imgsz": 320, "batch": 1, "precision": "fp32", "model": "",
               "sustained_duration_s": 600, "sustained_window_s": 60,
               "target_fps": 15.0, "concurrency": "separate-contexts",
               "leak_threshold_mb": 300, "req_fps": 15})
    assert r["status"] == S.UNSUPPORTED and r["tests"] == {}

def test_timestamps_present_in_meta():
    src = PacedSource({"seed": 0, "agent_frames": 2}, target_fps=60.0).start()
    item = src.get(timeout=5.0)
    src.stop()
    assert item not in (None, "retry")
    _, meta = item
    for k in ("t_acquire_start", "t_decode_done", "t_capture", "t_enqueue", "t_dequeue"):
        assert k in meta
    assert meta["t_decode_done"] >= meta["t_acquire_start"]
    assert meta["t_dequeue"] >= meta["t_enqueue"] >= meta["t_capture"]
