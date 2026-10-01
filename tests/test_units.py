"""Profiles (builtin + YAML), tracker behavior, MAVLink framing, safety guard."""
import numpy as np

from nexus_bench.cli import _load_profile
from nexus_bench.comms_bench import mav_pack, mav_parse
from nexus_bench.profiles import get
from nexus_bench.safety import Guard, Limits
from nexus_bench.tracking_bench import Tracker, _greedy_match, make_scenario

# --- profiles ---
def test_yaml_profile_loads_and_cli_wins(tmp_path):
    p = tmp_path / "q.yaml"
    p.write_text("profile_base: smoke\nimgsz: 416\nmodules: [cpu]\n")
    cfg, mods, gate_cfg = _load_profile(str(p), {"imgsz": None})
    assert cfg["imgsz"] == 416 and mods == ["cpu"] and gate_cfg is None

def test_yaml_unknown_key_rejected(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("profile_base: smoke\nnope: 1\n")
    try:
        _load_profile(str(p), {})
    except KeyError:
        return
    raise AssertionError("unknown YAML key must be rejected")

def test_yaml_bad_base_rejected(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("profile_base: nope\n")
    try:
        _load_profile(str(p), {})
    except KeyError:
        return
    raise AssertionError("unknown profile_base must be rejected")

def test_builtin_unknown_rejected():
    try:
        get("nope")
    except KeyError:
        return
    raise AssertionError("unknown builtin profile must be rejected")

# --- tracker ---
def test_greedy_assignment_optimal_small():
    cost = np.array([[1.0, 5.0], [5.0, 1.0]])
    m, u_r, u_c = _greedy_match(cost, 2.0)
    assert sorted(m) == [(0, 0), (1, 1)] and not u_r and not u_c

def test_greedy_respects_gate():
    cost = np.array([[10.0]])
    m, u_r, u_c = _greedy_match(cost, 2.0)
    assert m == [] and u_r == [0] and u_c == [0]

def test_scenario_deterministic():
    a = make_scenario(seed=7)
    b = make_scenario(seed=7)
    assert a[0][0] == b[0][0] and len(a) == len(b)

def test_tracker_holds_linear_targets():
    frames = make_scenario(n_targets=3, n_frames=30, seed=3)
    tr = Tracker()
    live_counts = []
    for dets, _ in frames:
        feats = [np.zeros(128) for _ in dets]  # motion-only here; appearance tested live
        live_counts.append(len(tr.step(dets, feats)))
    assert max(live_counts) >= 2  # tracks confirm and persist through the occlusion gap
    assert tr._next <= 6  # no ID explosion on 3 targets

# --- mavlink ---
def test_mavlink_roundtrip():
    f = bytes(_heartbeat_for_test(5))
    msgid, payload, seq, consumed = mav_parse(bytearray(f))
    assert (msgid, seq, consumed) == (0, 5, len(f)) and len(payload) == 9

def _heartbeat_for_test(seq):
    import struct
    from nexus_bench.comms_bench import mav_pack
    return mav_pack(0, struct.pack("<BBBI BB", 2, 0, 0, 0, 4, 3), seq)

def test_mavlink_rejects_corruption():
    f = bytearray(_heartbeat_for_test(1))
    f[12] ^= 0xFF
    try:
        mav_parse(f)
    except ValueError:
        return
    raise AssertionError("corrupted frame must fail CRC")

def test_mavlink_rejects_truncated():
    try:
        mav_parse(bytearray(b"\xfd\x09"))
    except ValueError:
        return
    raise AssertionError("truncated frame must raise")

# --- guard ---
def test_guard_duration_abort():
    import time
    g = Guard(Limits(max_duration_s=0.01))
    time.sleep(0.02)
    assert not g.ok({}) and "duration" in g.reason

def test_guard_temp_abort():
    g = Guard(Limits(max_temp_c=80.0))
    assert not g.ok({"gpu_temp_c": 90.0})

def test_guard_ok_normal():
    assert Guard().ok({"ram_pct": 10.0})
