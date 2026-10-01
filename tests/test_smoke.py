"""Smoke tests: every module runs independently and serializes to reports."""
import json
import subprocess
import sys
from pathlib import Path

import nexus_bench.cpu_bench as cpu_bench
import nexus_bench.comms_bench as comms_bench
import nexus_bench.gpu_bench as gpu_bench
import nexus_bench.mapping3d_bench as mapping3d_bench
import nexus_bench.memory_bench as memory_bench
import nexus_bench.vision_bench as vision_bench
from nexus_bench import profiler, report
from nexus_bench.profiles import PROFILES, get
from nexus_bench.stats import summarize

CFG = {"seed": 0, "warmup": 1, "repeats": 2, "duration_s": 2,
       "device": "cpu", "imgsz": 320, "batch": 1, "precision": "fp32",
       "model": "", "video": "", "image_dir": "", "out": "reports"}

def test_stats_percentiles():
    s = summarize([10.0, 20.0, 30.0, 40.0])
    assert s["n"] == 4 and s["median_ms"] == 25.0
    assert s["p95_ms"] > s["median_ms"] and s["p99_ms"] >= s["p95_ms"]

def test_profiler_never_hardcodes():
    p = profiler.profile()
    assert {"os", "cpu", "memory", "gpu", "accelerators"} <= set(p)

def test_profiles_known():
    assert {"smoke", "uav", "rover", "full"} <= set(PROFILES)
    assert get("smoke")["imgsz"] >= 1

def test_cpu_gpu_memory_comms_mapping():
    for mod in (cpu_bench, gpu_bench, memory_bench, comms_bench, mapping3d_bench):
        r = mod.run(CFG)
        assert "tests" in r and isinstance(r.get("errors"), list)

def test_vision_reference():
    r = vision_bench.run(CFG)
    assert "detect_track_ms" in r["tests"]

def test_report_serializes(tmp_path):
    results = {"_profile": profiler.profile(), "cpu": cpu_bench.run(CFG)}
    paths = report.write_all(results, tmp_path, req_fps=15)
    for k in ("json", "csv", "html"):
        assert Path(paths[k]).exists()
    assert json.loads(Path(paths["json"]).read_text())["_meta"]["feasibility"]

def test_cli_smoke(tmp_path):
    out = subprocess.run([sys.executable, "-m", "nexus_bench.cli", "system", "comms",
                          "--profile", "smoke", "--out", str(tmp_path)],
                         capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
