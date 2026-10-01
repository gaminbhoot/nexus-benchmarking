"""CLI: supervising controller. Each module runs in a fresh worker process."""
import argparse
import json
import os
import random
import subprocess
import sys
import tempfile

import numpy as np

from nexus_bench import gate as gate_mod
from nexus_bench import profiler, provenance, report
from nexus_bench import statuses as S
from nexus_bench.monitor import Monitor
from nexus_bench.profiles import PROFILES, get

MODULES = {"cpu": "nexus_bench.cpu_bench", "gpu": "nexus_bench.gpu_bench",
           "memory": "nexus_bench.memory_bench", "inference": "nexus_bench.inference_bench",
           "reid": "nexus_bench.reid_embed", "tracking": "nexus_bench.tracking_bench",
           "vision": "nexus_bench.vision_bench", "pipeline": "nexus_bench.pipeline_bench",
           "mapping3d": "nexus_bench.mapping3d_bench", "comms": "nexus_bench.comms_bench",
           "integrated": "nexus_bench.integrated_bench", "thermal": "nexus_bench.thermal_bench",
           "backends": "nexus_bench.backends_bench", "accuracy": "nexus_bench.accuracy_bench",
           "coldstart": "nexus_bench.coldstart_bench", "decode": "nexus_bench.decode_bench",
           "matrix": "nexus_bench.matrix_bench",
           "system": None}

def _load_profile(name_or_path, overrides):
    if os.path.isfile(name_or_path or ""):
        import yaml
        with open(name_or_path) as f:
            doc = yaml.safe_load(f) or {}
        base = doc.pop("profile_base", "smoke")
        if base not in PROFILES:
            raise KeyError(f"profile_base {base!r} unknown; choose from {sorted(PROFILES)}")
        cfg = get(base)
        mods = doc.pop("modules", None)
        gate_cfg = doc.pop("gate", None)
        allowed = set(cfg)
        for k, v in doc.items():
            if k not in allowed:
                raise KeyError(f"unknown profile key {k!r} in {name_or_path}")
            cfg[k] = v
        return cfg, mods, gate_cfg
    return get(name_or_path, **overrides), None, None

def _args():
    p = argparse.ArgumentParser(description="NEXUS compute benchmarking utility (supervising controller)")
    p.add_argument("modules", nargs="*", default=["all"],
                   help=f"modules to run {sorted(MODULES)} or 'all' (default: all)")
    p.add_argument("--profile", default="smoke",
                   help=f"builtin profile {sorted(PROFILES)} OR path to a YAML profile")
    p.add_argument("--out", default="reports", help="report directory")
    p.add_argument("--req-fps", type=float, default=15, help="feasibility threshold FPS")
    p.add_argument("--model", default=None, help="YOLO weights path (.pt)")
    p.add_argument("--reid-model", default=None, dest="reid_model",
                   help="Re-ID embedding state_dict compatible with the reference net")
    p.add_argument("--video", default=None, help="video file for pipeline/integrated agents")
    p.add_argument("--uav-video", default=None, dest="uav_video")
    p.add_argument("--rover-video", default=None, dest="rover_video")
    p.add_argument("--image-dir", default=None, help="image directory replayed as a stream @30fps")
    p.add_argument("--val-data", default=None, dest="val_data", help="YOLO data yaml for accuracy mAP")
    p.add_argument("--imgsz", type=int, default=None)
    p.add_argument("--imgsz-list", nargs="+", type=int, default=None, dest="imgsz_list")
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--batch-list", nargs="+", type=int, default=None, dest="batch_list")
    p.add_argument("--precision", default=None, choices=["fp32", "fp16"])
    p.add_argument("--device", default=None, help="auto|cuda|mps|cpu")
    p.add_argument("--duration", type=float, default=None, dest="duration_s")
    p.add_argument("--repeats", type=int, default=None)
    p.add_argument("--runs", type=int, default=1, help="independent repetitions per module (variance)")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--blas-threads", type=int, default=None, dest="blas_threads",
                   help="pin worker BLAS threads (default: inherit)")
    p.add_argument("--worker-timeout", type=float, default=None, dest="worker_timeout_s")
    p.add_argument("--gate", default=None, help="path to gate YAML (acceptance criteria)")
    p.add_argument("--gate-strict", action="store_true", dest="gate_strict",
                   help="exit 3 unless the gate verdict is PASS/PASS_WITH_HEADROOM")
    p.add_argument("--list-profiles", action="store_true")
    return p.parse_args()

def _worker_env(cblas):
    env = dict(os.environ)
    if cblas:
        for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            env[k] = str(cblas)
    return env

def _run_worker(submod, cfg, timeout_s, cblas):
    """Fresh process per module. Returns (result_dict, controller_telemetry)."""
    with tempfile.TemporaryDirectory() as td:
        cfg_p = os.path.join(td, "cfg.json")
        out_p = os.path.join(td, "res.json")
        with open(cfg_p, "w") as f:
            json.dump(cfg, f, default=str)
        mon = Monitor()
        mon.__enter__()
        proc = subprocess.Popen([sys.executable, "-m", "nexus_bench.worker",
                                 submod, cfg_p, out_p], env=_worker_env(cblas))
        try:
            proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=30)
            mon.stop()
            return ({"module": submod, "status": S.ABORTED, "tests": {},
                     "errors": [f"supervisor timeout after {timeout_s}s (worker killed)"]},
                    mon.summary())
        mon.stop()
        tele = mon.summary()
        if os.path.exists(out_p):
            with open(out_p) as f:
                return json.load(f), tele
        return ({"module": submod, "status": S.FAILED, "tests": {},
                 "errors": [f"worker exited {proc.returncode} with no result file"]}, tele)

def main():
    a = _args()
    if a.list_profiles:
        print(json.dumps({k: v for k, v in PROFILES.items()}, indent=2, default=str))
        return 0
    mods = list(MODULES) if a.modules == ["all"] else a.modules
    for m in mods:
        if m not in MODULES:
            print(f"unknown module {m!r}; choose from {sorted(MODULES)}", file=sys.stderr)
            return 2
    overrides = dict(model=a.model, reid_model=a.reid_model, video=a.video,
                     uav_video=a.uav_video, rover_video=a.rover_video,
                     image_dir=a.image_dir, val_data=a.val_data, imgsz=a.imgsz,
                     imgsz_list=a.imgsz_list, batch=a.batch, batch_list=a.batch_list,
                     precision=a.precision, device=a.device, duration_s=a.duration_s,
                     repeats=a.repeats, runs=a.runs, seed=a.seed, blas_threads=a.blas_threads)
    try:
        cfg, yaml_mods, yaml_gate = _load_profile(a.profile, overrides)
    except (KeyError, FileNotFoundError) as e:
        print(f"profile error: {e}", file=sys.stderr)
        return 2
    if yaml_mods and a.modules == ["all"]:
        mods = [m for m in yaml_mods if m in MODULES]
    for k, v in overrides.items():
        if k == "runs" and "--runs" not in sys.argv:
            continue  # YAML-declared repeat counts survive unless CLI overrides
        if v is not None and k in cfg:
            cfg[k] = v
    if "--out" in sys.argv:
        cfg["out"] = a.out
    gate_cfg = dict(gate_mod.DEFAULT_GATE)
    if yaml_gate:
        gate_cfg.update(yaml_gate)
    if a.gate:
        import yaml
        with open(a.gate) as f:
            gate_cfg.update(yaml.safe_load(f) or {})
    gate_cfg["req_fps"] = a.req_fps
    random.seed(cfg["seed"]); np.random.seed(cfg["seed"] % (2**32))
    timeout = a.worker_timeout_s or cfg.get("worker_timeout_s") or max(180.0, (cfg.get("duration_s", 60) + 180))
    print(f"[nexus-bench] profile={a.profile} modules={mods} device={cfg['device']} "
          f"runs={a.runs} timeout={timeout}s")
    results = {"_profile": profiler.profile(), "_config": cfg,
               "_provenance": provenance.manifest(cfg)}
    cancelled = False
    for m in mods:
        if m == "system":
            continue
        for rep in range(a.runs):
            key = m if a.runs == 1 else f"{m}__run{rep + 1}"
            print(f"[nexus-bench] running {key} (isolated worker) ...", flush=True)
            try:
                res, tele = _run_worker(MODULES[m], cfg, timeout, cfg.get("blas_threads"))
                res["controller_telemetry"] = tele
                results[key] = res
            except KeyboardInterrupt:
                results[key] = {"module": m, "status": S.ABORTED, "tests": {},
                                "errors": ["cancelled by user (supervisor)"]}
                for m2 in mods[mods.index(m) + 1:]:
                    if m2 != "system":
                        results[m2] = {"module": m2, "status": S.NOT_RUN, "tests": {},
                                       "errors": ["not reached after cancellation"]}
                cancelled = True
                break
            except Exception as e:
                results[key] = {"module": m, "status": S.FAILED, "tests": {},
                                "errors": [f"supervisor: {e}"]}
        if cancelled:
            break
    gate_res = gate_mod.evaluate({k: v for k, v in results.items() if "__run" not in k}, gate_cfg)
    results["_gate"] = gate_res
    paths = report.write_all(results, cfg.get("out", a.out), req_fps=a.req_fps)
    print(f"[nexus-bench] GATE: {gate_res['verdict']}")
    for c in gate_res["checks"]:
        if not c["pass"]:
            print(f"  FAIL {c['check']}: {c['detail']}")
    print(f"[nexus-bench] wrote {paths['json']}\n[nexus-bench] wrote {paths['csv']}\n[nexus-bench] wrote {paths['html']}")
    if cancelled:
        return 130
    if a.gate_strict and gate_res["verdict"] not in (S.PASS, S.PASS_WITH_HEADROOM):
        return 3
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
