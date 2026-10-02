"""CLI: supervising controller. Each module runs in a fresh worker process."""
import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import tempfile

import numpy as np

from nexus_bench import gate as gate_mod
from nexus_bench import power as power_mod
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
           "matrix": "nexus_bench.matrix_bench", "sustained": "nexus_bench.sustained_bench",
           "preflight": "nexus_bench.preflight_bench",
           "system": None}

def _load_profile(name_or_path, overrides):
    if os.path.isfile(name_or_path or ""):
        import yaml
        with open(name_or_path, encoding="utf-8") as f:
            doc = yaml.safe_load(f) or {}
        base = doc.pop("profile_base", "smoke")
        if base not in PROFILES:
            raise KeyError(f"profile_base {base!r} unknown; choose from {sorted(PROFILES)}")
        cfg = get(base)
        mods = doc.pop("modules", None)
        if mods == ["all"]:
            mods = [m for m in MODULES if m != "system"]  # [all] really means all
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
    p.add_argument("--extended", action="store_true", help="900 s sustained (with qualify)")
    p.add_argument("--yes", action="store_true", help="assume yes to prompts (with qualify)")
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
        with open(cfg_p, "w", encoding="utf-8") as f:
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
            with open(out_p, encoding="utf-8") as f:
                return json.load(f), tele
        return ({"module": submod, "status": S.FAILED, "tests": {},
                 "errors": [f"worker exited {proc.returncode} with no result file"]}, tele)

def main():
    a = _args()
    if a.list_profiles:
        print(json.dumps({k: v for k, v in PROFILES.items()}, indent=2, default=str))
        return 0
    mods = list(MODULES) if a.modules == ["all"] else a.modules
    if mods == ["wizard"]:
        from nexus_bench.wizard import run_wizard
        return run_wizard()
    if mods == ["doctor"]:
        from nexus_bench.doctor import main as doctor_main
        return doctor_main()
    if mods == ["qualify"] or mods == ["qualify-extended"]:
        from nexus_bench.qualify import run_qualify, VERDICT_EXIT
        _folder, _zipp, _results = run_qualify(
            extended=(mods == ["qualify-extended"] or a.extended),
            assume_yes=a.yes)
        return VERDICT_EXIT.get((_results.get("_gate") or {}).get("verdict"), 3)
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
        with open(a.gate, encoding="utf-8") as f:
            gate_cfg.update(yaml.safe_load(f) or {})
    gate_cfg["req_fps"] = a.req_fps
    cfg["gate_cfg_sha256"] = hashlib.sha256(
        json.dumps(gate_cfg, sort_keys=True, default=str).encode()).hexdigest()
    random.seed(cfg["seed"]); np.random.seed(cfg["seed"] % (2**32))
    _, _, code = controller_run(cfg, mods, gate_cfg, req_fps=a.req_fps, runs=a.runs,
                                timeout=(a.worker_timeout_s or cfg.get("worker_timeout_s")),
                                cblas=cfg.get("blas_threads"), out=cfg.get("out", a.out),
                                gate_strict=a.gate_strict,
                                header=f"[nexus-bench] profile={a.profile} modules={mods} "
                                       f"device={cfg['device']} runs={a.runs}")
    return code

def controller_run(cfg, mods, gate_cfg, req_fps=15, runs=1, timeout=None,
                   cblas=None, out="reports", gate_strict=False, progress=None,
                   header=None):
    """Supervised run usable by the CLI and the wizard. Returns (paths, results)."""
    import random as _random
    import numpy as _np
    timeout = timeout or max(180.0, (cfg.get("duration_s", 60) + 180))
    # The longest single module bounds the timeout: a 600 s sustained worker
    # must never be killed by a timeout derived from the short module duration.
    longest = max(cfg.get("duration_s", 60), cfg.get("sustained_duration_s", 0))
    timeout = max(timeout, longest + 300.0)
    if header:
        print(header + f" timeout={timeout}s")
    results = {"_profile": profiler.profile(), "_config": cfg,
               "_provenance": provenance.manifest(cfg),
               "_power": power_mod.collect(cfg.get("device", "auto"))}
    cancelled = False
    total = sum(runs for m in mods if m != "system")
    idx = 0
    for m in mods:
        if m == "system":
            continue
        for rep in range(runs):
            key = m if runs == 1 else f"{m}__run{rep + 1}"
            idx += 1
            if progress:
                progress(key, idx, total)
            else:
                print(f"[nexus-bench] running {key} (isolated worker) ...", flush=True)
            try:
                res, tele = _run_worker(MODULES[m], cfg, timeout, cblas)
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
    results = gate_mod.aggregate_runs(results)  # worst-case across --runs, individuals kept
    gate_res = gate_mod.evaluate({k: v for k, v in results.items() if "__run" not in k}, gate_cfg)
    results["_gate"] = gate_res
    paths = report.write_all(results, out, req_fps=req_fps)
    print(f"[nexus-bench] GATE: {gate_res['verdict']}")
    for c in gate_res["checks"]:
        if not c["pass"]:
            print(f"  FAIL {c['check']}: {c['detail']}")
    print(f"[nexus-bench] wrote {paths['json']}\n[nexus-bench] wrote {paths['csv']}\n[nexus-bench] wrote {paths['html']}")
    code = 130 if cancelled else (3 if (gate_strict and gate_res["verdict"] not in
                                        (S.PASS, S.PASS_WITH_HEADROOM)) else 0)
    return paths, results, code

if __name__ == "__main__":
    raise SystemExit(main())
