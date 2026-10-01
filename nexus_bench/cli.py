"""CLI: primary interface. Every module runs independently."""
import argparse
import json
import os
import random
import sys

import numpy as np

from nexus_bench import (comms_bench, cpu_bench, gpu_bench, inference_bench,
                         integrated_bench, mapping3d_bench, memory_bench,
                         pipeline_bench, reid_embed, thermal_bench,
                         tracking_bench, vision_bench)
from nexus_bench import profiler, report
from nexus_bench.profiles import PROFILES, get

MODULES = {"system": None, "cpu": cpu_bench, "gpu": gpu_bench, "memory": memory_bench,
           "inference": inference_bench, "reid": reid_embed, "tracking": tracking_bench,
           "vision": vision_bench, "pipeline": pipeline_bench, "mapping3d": mapping3d_bench,
           "comms": comms_bench, "integrated": integrated_bench, "thermal": thermal_bench}

def _load_profile(name_or_path, overrides):
    if os.path.isfile(name_or_path or ""):
        import yaml
        with open(name_or_path) as f:
            doc = yaml.safe_load(f) or {}
        base = doc.pop("profile_base", "smoke")
        if base not in PROFILES:
            raise KeyError(f"profile_base {base!r} unknown; choose from {sorted(PROFILES)}")
        cfg = get(base)
        allowed = set(cfg)
        for k, v in doc.items():
            if k == "modules":
                continue
            if k not in allowed:
                raise KeyError(f"unknown profile key {k!r} in {name_or_path}")
            cfg[k] = v
        mods = doc.get("modules")
        return cfg, mods
    cfg = get(name_or_path, **overrides)
    return cfg, None

def _args():
    p = argparse.ArgumentParser(description="NEXUS compute benchmarking utility")
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
    p.add_argument("--uav-video", default=None, dest="uav_video", help="UAV agent video (falls back to --video)")
    p.add_argument("--rover-video", default=None, dest="rover_video", help="rover agent video (falls back to --video)")
    p.add_argument("--image-dir", default=None, help="image directory replayed as a stream @30fps")
    p.add_argument("--imgsz", type=int, default=None)
    p.add_argument("--imgsz-list", nargs="+", type=int, default=None, dest="imgsz_list",
                   help="sweep, e.g. --imgsz-list 320 480 640 (inference module)")
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--batch-list", nargs="+", type=int, default=None, dest="batch_list",
                   help="sweep, e.g. --batch-list 1 2 4 (inference module)")
    p.add_argument("--precision", default=None, choices=["fp32", "fp16"])
    p.add_argument("--device", default=None, help="auto|cuda|mps|cpu")
    p.add_argument("--duration", type=float, default=None, dest="duration_s")
    p.add_argument("--repeats", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--list-profiles", action="store_true")
    return p.parse_args()

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
                     image_dir=a.image_dir, imgsz=a.imgsz, imgsz_list=a.imgsz_list,
                     batch=a.batch, batch_list=a.batch_list, precision=a.precision,
                     device=a.device, duration_s=a.duration_s, repeats=a.repeats,
                     seed=a.seed)
    try:
        cfg, yaml_mods = _load_profile(a.profile, overrides)
    except (KeyError, FileNotFoundError) as e:
        print(f"profile error: {e}", file=sys.stderr)
        return 2
    if yaml_mods and a.modules == ["all"]:
        mods = [m for m in yaml_mods if m in MODULES]
    # CLI flags always win over file/base values when explicitly given.
    for k, v in overrides.items():
        if v is not None and k in cfg:
            cfg[k] = v
    if "--out" in sys.argv:
        cfg["out"] = a.out
    random.seed(cfg["seed"]); np.random.seed(cfg["seed"] % (2**32))
    try:
        import torch
        torch.manual_seed(cfg["seed"])
    except Exception:
        pass
    print(f"[nexus-bench] profile={a.profile} modules={mods} device={cfg['device']}")
    results = {"_profile": profiler.profile(), "_config": cfg}
    cancelled = False
    for m in mods:
        if m == "system":
            continue  # profiler output already in _profile
        print(f"[nexus-bench] running {m} ...", flush=True)
        try:
            results[m] = MODULES[m].run(cfg)
        except KeyboardInterrupt:
            results[m] = {"module": m, "tests": {}, "errors": ["cancelled by user"]}
            cancelled = True
            break
        except Exception as e:
            results[m] = {"module": m, "tests": {}, "errors": [f"runner: {e}"]}
    paths = report.write_all(results, cfg.get("out", a.out), req_fps=a.req_fps)
    feas = results["_meta"]["feasibility"]
    print(f"[nexus-bench] feasibility: {feas}")
    print(f"[nexus-bench] wrote {paths['json']}\n[nexus-bench] wrote {paths['csv']}\n[nexus-bench] wrote {paths['html']}")
    return 130 if cancelled else 0

if __name__ == "__main__":
    raise SystemExit(main())
