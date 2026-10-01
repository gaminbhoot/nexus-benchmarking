"""CLI: primary interface. Every module runs independently."""
import argparse
import json
import random
import sys

import numpy as np

from nexus_bench import (comms_bench, cpu_bench, gpu_bench, inference_bench,
                         integrated_bench, mapping3d_bench, memory_bench,
                         thermal_bench, vision_bench)
from nexus_bench import profiler, report
from nexus_bench.profiles import PROFILES, get

MODULES = {"system": None, "cpu": cpu_bench, "gpu": gpu_bench, "memory": memory_bench,
           "inference": inference_bench, "vision": vision_bench, "mapping3d": mapping3d_bench,
           "comms": comms_bench, "integrated": integrated_bench, "thermal": thermal_bench}

def _args():
    p = argparse.ArgumentParser(description="NEXUS compute benchmarking utility")
    p.add_argument("modules", nargs="*", default=["all"],
                   help=f"modules to run {sorted(MODULES)} or 'all' (default: all)")
    p.add_argument("--profile", default="smoke",
                   help=f"workload profile (default: smoke). Choices: {sorted(PROFILES)}")
    p.add_argument("--out", default="reports", help="report directory")
    p.add_argument("--req-fps", type=float, default=15, help="feasibility threshold FPS")
    p.add_argument("--model", default=None, help="YOLO weights path")
    p.add_argument("--video", default=None, help="video file (reserved for dataset runs)")
    p.add_argument("--image-dir", default=None, help="image dir (reserved for dataset runs)")
    p.add_argument("--imgsz", type=int, default=None)
    p.add_argument("--batch", type=int, default=None)
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
    cfg = get(a.profile, model=a.model, video=a.video, image_dir=a.image_dir,
              imgsz=a.imgsz, batch=a.batch, precision=a.precision, device=a.device,
              duration_s=a.duration_s, repeats=a.repeats, seed=a.seed)
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
