"""Selectable NEXUS workload profiles. Conservative defaults for 4GB VRAM."""
from copy import deepcopy

BASE = {
    "seed": 0,
    "warmup": 3,
    "repeats": 20,
    "duration_s": 20,
    "device": "auto",       # auto | cuda | mps | cpu
    "imgsz": 640,
    "batch": 1,
    "precision": "fp32",    # fp32 | fp16
    "model": "",            # path to YOLO weights; empty = module reports unsupported
    "reid_model": "",        # optional embedding state_dict for the reid module
    "video": "",
    "uav_video": "",         # falls back to video
    "rover_video": "",       # falls back to video
    "image_dir": "",
    "imgsz_list": None,      # optional sweep, e.g. [320, 480, 640]
    "batch_list": None,      # optional sweep, e.g. [1, 2, 4]
    "agent_frames": 60,      # frames per agent in throughput-mode runs
    "target_fps": 15.0,      # paced input rate for realtime pipeline/integrated/matrix
    "target_fps_list": None, # matrix sweep, e.g. [15, 30]
    "integrated_mode": "realtime",  # realtime | throughput
    "val_data": "",          # YOLO data yaml for accuracy mAP (optional)
    "blas_threads": None,    # pin worker BLAS threads (None = inherit)
    "worker_timeout_s": None,# supervisor kill timeout per module (None = auto)
    "runs": 1,               # independent repetitions (variance)
    "throttle_temp_c": 83,   # evidence threshold for the thermal verdict
    "req_fps": 15,
    "out": "reports",
}

PROFILES = {
    # Quick smoke: validates every module in seconds.
    "smoke": {**BASE, "repeats": 3, "warmup": 1, "duration_s": 3, "imgsz": 320},
    # Per-platform perception presets.
    "uav": {**BASE, "imgsz": 640, "repeats": 30},
    "rover": {**BASE, "imgsz": 480, "repeats": 30},
    "uav_rover_concurrent": {**BASE, "imgsz": 640, "repeats": 20, "duration_s": 30},
    "tracking": {**BASE, "imgsz": 640, "repeats": 20},
    "reid_tracking": {**BASE, "imgsz": 640, "repeats": 10},
    "mapping": {**BASE, "repeats": 10},
    "comms_fusion": {**BASE, "repeats": 20},
    # Full stack: everything, sustained.
    "full": {**BASE, "repeats": 50, "duration_s": 60},
}

def get(name, **overrides):
    if name not in PROFILES:
        raise KeyError(f"unknown profile {name!r}; choose from {sorted(PROFILES)}")
    cfg = deepcopy(PROFILES[name])
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    return cfg

def resolve_device(want="auto"):
    import torch
    if want == "auto":
        if torch.cuda.is_available():
            return "cuda"
        try:
            if torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"
    return want
