# NEXUS Benchmarking

Hardware-aware benchmarking utility for the NEXUS autonomous air–ground robotic system.
Establishes realistic local compute capacity, finds bottlenecks, and reports which
NEXUS workloads fit the machine — measured numbers only, no hardcoded hardware claims.

## Quick start

```bash
pip install -e ".[dev]"
nexus-bench --profile smoke                       # validate everything in seconds
nexus-bench --profile profiles/rtx3050_uav.yaml --model yolo11n.pt
nexus-bench inference memory --profile profiles/rtx3050_4gb.yaml --model nexus_best.pt \
  --precision fp16                                # VRAM boundary sweep on the 3050
nexus-bench pipeline --video drone.mp4 --model yolo11n.pt --req-fps 15
nexus-bench --list-profiles
```

`--profile` takes a builtin name (`smoke uav rover uav_rover_concurrent tracking
reid_tracking mapping comms_fusion full`) or a YAML file (`profile_base` + overrides).
Reports land in `reports/`: JSON (automation), CSV (detail), HTML (charts + feasibility).

## Methodology: what each number actually is

| Module | Status |
|---|---|
| CUDA GEMM TFLOPS, bandwidth, VRAM ladder | Synthetic microbenchmark |
| YOLO inference grid (imgsz × batch × precision) | Actual NEXUS component — requires `--model`; without weights the module reports `unsupported`, never proxy numbers |
| Re-ID embedding (crop → CNN → gallery match) | Actual component at reference scale (lightweight CNN); pass `--reid-model` for production weights |
| DeepSORT-style tracking (Kalman + appearance cascade, greedy assignment) | Actual component (greedy, not Hungarian — no scipy dep) |
| UAV/rover video pipeline (decode→detect→track→telemetry) | Actual NEXUS workload on `--video`/`--image-dir`; synthetic stream otherwise (labelled) |
| UAV + rover concurrent + fusion | **Primary NEXUS benchmark** |
| Two-view VO (ORB→E→pose→triangulate vs ground truth) | Component benchmark with geometric verification |
| MAVLink v2 serialize/parse + UDP loopback RTT | Real framing/stack measurement (localhost = stack cost; target the vehicle for link numbers) |
| `sim_link_model` | Simulation — never a measurement |

Key rules enforced in code: `--precision fp16` casts weights AND passes `half=True`;
batch tests pass all N images in ONE `predict` call; throttling is claimed only on
temp + clock-drop + load evidence; feasibility uses p95 + deadline-miss % + drops +
measured FPS vs `--req-fps` (median never decides alone). See `tests/` — including
`test_inference.py`, which pins the no-weights honesty guarantee.

## Modules

`system cpu gpu memory inference reid tracking vision pipeline mapping3d comms integrated thermal`

## Known limitations

- No scipy → greedy (not Hungarian) association; swap in `linear_sum_assignment` if scipy exists.
- No pymavlink/ROS 2 → hand-rolled MAVLink v2 framing; `rclpy` probed, not benchmarked.
- No TensorRT path yet (probe reports presence; add an engine runner for INT8 on the 3050).
- Reference Re-ID CNN is small by design — compare relatively, or bring your weights.
- Laptop power-mode/firmware caps are read (power limit via nvidia-smi), never changed.

## Safety

Temp/RAM/VRAM/duration caps abort gracefully; CUDA OOMs are caught with cache
cleanup; Ctrl-C cancels with partial results. Missing sensors are "unavailable",
never fatal. No firmware, overclock, or voltage changes — ever.

## Layout

```
nexus_bench/   profiler monitor stats safety profiles cli report + *_bench.py
tests/         pytest suite: smoke + metrics/feasibility + units + inference honesty
profiles/      reproducible YAML benchmark profiles
reports/       generated artifacts (git-ignored)
examples/sample_report/  committed example output
```
