# NEXUS Benchmarking

Hardware-aware benchmarking utility for the NEXUS autonomous air–ground robotic system.
Establishes realistic local compute capacity, finds bottlenecks, and reports which
NEXUS workloads fit the machine — measured numbers only, no hardcoded hardware claims.

## Quick start

```bash
pip install -e ".[dev]"
nexus-bench --profile smoke                       # validate everything in seconds
nexus-bench --profile uav --model yolo11n.pt      # UAV perception with real weights
nexus-bench cpu gpu memory --profile rover        # individual modules
nexus-bench --list-profiles
```

Reports land in `reports/`: JSON (automation), CSV (detail), HTML (charts + feasibility).

## Modules

| CLI name | Covers |
|---|---|
| `system` | profiler only: CPU/GPU/RAM/VRAM/storage/OS/drivers/CUDA |
| `cpu` | single/multi-thread, preprocess, decode, sustained |
| `gpu` | matmul, conv, bandwidth (+fp16/int8 on CUDA) |
| `memory` | RAM throughput, VRAM pressure, peak RSS, OOM recovery |
| `inference` | YOLO imgsz/batch/precision grid (synthetic proxy without weights) |
| `vision` | detection+tracking, Re-ID, optical flow, video pipeline |
| `mapping3d` | point-cloud ops, ORB features, visual-odometry sim |
| `comms` | MAVLink-style parse, telemetry throughput, link-latency sim |
| `integrated` | UAV+rover concurrently: contention ratio + mission sim |
| `thermal` | sustained load, degradation %, throttling flag |

## Profiles

`smoke` (validation) · `uav` · `rover` · `uav_rover_concurrent` · `tracking` ·
`reid_tracking` · `mapping` · `comms_fusion` · `full` (sustained 60 s).

Override anything inline:
```bash
nexus-bench --profile uav --imgsz 640 --batch 1 --precision fp16 --device cuda --duration 30
```

## Safety

Configurable temp/RAM/VRAM/duration caps abort runs gracefully; CUDA OOMs are
caught with cache cleanup; Ctrl-C cancels with partial results. Missing sensors
(NVML, temp, ROS 2) are reported as unavailable, never fatal. No firmware,
overclock, or voltage changes — ever.

## Layout

```
nexus_bench/   profiler monitor stats safety profiles cli report + *_bench.py
tests/         pytest smoke suite (all modules + CLI + serialization)
profiles/      reproducible YAML benchmark profiles
reports/       generated artifacts (git-ignored)
```
