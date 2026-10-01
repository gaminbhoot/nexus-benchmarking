# NEXUS Benchmarking

Hardware-qualification system for the NEXUS autonomous air–ground robotic system.
Answers one question: **given this exact machine, can the exact NEXUS workload run
continuously within latency, accuracy, memory, thermal, and power margins?** —
and refuses to pass when critical evidence is missing.

## Quick start

```bash
pip install -e ".[dev]"
nexus-bench --profile smoke                       # validate everything in seconds
nexus-bench --profile profiles/purchase_gate.yaml --model nexus.pt \
  --video feed.mp4 --gate-strict                  # the purchase gate (exit 3 unless PASS)
nexus-bench matrix --profile profiles/rtx3050_4gb.yaml --model nexus.pt
```

`--profile` takes a builtin name or a YAML file (`profile_base` + overrides + `gate:`).
Every module runs in a **fresh worker process** under a supervising controller
(timeout kill → ABORTED, controller-side telemetry). Reports in `reports/`:
JSON, CSV, HTML (gate checks + provenance + charts).

## Gate, not vibe

Only `FULL`-status evidence counts toward the gate. `PARTIAL` (fallback detector,
synthetic pixels), `UNSUPPORTED`, `FAILED`, `INCONCLUSIVE`, `ABORTED`, `NOT_RUN`
fail their checks with reasons. Missing accuracy evidence → INCONCLUSIVE, never PASS.
See `profiles/purchase_gate.yaml` for the acceptance criteria (per-workload FPS /
p95 / p99 / miss / drop thresholds, VRAM headroom, throttle and OOM vetoes).

## Methodology: what each number actually is

| Module | Status |
|---|---|
| GEMM TFLOPS, bandwidth, VRAM ladder, allocator stats | Synthetic microbenchmark |
| YOLO grid (imgsz × batch × verified precision) | Actual component — needs `--model`; effective precision verified post-warmup, fail-closed |
| Backend matrix (torch fp32/fp16, ONNX CUDA, TensorRT fp16/int8) | Actual component where the stack exists, else `NOT_RUN` |
| Accuracy (fp32-vs-candidate agreement + mAP) | Preservation gate — needs `--model` (`--val-data` for mAP) |
| Re-ID embedding / DeepSORT-style tracking (greedy) | Actual components at reference scale |
| Video pipeline on paced streams | Actual workload on `--video` (FULL) else PARTIAL |
| UAV + rover concurrent + fusion | **Primary benchmark** — realtime (paced, sustained) and throughput modes |
| Two-view VO vs ground truth | Component benchmark with geometric verification |
| MAVLink v2 framing + UDP loopback | Real stack measurement (localhost = stack cost) |
| Cold start, decode path, thermal windows | Qualification measurements |

Key guarantees (all test-pinned): no weights → `UNSUPPORTED` with zero numbers;
requested fp16 not in effect → numbers withheld; requested model unloadable in
thermal → `FAILED` (never a silent GEMM swap); `sim_link_model` is simulation only.

## Modules

`system cpu gpu memory inference backends accuracy reid tracking vision pipeline
matrix mapping3d comms integrated thermal coldstart decode`

## Known limitations (roadmap, not silence)

- TensorRT/INT8 paths need the NVIDIA host (gated `NOT_RUN` here); Hungarian
  assignment needs scipy (greedy labelled); `rclpy`/physical links probed, not run.
- Reference Re-ID CNN is small by design (`--reid-model` for production weights).
- Depth-model, EKF fusion, and planner benchmarks are not yet implemented —
  the rover agent uses an occupancy-grid proxy (labelled PARTIAL-relevant).
- Laptop power-mode/firmware caps are read, never changed.

## Safety

Independent CPU/GPU temp limits, RAM/VRAM/duration caps, supervisor timeout kills,
CUDA OOMs caught with cleanup, Ctrl-C cancels the rest as `NOT_RUN`. Missing
sensors are "unavailable", never fatal or zero.

## Layout

```
nexus_bench/   cli (supervisor) worker stream gate statuses provenance
               profiler monitor stats safety profiles report yolo_util + *_bench.py
tests/         57 tests: smoke + metrics + gate + units + inference + qual
profiles/      purchase_gate.yaml, rtx3050_4gb.yaml, rtx3050_uav.yaml, full_stack.yaml
reports/       generated artifacts (git-ignored)
examples/sample_report/  committed example output
```
