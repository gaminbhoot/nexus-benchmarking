# Changelog

## 0.2.0 — honest measurements

- `inference`: `--precision fp16` now casts weights AND passes the precision flag
  (`quantize` on new ultralytics, `half` on old, probed); batch tests run all N
  images in ONE `predict` call with per-image + batch throughput reported.
- No weights → module reports `unsupported` with zero latency numbers (was: conv proxy).
- New `tracking` module: DeepSORT-style Kalman + appearance cascade (greedy assignment),
  seeded GT scenario with ID-switch rate. New `reid` module: real torch embedding
  CNN on crops (`--reid-model` for custom weights).
- New `pipeline` module: decode→detect→embed→associate→telemetry on
  `--video`/`--image-dir`/labelled synthetic, with FPS, e2e p95/p99, deadline-miss %, drops.
- `integrated`: explicit UAV + rover + central fusion agents, warmed-up baseline,
  per-agent own-time FPS, contention ratio, fusion ticks, queue drops.
- `thermal`: device-sized load, temp+clock+power time series, throttling claimed
  only on evidence conjunction.
- `feasibility`: p95 + deadline-miss + drops + measured FPS vs `--req-fps`, with
  per-verdict reasons; OOM outranks empty results.
- `gpu`: explicit TFLOPS math + directional H2D/D2D/D2H bandwidth. `memory`: VRAM
  ladder to ~3.5 GB with touch/compute per step, YOLO footprint + remainder.
- `comms`: hand-rolled MAVLink v2 CRC framing (serialize/parse/reject) + real UDP
  loopback RTT; RNG latency kept as labelled `sim_link_model` only.
- `mapping3d`: two-view ORB→E→pose→triangulate vs ground truth (rotation/translation/
  reprojection errors); voxel work labelled microbenchmark.
- CLI: YAML profiles (`--profile path.yaml`), `--video/--uav-video/--rover-video/
  --image-dir/--reid-model/--imgsz-list/--batch-list` wired through.
- Monitor: GPU clocks, power limit, throttle flags, CPU freq/temp, AC state.
- Profiler: CPU brand, power limit/current, cuDNN/TensorRT/OpenCV versions.
- Tests: 37 (smoke + metrics/feasibility + units + inference honesty).

## 0.1.0 — modular skeleton

Initial framework: profiler, monitor, per-module scripts with synthetic proxies,
median-based feasibility, smoke tests.
