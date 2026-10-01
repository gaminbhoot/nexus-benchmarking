# Changelog

## 0.3.0 — qualification system

- Explicit precision adapter (`yolo_util`): version-aware `quantize="fp16"` /
  legacy `half`, no inference probing, post-warmup `effective_precision`
  verification, fail-closed on mismatch (requested/effective/backend/version recorded).
- Measurement states everywhere: FULL/PARTIAL/FALLBACK/UNSUPPORTED/FAILED/
  INCONCLUSIVE/ABORTED/NOT_RUN. Only FULL is gate-eligible.
- Process isolation: every module runs in a fresh worker process under a
  supervising controller (timeout kill -> ABORTED, controller-side telemetry,
  `--runs N` repetitions, `--blas-threads` pinning, `--gate-strict` exit code).
- Purchase-gate engine (`gate.py` + `profiles/purchase_gate.yaml`): per-workload
  criteria (uav/rover/dual), headroom margins, resource vetoes, accuracy veto,
  INCONCLUSIVE-never-PASS semantics, per-check reasons in the HTML report.
- Real-time streams (`stream.py`): paced producer thread at target_fps, bounded
  queues, independent acquisition (read+decode) latency, queue-wait in e2e.
- Pipeline rewritten on paced streams: input vs processed FPS, drops, misses,
  FULL only with verified YOLO + real pixels.
- Integrated rewritten: fixed the `pop("_threads")` fusion-killer, realtime
  (paced, sustained `duration_s`, bounded queues, drops/depth) + throughput modes.
- Thermal rewritten: fixed 5 s windows (burst/steady/min), fail-closed on
  requested-model failure (no silent GEMM swap), same precision path as inference.
- New modules: `backends` (torch fp32/fp16 verified, ONNX CUDA, TensorRT fp16/int8
  gated NOT_RUN without stack), `accuracy` (fp32-vs-candidate agreement + mAP via
  `--val-data`), `coldstart` (load/first/steady in fresh process), `decode`
  (CPU/HW probe, FULL on `--video`), `matrix` (res x batch x FPS sweep, per-cell
  fresh processes, OOM pruning).
- VRAM: driver `mem_get_info`, allocator `memory_stats`, free-before/peak/
  free-after model protocol with working-set-above-baseline (no more
  total-minus-current).
- Safety: CPU and GPU temps evaluated independently (was: GPU masked by CPU).
- CPU: BLAS identity recorded (threadpoolctl/numpy config); heavier sustained test
  with degradation %.
- Provenance manifest in every report (git SHA/dirty, versions, model/config
  SHA256, power state). 57 tests green.

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
