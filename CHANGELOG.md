# Changelog

## 1.0.1 — Windows launcher that actually parses

- Root cause of instant-close: LF-only batch files + fragile parenthesized
  blocks. Launchers are now CRLF + pure ASCII in flat label/goto style
  (locked by test); shell launchers stay LF.
- Setup announces the ~2 GB download; broken venvs rebuild; exit codes propagate.

## 1.0.0 — first complete seller product

- Midway stop: Ctrl+C kills the worker (nothing orphaned) and packages partial
  evidence as ABORTED; a STOP file next to the launcher does the same between
  modules and inside long runs (flagged per-agent as stopped_early).
- Results are unmissable: absolute report/ZIP/folder paths printed, ZIP
  revealed in Explorer/Finder/file browser, HTML auto-opened.
- Diagnose-first launchers: `nexus-bench doctor` + staged WHAT/WHAT-TO-DO
  failures, venv health (not existence), exit-code propagation.

## 0.8.1 — seller launcher that diagnoses instead of flashing

- `nexus-bench doctor`: pre-install environment check (version/arch/venv/pip/
  disk/internet/assets) with OK/FAIL + exact fix per line; launchers run it
  first and stop with a readable message instead of a flash-and-close.
- Launchers verify venv HEALTH (imports + version), not mere existence;
  broken installs are discarded and rebuilt automatically; every pip step is
  error-checked (including the editable install); import verification after
  install; qualification exit code (0/2/3/4) propagated to the caller.
- Setup announces the ~2 GB one-time download; failures print WHAT happened /
  WHAT it means / WHAT to do plus the last log lines and log path.
- All file I/O declares UTF-8 (Windows cp1252 crash class eliminated, locked
  by static test).

## 0.8.0 — YOLO26m production workload (Qualification v2.0)

- Official workload is now **YOLO26m** (21.9M params measured, 44 MB,
  pinned SHA-256 from the official v8.4.0 release assets): the RTX 3050
  purchase gate measures the actual deployment-class detector, not YOLO11n.
- Qualification bumped v1.0 → **v2.0** (model change = new procedure);
  Reference vs Production layers documented (reference = fixed 26m weights;
  production = your exact checkpoint via engineering CLI).
- Validated on YOLO26m: FP16 effective-precision, FP32-vs-FP16 mAP comparison
  (drop 0.0 on COCO8), dual-context sustained FULL.
- Old reports/samples from the YOLO11n era must not be used for purchase.

## 0.7.0 — reproducible immutable product

- Pinned runtime (`requirements.lock`, verified exact versions): seller
  launchers install pins, never `>=` ranges; release records + verifies the
  requirements hash; provenance reports the four versions separately (software /
  qualification procedure / gate / asset set).
- Official model bundled in the repo + release ZIP (Mode A complete);
  bootstrap is recovery-only. Release builder bootstraps before verifying.
- True per-window drops from generation/delivery timestamps (uniform-spread
  assumption deleted); sustained thermal classifier (temp + clock decline +
  degradation) independently authoritative; headroom computed from sustained
  evidence; schema-aware worst-run aggregation incl. nested sustained metrics.
- Missing-evidence audit across every gate check (`_check_num`); fusion
  contract; duration/conservation proof; AC policy; hardware mismatch;
  deployment identity with engine/provider/runtime.
- Report rows carry PASS/FAIL/INCONCLUSIVE/NOT TESTED; limits from gate
  criteria; deployment identity, asset verification, LIMITATIONS, version
  stamps, duration proof, sustained charts, seller summary, packaged ZIP.
- `qualify` exit codes authoritative (0/2/3/4); preflight gate; pre/post
  sanity + recovery; release identity without Git (`NEXUS_DEV=1` for devs).
- PyInstaller spec + CI per-OS one-binary job (experimental); CI enforces
  pytest × 3 OS + release integrity. 99 tests green.

## 0.6.0 — immutable seller appliance

- Release system: `tools/build_release.py` assembles versioned self-contained
  `dist/NEXUS-Qualification-<platform>-vX.zip` (+ published SHA-256); runtime
  verifies source/gate/asset-manifest hashes before any qualification runs
  (tampered package → QUALIFICATION INVALID); profile modification detected via
  gate hash. `NEXUS_DEV=1` keeps developers unblocked.
- True per-window drops from generation/delivery timestamps (uniform-spread
  assumption removed; regression test proves a collapsing window can't hide).
- Gate missing-evidence audit: new `_check_num` helper — absent metrics yield
  INCONCLUSIVE, never fabricated PASS/FAIL; fusion contract (ticks/s, tick p95),
  actual-duration proof, conservation, leak-trajectory, power/battery, hardware
  identity, deployment identity (now with engine/provider/runtime versions).
- Report rows carry PASS/FAIL/INCONCLUSIVE/NOT TESTED individually; summary
  limits sourced from gate criteria; deployment identity, asset verification,
  LIMITATIONS, version stamps, duration proof sections.
- Accuracy tree bundled + hash-verified (extract-from-verified-ZIP fallback).
- Pipeline uses measured wall time; sustained resources align to SUSTAINED_START.
- `qualify` exit codes authoritative (0/2/3/4); preflight module; power section
  with AC prompt; pre/post sanity + recovery evidence.
- Docs split: seller `README_FIRST.txt` + short README vs `docs/DEVELOPER.md`;
  wizard explicitly developer-only. CI (pytest × 3 OS + release integrity).
  96 tests green.

## 0.5.0 — plug-and-play qualification product

- Seller appliance (`nexus-bench qualify`, launchers `NEXUS_Qualification.*`):
  hardware display → official assets → power/charger prompt → fixed official
  config → pre-flight (aborts before 10 min) → baseline sanity → run → post
  sanity/recovery → package ZIP → auto-open. Interruptions yield ABORTED, never PASS.
- Official assets (`assets/` + manifest + `assets.py`): pinned YOLO11n model,
  seeded UAV/rover sequences (byte-deterministic generator), COCO8 accuracy set —
  every file SHA-256 verified; bootstrap from hardcoded trusted URLs on failure;
  random user files never used. Replays loop short clips with replay counts.
- New `preflight` module (model/backend/precision/assets/decode/RAM/disk checks).
- Gate: worst-window sustained criteria, actual-vs-requested duration proof,
  deployment-identity match across modules, conservation requirement, AC/battery
  policy, sustained thermal telemetry, hardware mismatch FAIL, dirty-tree
  INCONCLUSIVE (releases exempt via release identity, no Git needed).
- Report: hero verdict + Why + criteria-sourced summary table, sustained charts,
  deployment identity, asset verification, LIMITATIONS, power honesty,
  qualification version stamps, plain-text seller summary, packaged folder + ZIP.
- Provenance: release.json identity (no .git required), full SHA-256 of all
  inputs. `tools/make_release.py` stamps immutable releases.
- Fixed: worker timeout < sustained duration; queue-drop/delivered accounting;
  runs aggregation preserving window series; report limits from gate criteria;
  launcher pipefail masking; missing module statuses; thermal window order +
  deployment imgsz; `full_stack [all]`; vision adapter. 90+ tests green.

## 0.4.0 — sustained qualification + seller operation

- New `sustained` module (600/900 s): dual-agent representative workload
  (UAV 640 + rover 480 + fusion) on paced independent streams with bounded
  queues, 60 s evolution windows (throughput/p95/VRAM/temp/util/power/queue/
  drops), memory-growth leak flag, WORKLOAD_OOM accounting, explicit depth
  limitation. Gate consumes sustained evidence (steady FPS, drops, degradation,
  duration, leak) and prefers sustained VRAM peaks.
- Conservation accounting (`stream.py`): generated/acquired/enqueued/dequeued/
  processed/delivered + producer/queue drops + failures + unfinished with a
  checked identity; one `drop_pct` definition used by pipeline/integrated/
  sustained. Full per-frame timestamp taxonomy (acquire/decode/enqueue/dequeue/
  stages/output).
- Concurrency contract (`concurrency.py`): separate-contexts (locked forwards),
  serialized, and CUDA-only separate-lockfree (refuses elsewhere). Fixed a real
  segfault: concurrent same-process Ultralytics forwards crash this stack
  (faulthandler-traced); integrated uses the same contract.
- UAV/rover sources truly separate (`--uav-video`/`--rover-video` with recorded
  fallback chains; same-file use is visible, never silent).
- VRAM kinds: CAPACITY_PROBE_OOM (exploratory, no veto) vs WORKLOAD_OOM (veto);
  gate uses sustained peak first.
- Matrix: real single-call batches with per-cell actual batch recorded; every
  target FPS maps to its own deadline (15→66.7 ms, 30→33.3 ms).
- Backends: ONNX CUDA verified via session providers (CPU fallback rejected by a
  unit-tested helper). Accuracy: full candidate-vs-baseline mAP50/mAP50-95/
  precision/recall/per-class-AP on shared `--val-data` + real-frame agreement
  (video → dir → synthetic content; black frames never used); gate reads the
  worst candidate's absolute drop.
- Statuses: worker turns unset/unknown status into FAILED (was: default FULL).
  Thermal tokens normalized; unknown/slowdown_cause_unknown → INCONCLUSIVE.
  `full_stack` `[all]` expansion fixed + regression test.
- Gate: FAIL (measured violation) vs INCONCLUSIVE (missing evidence) via check
  kinds; hardware identity match (TARGET MISMATCH banner); dirty-tree policy
  (`UNRELEASED / DIRTY SOURCE`, strict INCONCLUSIVE); `--runs` worst-case
  aggregation.
- Provenance: full SHA-256 of model, videos, datasets, profile+gate config.
  Power section with measured/unavailable/n-a honesty.
- Operator experience: `nexus-bench wizard` (discovery, plain-language progress,
  no tracebacks, WHAT/MEANS/COLLECTED failures), `run_nexus_bench.{sh,bat}`
  launchers, packaged `NEXUS_Qualification_<stamp>/` folder + ZIP + auto-open,
  seller hero report (verdict, Why, summary table, sustained charts, power,
  NOT TESTED), plain-text `qualification_summary.txt`. 85 tests green.

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
