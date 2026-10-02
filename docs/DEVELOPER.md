# NEXUS Benchmarking — Developer Documentation

> Seller? Read `README_FIRST.txt` instead — this document is engineering detail.

Hardware-qualification system for the NEXUS autonomous air–ground robotic system.
It answers one question: **given this exact machine, can the exact NEXUS workload
run continuously within latency, accuracy, memory, thermal, and power margins?**
— and it refuses to pass when the evidence is missing.

## Seller path (no technical knowledge needed)

Read **`README_FIRST.txt`**: double-click `NEXUS_Qualification` (`.bat` Windows,
`.command` macOS, `.sh` Linux), connect the charger if asked, wait ~10 minutes,
send the generated ZIP to the buyer. The seller provides **only the computer** —
model, videos, validation data, profile, and gate all come from the official
package (`assets/` + manifest, SHA-256 verified, bootstrap from pinned URLs if
a file is missing). `nexus-bench qualify [--extended]` is the same appliance
flow for terminals.

## Developer path

`nexus-bench ...` CLI with profiles/custom models/videos/matrices (see below).
`run_nexus_bench.{sh,bat}` are the developer launchers. `tools/` builds official
assets (`make_official_assets.py`, deterministic) and release identity
(`make_release.py` → `release/release.json`; never commit it).

## How a Seller Runs the Qualification (no Python knowledge needed)

1. **Get the folder** on the laptop (download + extract the ZIP, or clone the repo).
2. **Put the supplied files in that folder**: the model file (`*.pt`, e.g. `nexus.pt`)
   and, if you have them, two videos (`uav.mp4`, `rover.mp4`).
3. **Double-click the launcher**: `run_nexus_bench.bat` on Windows,
   `run_nexus_bench.sh` (or `bash run_nexus_bench.sh`) on Linux/Mac.
   The launcher checks Python, sets up an isolated environment, installs what is
   needed, and starts the wizard. If something is missing it says so in plain
   language — it never installs untrusted software silently.
4. **Answer the wizard's questions** (all have safe defaults — Enter accepts them):
   what to test, which model/video files to use (it finds them for you),
   10 or 15 minutes for the sustained run, then confirm.
5. **Wait.** Plain-language progress is shown (`[7/14] Testing UAV + rover
   together...`, plus a minute counter during the sustained run). Python errors
   are never dumped on screen; failures are explained as WHAT / WHAT IT MEANS /
   WHAT WAS STILL COLLECTED.
6. **The report opens in the browser** when finished, and everything is saved in
   a folder like `NEXUS_Qualification_2026-10-02_143522/` containing the HTML
   report, `results.json`, `results.csv`, `errors.log`, `provenance.json`, and
   `qualification_summary.txt` — plus a ZIP of the whole folder.
7. **Send the ZIP to the buyer.** That is the complete evidence package.

Technical shortcut (same engine, no wizard): `nexus-bench wizard`, or
`nexus-bench --profile profiles/purchase_gate.yaml --model nexus.pt
--uav-video uav.mp4 --rover-video rover.mp4 --gate-strict`.

## What the Final Report Means

The top of the HTML report shows one large result:

- **PASS / PASS_WITH_HEADROOM** — every required workload held its target for the
  full sustained run on FULL-status (real-workload) evidence: throughput, p95/p99
  latency, drops, VRAM headroom, no workload OOM, no throttling evidence, accuracy
  within tolerance. HEADROOM means comfortable margins, not borderline.
- **FAIL** — a measured number violated a limit (too slow, too many drops, VRAM
  over headroom, workload OOM, throttling evidence, accuracy drop, or the tested
  machine is not the expected hardware). The "Why?" list names each one.
- **INCONCLUSIVE** — the machine was *not proven bad*, but the evidence is
  insufficient to certify it (missing model/video, unavailable thermal sensors,
  no accuracy comparison, dirty source tree under a strict gate). Never read this
  as a pass.
- **NOT TESTED** — listed by module name: anything without FULL-status evidence
  (fallback detectors, synthetic pixels, missing weights/backends).

A nice-looking report never overrules missing evidence: synthetic, partial, or
unsupported measurements cannot produce PASS — the gate logic enforces this and
it is covered by a dedicated regression test (`test_garbage_result_never_passes`).

## What Is Actually Tested (10 or 15 minutes sustained)

`sustained_duration_s: 600` (or `900`) runs the representative workload with
**real-time paced inputs, independent per-agent streams, bounded queues**:

- **UAV agent** (640px): stream → preprocess → YOLO → Re-ID embedding →
  DeepSORT-style tracking → telemetry → fusion queue.
- **Rover agent** (480px): same, plus occupancy-grid obstacle cells.
  *Limitation, stated in every report: no depth-estimation network is implemented
  yet (occupancy proxy), so rover depth is NOT TESTED.*
- **Central fusion**: queue drain → world-state merge → allocation tick.

Concurrency contract (reported verbatim): `separate-contexts` = one YOLO object
per agent (real VRAM cost), forwards serialized by a shared lock — concurrent
same-process Ultralytics forwards segfault this stack (observed via faulthandler),
so lock-free execution is only offered as `separate-lockfree` on CUDA, and it
refuses elsewhere instead of risking a crash.

Windows (default 60 s) track evolution: initial/steady/min/final throughput,
degradation %, p95 init/final/max, VRAM init/peak/final/growth, temperatures,
utilization, power, queue depth, drops, deadline misses, OOMs (WORKLOAD_OOM vetoes;
exploratory CAPACITY_PROBE_OOM does not), memory-growth leak flag. A machine that
decays from 18 to 9 FPS is failed by its degradation %, never hidden by an average.

Frame accounting is conserved end-to-end
(`generated = delivered + producer_drops + queue_drops + processing_failures + unfinished`)
with one uniform definition: `drop_pct = 100·(generated−delivered)/generated`.

## For Engineers

- Every module runs in a fresh worker process (no CUDA-state contamination);
  supervisor kills on timeout → ABORTED. `--runs N` aggregates worst-case
  (min FPS, max latency/miss/drop) for the gate; individuals are kept.
- Precision is fail-closed: requested vs effective vs backend recorded; fp16 that
  is not really fp16 is withheld, never relabeled. ONNX CUDA is verified in the
  session providers (CPU fallback rejected); TensorRT/INT8 run where the stack
  exists, else NOT_RUN. INT8 needs its accuracy comparison like any candidate.
- Accuracy: candidate-vs-FP32 mAP50/mAP50-95/precision/recall/per-class AP on the
  same `--val-data`, plus prediction agreement on real frames (video → image-dir
  → synthetic content; black frames never used).
- Matrix (`resolution × batch × FPS`): batches are real single-call batches;
  each FPS column carries its own deadline (15→66.7 ms, 30→33.3 ms); OOM prunes.
- Provenance: full SHA-256 of model, videos, datasets, profile+gate config, git
  SHA + dirty flag (`UNRELEASED / DIRTY SOURCE` banner; strict gate goes
  INCONCLUSIVE on dirty). Expected hardware (`gpu_contains`, VRAM class) is
  verified — a wrong machine fails loudly instead of masquerading.
- Thermal: `none_detected` passes, `likely` fails, `unknown`/`slowdown_cause_unknown`
  → INCONCLUSIVE. Power signals are reported as measured/unavailable/n-a; the
  report never claims power was qualified on GPU-power alone.

Modules: `system cpu gpu memory inference backends accuracy reid tracking vision
pipeline matrix mapping3d comms integrated thermal coldstart decode sustained`

## Layout

```
nexus_bench/   supervisor+worker, stream (paced+accounted), gate, statuses,
               provenance, power, concurrency, yolo_util, backends/accuracy/
               sustained/matrix/pipeline/integrated + microbenchmarks
profiles/      purchase_gate.yaml (the gate), rtx3050_4gb.yaml, rtx3050_uav.yaml, full_stack.yaml
tests/         100+ tests incl. gate fail-closed, accounting, precision, OOM kinds
run_nexus_bench.{sh,bat}   one-click launchers
reports/       local runs (git-ignored)   examples/sample_report/  committed example
NEXUS_Qualification_<stamp>/  packaged output (report+JSON+CSV+log+provenance+summary+.zip)
```
