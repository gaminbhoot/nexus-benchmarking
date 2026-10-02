# NEXUS Benchmarking — Reference Hardware Qualification v1.0

This repository builds the **NEXUS Reference Hardware Qualification**: a fixed,
versioned procedure that tests whether a machine can run the reference NEXUS
workload (YOLO11n + reference Re-ID + DeepSORT-style tracking on official
sequences, UAV + rover + fusion) continuously for 10–15 minutes.

> It does **not** certify every future production NEXUS model — only that this
> machine passed this exact reference procedure. The report states the exact
> model, data, code, and gate versions measured.

## Seller workflow (the only seller workflow)

Read **`README_FIRST.txt`**:

```text
Download → Extract → Run NEXUS_Qualification → charger if asked → wait → send ZIP
```

The seller provides **only the computer**. No model, video, dataset, profile,
gate, precision, or CLI choices exist in the seller path (`nexus-bench qualify`).

## What the report means

- **PASS / PASS_WITH_HEADROOM** — every required check held on FULL-status
  (real-workload) evidence for the full duration.
- **FAIL** — a measured number violated a limit; the Why list names it.
- **INCONCLUSIVE** — evidence insufficient (missing sensor/data/hardware, dirty
  dev tree under strict gate). Never read as pass.
- **NOT TESTED** — listed by module; includes depth network, physical link,
  mission planner (proxy only) — this is a *compute* qualification.

## Developer documentation

All engineering detail lives in **`docs/DEVELOPER.md`**: architecture, modules,
precision adapter, gate engine, statuses, profiles, accuracy/backends, release
process, and test layout. The interactive `nexus-bench wizard` is a developer
guided mode, not a qualification path.

## Layout

```text
nexus_bench/   qualify (seller appliance) + supervisor/workers + gate + assets
assets/        official pinned test assets (model/videos/COCO8, SHA-256)
profiles/      purchase_gate.yaml (official gate) + engineering profiles
tools/         make_official_assets.py, make_release.py, build_release.py
tests/         regression + seller integration suites
NEXUS_Qualification.{bat,sh,command}   seller launchers
run_nexus_bench.{sh,bat}               developer launchers
```
