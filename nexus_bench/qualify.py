"""Seller appliance: the official NEXUS qualification, fully automatic.

Flow: hardware display -> official assets (verify/bootstrap, never user files)
-> power check (charger prompt is the only physical interaction) -> official
fixed config -> pre-flight (abort before 10 min on failure) -> baseline sanity
-> qualification run -> post sanity + recovery -> package + open browser.

Engineering CLI stays separate (`nexus-bench ...` with profiles/models/videos).
`nexus-bench qualify [--extended] [--yes]` is the seller entry point.
"""
import json
import os
import time
import webbrowser

OFFICIAL_MODULES = ["preflight", "coldstart", "inference", "backends", "accuracy",
                    "pipeline", "tracking", "reid", "decode", "memory",
                    "integrated", "thermal", "comms", "sustained"]

# Authoritative process exit codes (report stays human-facing, code is machine-facing).
VERDICT_EXIT = {"PASS": 0, "PASS_WITH_HEADROOM": 0, "FAIL": 2,
                "INCONCLUSIVE": 3, "ABORTED": 4}

PLAIN_STEPS = {
    "preflight": "Pre-flight check", "coldstart": "Measuring startup time",
    "inference": "Testing AI detection speed", "backends": "Testing AI backends",
    "accuracy": "Checking AI accuracy", "pipeline": "Testing camera pipeline",
    "tracking": "Testing object tracking", "reid": "Testing re-identification",
    "decode": "Testing video decoding", "memory": "Checking memory headroom",
    "integrated": "Testing UAV + rover together", "thermal": "Checking heat behavior",
    "comms": "Testing communications", "sustained": "Sustained qualification",
}

def _say(msg, quiet=False):
    if not quiet:
        print(msg, flush=True)

def _prompt_yes(msg, default=True):
    try:
        raw = input(msg).strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return default
    if not raw:
        return default
    return raw in ("y", "yes")

def official_config(root, duration_s, out_dir):
    """The ONE official configuration. No seller choices inside."""
    from nexus_bench import assets as A
    from nexus_bench.profiles import get
    paths, records = A.ensure_all(root)
    cfg = get("full")
    acc_yaml = os.path.join(paths["assets_dir"], "accuracy", "coco8.yaml")
    if not os.path.exists(acc_yaml):
        # Real COCO class names from the installed ultralytics coco8, repointed
        # at the verified bundled tree (absolute path, deterministic content).
        import yaml as _yaml
        import ultralytics
        bundled = os.path.join(os.path.dirname(ultralytics.__file__),
                               "cfg", "datasets", "coco8.yaml")
        with open(bundled, encoding="utf-8") as f:
            data = _yaml.safe_load(f)
        data["path"] = paths["accuracy_dir"]
        with open(acc_yaml, "w", encoding="utf-8") as f:
            _yaml.safe_dump(data, f)
    cfg.update({"model": paths["model"], "uav_video": paths["uav_video"],
                "rover_video": paths["rover_video"], "video": paths["uav_video"],
                "val_data": acc_yaml, "precision": "fp16", "device": "auto",
                "imgsz": 640, "batch": 1, "target_fps": 15.0,
                "integrated_mode": "realtime", "concurrency": "separate-contexts",
                "duration_s": 60, "sustained_duration_s": duration_s,
                "sustained_window_s": 60, "repeats": 30, "warmup": 5, "runs": 1,
                "req_fps": 15, "seed": 0, "out": out_dir,
                "official_qualification": True})
    return cfg, paths, records

def official_gate(root):
    from nexus_bench import gate as G
    import yaml
    with open(os.path.join(root or ".", "profiles", "purchase_gate.yaml"), encoding="utf-8") as f:
        doc = yaml.safe_load(f) or {}
    gate_cfg = dict(G.DEFAULT_GATE)
    gate_cfg.update(doc.get("gate", {}))
    return gate_cfg

def sanity_snapshot():
    """Pre/post resource snapshot for the recovery check."""
    snap = {}
    try:
        import psutil
        snap["ram_pct"] = psutil.virtual_memory().percent
        snap["proc_rss_mb"] = round(psutil.Process().memory_info().rss / 1e6, 1)
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            snap["vram_alloc_mb"] = round(torch.cuda.memory_allocated() / 1e6, 1)
    except Exception:
        pass
    try:
        from nexus_bench import power as P
        snap["power"] = {k: v.get("value") for k, v in P.collect().get("signals", {}).items()}
    except Exception:
        pass
    return snap

def _fmt_gb(v):
    return f"{v} GB" if v != "?" else "?"

def run_qualify(root=None, extended=False, assume_yes=False, out_parent=".",
                progress_out=None, _test_overrides=None):
    """Returns (package_dir, zip_path, results). Never raises on benchmark
    failure — interruptions produce ABORTED evidence, never PASS.
    _test_overrides: test-only config shrink (never used by sellers)."""
    from nexus_bench import assets as A
    from nexus_bench import gate as G
    from nexus_bench import profiler
    from nexus_bench.cli import controller_run
    from nexus_bench.wizard import package_results
    root = root or A.package_root()
    duration_s = 900 if extended else 600
    say = (lambda m: None) if progress_out == "silent" else _say
    say("=" * 55)
    say("       NEXUS HARDWARE QUALIFICATION")
    say("=" * 55)
    say("This application automatically tests this computer against the")
    say("official NEXUS workload. No files need to be selected.")
    from nexus_bench import release as R
    dev_override = os.environ.get("NEXUS_DEV") == "1"
    rel_ok, rel_problems = (True, ["developer override NEXUS_DEV=1: release identity "
                                   "checked but not enforced"]) if dev_override else R.verify_runtime(root)
    rel = None if dev_override else R.read_release(root)
    if rel is not None:
        if not rel_ok:
            return _abort_package(out_parent, "release-integrity",
                                  "QUALIFICATION INVALID — this package differs from the "
                                  "official release it claims to be.\n" +
                                  "\n".join(f"- {p}" for p in rel_problems) +
                                  "\nNo qualification result was produced.", say)
        say(f"\nRelease: NEXUS Qualification v{rel.get('qualification_version', '?')} "
            f"({rel.get('release_id', '?')}) — identity verified.")
    else:
        say("\nDistribution: developer tree (no release stamp; release checks skipped).")
    prof = profiler.profile()
    gpu, mem, cpu = prof.get("gpu", {}), prof.get("memory", {}), prof.get("cpu", {})
    say("")
    say("Detected hardware:")
    say(f"  CPU: {cpu.get('brand', '?')}")
    say(f"  GPU: {gpu.get('name', gpu.get('note', '?'))}")
    say(f"  VRAM: {gpu.get('vram_total', '?')}")
    say(f"  RAM: {_fmt_gb(mem.get('ram_total_gb', '?'))}")
    # --- assets (bootstrap if needed; abort cleanly if impossible) ---
    say("\nOfficial test package:")
    try:
        paths, records = A.ensure_all(root, progress=lambda *a: None)
        tree = A.ensure_accuracy_tree(root)
        records = list(records) + [{"asset_id": "nexus-qual-coco8-v1-tree",
                                    "path": tree, "status": "OK",
                                    "detail": "extracted tree hash verified"}]
        for r in records:
            nm = r["asset_id"].rsplit("-", 1)[0].split("nexus-qual-")[-1]
            say(f"  {nm:.<22} VERIFIED")
        say("  accuracy tree........ VERIFIED")
    except A.AssetError as e:
        return _abort_package(out_parent, "assets",
                              "The official NEXUS test files could not be prepared.\n"
                                "The benchmark has NOT started. No qualification result was produced.\n"
                                f"Reason: {e}\nCheck the internet connection and run again.", say)
    # --- power (physical prompt allowed) ---
    from nexus_bench import power as P
    power = P.collect()
    ac = (power["signals"].get("ac_connected") or {}).get("value")
    say(f"  Power: {'AC' if ac is True else ('BATTERY' if ac is False else 'unknown')}")
    on_battery = False
    if ac is False and not assume_yes:
        say("\nThis qualification is designed to run on AC power.")
        say("Please connect the laptop charger.")
        try:
            input("Press ENTER after connecting power (or Ctrl-C to stop)... ")
        except (EOFError, KeyboardInterrupt):
            pass
        power = P.collect()
        ac = (power["signals"].get("ac_connected") or {}).get("value")
    if ac is False:
        on_battery = True
        say("TEST CONDITION: BATTERY POWER — recorded; strict gate will not certify it as AC.")
    # --- official config + preflight ---
    cfg, _, _ = official_config(root, duration_s, os.path.join(out_parent, "reports"))
    if _test_overrides:
        cfg.update(_test_overrides)
    if on_battery:
        cfg["power_condition"] = "BATTERY"
    gate_cfg = official_gate(root)
    from nexus_bench.cli import _run_worker
    import copy
    pre_cfg = copy.deepcopy(cfg)
    pre_cfg.update({"duration_s": 5})
    say("\nPre-flight check")
    pre, _ = _run_worker("nexus_bench.preflight_bench", pre_cfg, 300, None)
    for c in (pre.get("tests") or {}).get("checks", []):
        say(f"  {c['check']:.<22} {'OK' if c['pass'] else 'FAIL'}")
    if pre.get("status") != "FULL":
        return _abort_package(out_parent, "preflight",
                              "Pre-flight FAILED — the 10-minute test was not started.\n" +
                              "\n".join((pre.get("errors") or ["unknown"])[:4]), say,
                              extra={"preflight": pre})
    say("\nPre-flight: all critical checks OK. Starting qualification...")
    # --- baseline sanity, run, post sanity ---
    baseline = sanity_snapshot()
    total = len([m for m in OFFICIAL_MODULES if m != "preflight"])

    def progress(key, idx, n):
        base = PLAIN_STEPS.get(key.split("__")[0], key)
        step = idx + 1  # preflight was stage 1; controller steps follow in order
        stages = len(OFFICIAL_MODULES)
        if key == "sustained":
            el = _sustained_elapsed.get("s", 0)
            say(f"Stage {step}/{stages} {base}: {el // 60:02d}:{el % 60:02d} / "
                f"{duration_s // 60:02d}:{duration_s % 60:02d} (sustained clock)")
        else:
            say(f"Stage {step}/{stages} {base}...")
    _sustained_elapsed = {"s": 0}
    ticker_stop = {"v": False}

    def ticker():
        t0 = time.time()
        while not ticker_stop["v"] and time.time() - t0 < duration_s + 60:
            _sustained_elapsed["s"] = int(time.time() - t0)
            time.sleep(5)

    import threading
    threading.Thread(target=ticker, daemon=True).start()
    try:
        paths, results, _code = controller_run(
            cfg, [m for m in OFFICIAL_MODULES if m != "preflight"],
            gate_cfg, req_fps=15, runs=1, timeout=None,
            cblas=None, out=cfg["out"], gate_strict=False, progress=progress)
    except KeyboardInterrupt:
        ticker_stop["v"] = True
        return _abort_package(out_parent, "interrupted",
                              "Qualification interrupted.\nRESULT: ABORTED\n"
                              "No qualification PASS was issued.", say)
    ticker_stop["v"] = True
    post = sanity_snapshot()
    results["preflight"] = pre  # prerequisite evidence travels with the package
    results["_sanity"] = {"baseline": baseline, "final": post,
                          "recovered": _recovery_ok(baseline, post)}
    results["_assets"] = {"manifest": records,
                          "qualification_version": __import__(
                              "nexus_bench.release", fromlist=["x"]).QUALIFICATION_VERSION}
    results["_power_condition"] = "BATTERY" if on_battery else ("AC" if ac is True else "unknown")
    folder, zipp = package_results(paths, results, parent=out_parent)
    say("\n" + "=" * 55)
    say("QUALIFICATION COMPLETE")
    say("=" * 55)
    say(f"\nRESULT: {results.get('_gate', {}).get('verdict', '?')}")
    say(f"\nReport:\n{os.path.join(folder, 'NEXUS_Qualification_Report.html')}")
    say(f"\nEvidence package:\n{zipp}")
    try:
        webbrowser.open("file://" + os.path.abspath(
            os.path.join(folder, "NEXUS_Qualification_Report.html")))
        say("\nThe report has been opened in your browser.")
    except Exception:
        say("\n(Open the HTML report above in a browser.)")
    return folder, zipp, results

def _recovery_ok(before, after):
    try:
        dr = (after.get("proc_rss_mb") or 0) - (before.get("proc_rss_mb") or 0)
        dv = (after.get("vram_alloc_mb") or 0) - (before.get("vram_alloc_mb") or 0)
        return {"rss_delta_mb": round(dr, 1), "vram_delta_mb": round(dv, 1),
                "recovered": dr < 500 and dv < 100,
                "note": "post-run vs pre-run in the controller process"}
    except Exception:
        return {"recovered": "unknown"}

def _abort_package(out_parent, stage, message, say, extra=None):
    import datetime
    say("\n" + message)
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    folder = os.path.join(out_parent, f"NEXUS_Qualification_{stamp}_ABORTED")
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "errors.log"), "w", encoding="utf-8") as f:
        f.write(f"stage: {stage}\n{message}\nRESULT: ABORTED\nNo qualification PASS was issued.\n")
    if extra:
        with open(os.path.join(folder, "results.json"), "w", encoding="utf-8") as f:
            json.dump(extra, f, indent=2, default=str)
    import zipfile
    zipp = folder + ".zip"
    with zipfile.ZipFile(zipp, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(folder):
            for fn in files:
                p = os.path.join(root, fn)
                z.write(p, os.path.relpath(p, folder))
    say(f"\nEvidence of the attempt was saved to:\n{folder}\n{zipp}")
    return folder, zipp, {"_gate": {"verdict": "ABORTED"}, "errors": [message]}
