"""Developer guided mode: interactive helper for ENGINEERING runs.

NOT the seller path. The seller appliance is `nexus-bench qualify` (fixed
official assets/config/gate, no file choices). This wizard exists so developers
can explore custom models/videos/profiles without memorizing CLI flags — every
choice here is an engineering choice and results are NOT official qualifications.
"""
import glob
import glob
import os
import shutil
import threading
import time
import webbrowser
import zipfile

GATE_MODULES = ["system", "coldstart", "inference", "backends", "accuracy",
                "pipeline", "tracking", "reid", "decode", "memory",
                "integrated", "thermal", "comms", "sustained"]
QUICK_MODULES = ["system", "coldstart", "inference", "memory", "pipeline"]

def _find(patterns):
    found = []
    for p in patterns:
        found.extend(glob.glob(p) + glob.glob(p.upper()))
    return sorted(set(f for f in found if os.path.isfile(f)))

def discover():
    return {"models": _find(["*.pt", "*.onnx", "*.engine"]),
            "videos": _find(["*.mp4", "*.avi", "*.mov", "*.mkv"])}

def _ask_menu(title, options, default=1):
    print(f"\n{title}")
    for i, o in enumerate(options, 1):
        print(f"  [{i}] {o}")
    try:
        raw = input(f"Choose [default {default}]: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return default
    return int(raw) if raw.isdigit() and 1 <= int(raw) <= len(options) else default

def _ask_yn(prompt, default=True):
    try:
        raw = input(f"{prompt} [{'Y/n' if default else 'y/N'}]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return default
    if not raw:
        return default
    return raw in ("y", "yes")

def _ask_video(role, videos, other_choice=None):
    opts = [f"{v}" for v in videos]
    if other_choice:
        opts.append(f"Use same file as UAV ({other_choice})")
    opts.append("Skip (synthetic pixels — report will say PARTIAL, not qualified)")
    i = _ask_menu(f"{role} video:", opts, default=1 if videos else len(opts))
    pick = opts[i - 1]
    if pick.startswith("Use same file"):
        return other_choice
    if pick.startswith("Skip"):
        return ""
    return pick

def reveal_in_file_manager(path):
    """Show the finished report/ZIP in the OS file manager (Explorer/Finder/
    file browser). Never raises: headless machines simply skip it. Returns True
    if a reveal command was launched."""
    import platform
    import subprocess
    ap = os.path.abspath(path)
    if not os.path.exists(ap):
        return False
    try:
        system = platform.system()
        if system == "Windows":
            subprocess.Popen(["explorer", "/select,", ap])
        elif system == "Darwin":
            subprocess.Popen(["open", "-R", ap])
        else:
            target = ap if os.path.isdir(ap) else os.path.dirname(ap)
            subprocess.Popen(["xdg-open", target])
        return True
    except Exception:
        return False

def run_wizard():
    from nexus_bench import profiler
    from nexus_bench.profiles import get
    print("=" * 60)
    print("NEXUS Developer Guided Mode (NOT an official qualification)")
    print("Custom models/videos/profiles are engineering choices.")
    print("For the official seller appliance, run: nexus-bench qualify")
    print("=" * 60)
    prof = profiler.profile()
    gpu = (prof.get("gpu") or {})
    print(f"Detected GPU:  {gpu.get('name', gpu.get('note', 'unknown'))}")
    print(f"Detected VRAM: {gpu.get('vram_total', 'unknown')}")
    print(f"Detected RAM:  {(prof.get('memory') or {}).get('ram_total_gb', '?')} GB")
    print(f"Detected CPU:  {(prof.get('cpu') or {}).get('brand', 'unknown')}")
    found = discover()
    print(f"\nFound models: {found['models'] or 'none — put the supplied .pt file in this folder'}")
    print(f"Found videos: {found['videos'] or 'none'}")

    purpose = _ask_menu("What are you testing?",
                        ["RTX 3050 4 GB purchase qualification (recommended)",
                         "Quick check (a few minutes, not a qualification)"], 1)
    mods = GATE_MODULES if purpose == 1 else QUICK_MODULES

    model = ""
    if found["models"]:
        mi = _ask_menu("Model:", found["models"] + ["Skip (modules needing weights report NOT TESTED)"],
                       1)
        pick = (found["models"] + [""])[mi - 1]
        model = "" if pick.startswith("Skip") else pick
    uav_video = _ask_video("UAV", found["videos"])
    rover_video = _ask_video("Rover", found["videos"], other_choice=uav_video or None)
    minutes = 10 if _ask_menu("Sustained test:", ["10 minutes", "15 minutes"], 1) == 1 else 15

    precision = "fp16" if _ask_yn("Use FP16 if supported? (recommended on NVIDIA) [Y/n]", True) else "fp32"
    print(f"\nPlan: {len(mods)} steps, sustained {minutes} min, model={model or 'none'}, "
          f"UAV={uav_video or 'synthetic'}, rover={rover_video or 'synthetic'}.")
    if not _ask_yn("Run qualification? [Y/n]", True):
        print("Cancelled. Nothing was run.")
        return 0

    from nexus_bench import gate as gate_mod
    cfg = get("full" if purpose == 1 else "smoke")
    cfg.update({"model": model, "uav_video": uav_video, "rover_video": rover_video,
                "video": uav_video, "precision": precision,
                "sustained_duration_s": minutes * 60, "runs": 1,
                "out": "NEXUS_Qualification_reports"})
    try:
        with open("profiles/purchase_gate.yaml", encoding="utf-8") as f:
            import yaml
            doc = yaml.safe_load(f) or {}
            gate_cfg = dict(gate_mod.DEFAULT_GATE)
            gate_cfg.update(doc.get("gate", {}))
    except Exception:
        gate_cfg = dict(gate_mod.DEFAULT_GATE)

    from nexus_bench.cli import controller_run
    stop_ticker = threading.Event()

    def ticker(total_s):
        t0 = time.time()
        while not stop_ticker.is_set() and time.time() - t0 < total_s:
            el = int(time.time() - t0)
            print(f"  ... sustained qualification running: {el // 60:02d}:{el % 60:02d} / "
                  f"{total_s // 60:02d}:00", end="\r", flush=True)
            time.sleep(15)
        print(flush=True)

    def progress(key, idx, total):
        plain = {"system": "Checking hardware", "coldstart": "Measuring startup time",
                 "inference": "Testing AI detection speed", "backends": "Testing AI backends",
                 "accuracy": "Checking AI accuracy", "pipeline": "Testing camera pipeline",
                 "tracking": "Testing object tracking", "reid": "Testing re-identification",
                 "decode": "Testing video decoding", "memory": "Checking memory",
                 "integrated": "Testing UAV + rover together",
                 "thermal": "Checking heat behavior", "comms": "Testing communications",
                 "sustained": f"Running sustained qualification ({minutes} minutes)"}
        print(f"[{idx}/{total}] {plain.get(key.split('__')[0], key)}...")
        if key == "sustained" and minutes * 60 > 120:
            threading.Thread(target=ticker, args=(minutes * 60,), daemon=True).start()

    try:
        paths, results, _code = controller_run(cfg, mods, gate_cfg, req_fps=15, runs=1,
                                               timeout=None, cblas=None,
                                               out=cfg["out"], gate_strict=False,
                                               progress=progress)
    finally:
        stop_ticker.set()
    print(f"\n[{len(mods)}/{len(mods)}] Qualification complete.")
    _explain_failures(results)
    folder, zipp = package_results(paths, results)
    print(f"\nREPORT READY\nFolder:      {folder}\nHTML report: {paths['html']}\nZIP package: {zipp}")
    try:
        webbrowser.open("file://" + os.path.abspath(paths["html"]))
        print("(Report opened in your browser.)")
    except Exception:
        print("(Could not open a browser automatically — open the HTML file above.)")
    return 0

def _explain_failures(results):
    bad = [(k, v) for k, v in results.items()
           if not k.startswith("_") and isinstance(v, dict)
           and v.get("status") in ("FAILED", "ABORTED")]
    if not bad:
        return
    print("\nSome steps did not finish normally:")
    for k, v in bad:
        errs = (v.get("errors") or ["unknown error"])[:2]
        print(f"\nWHAT HAPPENED: {k}: {errs[0][:200]}")
        print("WHAT IT MEANS: this part could not be measured; the report marks it "
              f"{v.get('status')} and it does not count toward qualification.")
    print("WHAT WAS STILL COLLECTED: all completed steps are in the report and ZIP.")

def package_results(paths, results, parent="."):
    import datetime
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    folder = os.path.join(parent, f"NEXUS_Qualification_{stamp}")
    os.makedirs(folder, exist_ok=True)
    from nexus_bench import report as report_mod
    mapping = {"html": "NEXUS_Qualification_Report.html", "json": "results.json",
               "csv": "results.csv", "log": "errors.log"}
    for k, name in mapping.items():
        if paths.get(k) and os.path.exists(paths[k]):
            shutil.copy(paths[k], os.path.join(folder, name))
    with open(os.path.join(folder, "provenance.json"), "w", encoding="utf-8") as f:
        import json
        json.dump(results.get("_provenance", {}), f, indent=2, default=str)
    for key, name in (("_assets", "asset_manifest.json"),):
        if results.get(key) is not None:
            with open(os.path.join(folder, name), "w", encoding="utf-8") as f:
                json.dump(results[key], f, indent=2, default=str)
    qman = {"generated_utc": stamp,
            "qualification_version": results.get("_assets", {}).get("qualification_version", "?"),
            "modules": sorted(k for k in results if not k.startswith("_") and isinstance(results[k], dict)),
            "verdict": (results.get("_gate") or {}).get("verdict", "?")}
    with open(os.path.join(folder, "qualification_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(qman, f, indent=2, default=str)
    with open(os.path.join(folder, "qualification_summary.txt"), "w", encoding="utf-8") as f:
        f.write(report_mod.plain_summary(results))
    zipp = folder + ".zip"
    with zipfile.ZipFile(zipp, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(folder):
            for fn in files:
                p = os.path.join(root, fn)
                z.write(p, os.path.relpath(p, folder))
    return folder, zipp
