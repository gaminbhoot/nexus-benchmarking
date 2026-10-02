"""Environment doctor: prove WHAT is broken before anything is installed.

`nexus-bench doctor` runs in seconds, changes nothing, and prints one line per
check with OK / FAIL / WARN plus the exact fix. The Windows launcher runs this
first so a seller never stares at a flash-and-close window: every failure ends
with a readable message and a log file instead of a traceback.
Exit codes: 0 = runnable, 2 = fixable problem found (details printed).
"""
import json
import os
import platform
import shutil
import sys
import urllib.request

MIN_PY = (3, 10)
MAX_PY = (3, 14)


def _check(results, name, ok, detail, fix=""):
    results.append({"check": name, "ok": bool(ok), "detail": detail, "fix": fix})
    return bool(ok)


def run_doctor():
    checks = []
    v = sys.version_info
    _check(checks, "python_version", MIN_PY <= (v.major, v.minor) <= MAX_PY,
           f"Python {platform.python_version()}",
           "" if MIN_PY <= (v.major, v.minor) <= MAX_PY else
           f"Install Python 3.{MIN_PY[1]}–3.{MAX_PY[1]} 64-bit from python.org")
    _check(checks, "python_arch", platform.machine().lower() not in ("x86", "i386", "arm32")
           and (sys.maxsize > 2 ** 32),
           platform.machine(),
           "Install 64-bit Python (32-bit cannot load torch/CUDA)")
    _check(checks, "venv_module", _has_module("venv"),
           "venv available" if _has_module("venv") else "missing",
           "Reinstall Python with default options (includes venv/pip)")
    _check(checks, "pip", _has_module("pip"), "pip available" if _has_module("pip") else "missing",
           "Run: python -m ensurepip")
    try:
        free_gb = shutil.disk_usage(os.getcwd()).free / 1e9
        _check(checks, "disk", free_gb > 6.0, f"{free_gb:.1f} GB free (need ~6 GB once)",
               "Free at least 6 GB (pinned torch/CUDA wheels are large)")
    except Exception as e:
        _check(checks, "disk", False, str(e), "")
    net = _url_ok("https://pypi.org/simple/", 15)
    _check(checks, "internet_pypi", net,
           "PyPI reachable" if net else "unreachable",
           "" if net else "Connect this machine to the internet for one-time setup")
    try:
        from nexus_bench import assets as A
        root = A.package_root()
        lock = os.path.join(root, "requirements.lock")
        _check(checks, "release_files", os.path.exists(lock),
               "requirements.lock present" if os.path.exists(lock) else "missing",
               "Extract the WHOLE zip first (right-click > Extract All), then re-run")
        ok, recs = A.verify_assets(root)
        bad = [r["asset_id"] for r in recs if r["status"] != "OK"]
        _check(checks, "official_assets", ok,
               "all verified" if ok else f"unverified: {bad}",
               "Keep internet on: missing files download automatically at startup")
    except Exception as e:
        _check(checks, "official_assets", False, f"check failed: {e}", "")
    try:
        import torch  # noqa
        cuda = torch.cuda.is_available()
        _check(checks, "torch_cuda", True,
               f"torch {torch.__version__}, CUDA={'yes' if cuda else 'no (CPU-only machine)'}", "")
    except Exception:
        _check(checks, "torch_cuda", True, "torch not installed yet (installed at setup)", "")
    return checks


def _has_module(name):
    try:
        __import__(name)
        return True
    except Exception:
        return False


def _url_ok(url, timeout):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status < 400
    except Exception:
        return False


def main():
    print("NEXUS environment check")
    print("-----------------------")
    checks = run_doctor()
    bad = 0
    for c in checks:
        mark = "OK  " if c["ok"] else ("FAIL" if c["check"] in
                                        ("python_version", "python_arch", "venv_module", "pip",
                                         "disk", "internet_pypi", "release_files") else "WARN")
        if mark != "OK  ":
            bad += 1
        print(f"{c['check']:.<22} {mark}  {c['detail']}")
        if mark != "OK  " and c["fix"]:
            print(f"  -> {c['fix']}")
    fails = [c for c in checks if not c["ok"] and c["check"] in
             ("python_version", "python_arch", "venv_module", "pip", "disk",
              "internet_pypi", "release_files")]
    if fails:
        print("\nSetup cannot continue until the FAIL lines above are fixed.")
        return 2
    if bad:
        print("\nWarnings only — setup can continue.")
    else:
        print("\nEverything looks good.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
