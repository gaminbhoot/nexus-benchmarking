#!/usr/bin/env python3
"""Assemble the immutable seller distribution.

Builds dist/NEXUS-Qualification-<platform>-v<qual>.zip containing the runtime
(source), official assets, profiles, gate, launchers, README_FIRST, and a
fresh release.json. Prints the artifact SHA-256 for publication. The seller
never builds anything; this runs on the maintainer/CI side.
"""
import hashlib
import os
import shutil
import subprocess
import sys
import time
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

INCLUDE_TOP = ["nexus_bench", "profiles", "assets", "release", "tools",
               "README_FIRST.txt", "README.md", "CHANGELOG.md", "LICENSE",
               "pyproject.toml", "requirements.lock"]
INCLUDE_LAUNCHERS = ["NEXUS_Qualification.bat", "NEXUS_Qualification.sh",
                     "NEXUS_Qualification.command"]
EXCLUDE_DIRS = {".venv", ".git", "__pycache__", "reports", "graphify-out",
                "runs", "dist", "backups", "scratch"}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build(root, outdir=None):
    from nexus_bench import assets as A
    from nexus_bench import release as R
    # verify -> bootstrap missing official assets -> verify again -> tree.
    # The builder never accepts an unverified package.
    ok, recs = A.verify_assets(root)
    if not ok:
        print("missing assets detected; bootstrapping official assets...")
        A.bootstrap(root)
        ok, recs = A.verify_assets(root)
    if not ok:
        bad = [r for r in recs if r["status"] != "OK"]
        raise SystemExit(f"refusing to build release with unverified assets: {bad}")
    A.ensure_accuracy_tree(root)
    # fresh release identity for exactly this tree
    import tools.make_release as MR
    MR.main()
    import platform as _pf
    plat = {"Windows": "Windows", "Darwin": "macOS"}.get(_pf.system(), "Linux")
    stamp = time.strftime("%Y%m%d")
    outdir = outdir or os.path.join(root, "dist")
    os.makedirs(outdir, exist_ok=True)
    name = f"NEXUS-Qualification-{plat}-v{R.QUALIFICATION_VERSION}-{stamp}.zip"
    dest = os.path.join(outdir, name)
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for top in INCLUDE_TOP + INCLUDE_LAUNCHERS:
            p = os.path.join(root, top)
            if os.path.isdir(p):
                for r, dirs, files in os.walk(p):
                    dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
                    for fn in sorted(files):
                        if fn.endswith((".pyc", ".log")) or fn == ".DS_Store":
                            continue
                        fp = os.path.join(r, fn)
                        z.write(fp, os.path.relpath(fp, root))
            elif os.path.exists(p):
                z.write(p, top)
    digest = sha256_file(dest)
    with open(dest + ".sha256", "w", encoding="utf-8") as f:
        f.write(f"{digest}  {name}\n")
    print(f"built {dest}\nsha256: {digest}")
    return dest, digest


if __name__ == "__main__":
    build(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
