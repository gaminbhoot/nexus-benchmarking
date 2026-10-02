"""Release identity: versioned qualification that needs no Git.

A packaged seller distribution carries release/release.json (written by
tools/make_release.py). Provenance prefers it; .git is not required and a
missing .git never penalizes a legitimate release. Changing model, video,
gate threshold, or benchmark code => new qualification version.
"""
import hashlib
import json
import os

QUALIFICATION_VERSION = "1.0"
GATE_VERSION = "1.0"

def package_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def release_path(root=None):
    return os.path.join(root or package_root(), "release", "release.json")

def read_release(root=None):
    try:
        with open(release_path(root)) as f:
            return json.load(f)
    except Exception:
        return None

def source_tree_sha(root=None):
    h = hashlib.sha256()
    bench = os.path.join(root or package_root(), "nexus_bench")
    for fn in sorted(os.listdir(bench)):
        if fn.endswith(".py"):
            with open(os.path.join(bench, fn), "rb") as f:
                h.update(fn.encode())
                h.update(f.read())
    return h.hexdigest()

def gate_sha(root=None):
    import glob
    h = hashlib.sha256()
    for p in sorted(glob.glob(os.path.join(root or package_root(), "profiles", "*.yaml"))):
        with open(p, "rb") as f:
            h.update(os.path.basename(p).encode())
            h.update(f.read())
    return h.hexdigest()
