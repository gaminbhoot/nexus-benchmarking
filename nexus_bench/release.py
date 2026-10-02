"""Release identity: versioned qualification that needs no Git.

A packaged seller distribution carries release/release.json (written by
tools/make_release.py). Provenance prefers it; .git is not required and a
missing .git never penalizes a legitimate release. Changing model, video,
gate threshold, or benchmark code => new qualification version.
"""
import hashlib
import json
import os

# Single source of truth: the asset manifest defines the qualification.
from nexus_bench.assets import ASSET_MANIFEST_VERSION, QUALIFICATION_VERSION
GATE_VERSION = "1.0"

def package_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def release_path(root=None):
    return os.path.join(root or package_root(), "release", "release.json")

def read_release(root=None):
    try:
        with open(release_path(root), encoding="utf-8") as f:
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

def requirements_sha(root=None):
    h = hashlib.sha256()
    with open(os.path.join(root or package_root(), "requirements.lock"), "rb") as f:
        h.update(f.read())
    return h.hexdigest()

def verify_runtime(root=None):
    """Verify the packaged release BEFORE any qualification runs.

    Checks: current source tree == release benchmark_source_sha256,
    current profiles == release gate_sha256, current asset manifest ==
    release asset_manifest_sha256. Any mismatch => the package was modified
    after release stamping (or hand-assembled) and must NOT qualify.
    Returns (ok, [problems]). No release.json => developer tree (no claim).
    """
    rel = read_release(root)
    if rel is None:
        return True, ["no release.json: developer tree, no release claim made"]
    problems = []
    if source_tree_sha(root) != rel.get("benchmark_source_sha256"):
        problems.append("benchmark source differs from release manifest "
                        "(code modified after release stamping)")
    if gate_sha(root) != rel.get("gate_sha256"):
        problems.append("gate/profile files differ from release manifest "
                        "(official qualification profile was modified)")
    try:
        if requirements_sha(root) != rel.get("requirements_sha256"):
            problems.append("requirements.lock differs from release manifest "
                            "(pinned runtime was modified)")
    except Exception:
        problems.append("requirements.lock missing — pinned runtime unverifiable")
    msha = hashlib.sha256(json.dumps(
        __import__("nexus_bench.assets", fromlist=["x"]).OFFICIAL_MANIFEST,
        sort_keys=True).encode()).hexdigest()
    if msha != rel.get("asset_manifest_sha256"):
        problems.append("asset manifest differs from release manifest")
    return (not problems), problems
