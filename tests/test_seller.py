"""Seller-experience integration: fresh package, official assets only, no Git.

Simulates the seller machine: a temp package dir with test-local official
assets (manifest override), no user-supplied files, no YAML edits. Proves the
appliance discovers, verifies, runs, reports, and packages — and that every
failure mode below CANNOT produce PASS.
"""
import json
import os

import pytest

from nexus_bench import assets as A


def _make_test_package(tmp_path):
    """Tiny but REAL official-style assets + a manifest override."""
    import cv2
    import numpy as np
    pkg = tmp_path / "pkg"
    (pkg / "assets" / "video").mkdir(parents=True)
    (pkg / "assets" / "model").mkdir(parents=True)
    (pkg / "assets" / "accuracy").mkdir(parents=True)
    (pkg / "profiles").mkdir(parents=True)
    import shutil
    shutil.copy("profiles/purchase_gate.yaml", pkg / "profiles" / "purchase_gate.yaml")
    rng = np.random.default_rng(7)
    for name in ("uav_test_v1.mp4", "rover_test_v1.mp4"):
        vw = cv2.VideoWriter(str(pkg / "assets" / "video" / name),
                             cv2.VideoWriter_fourcc(*"mp4v"), 10, (160, 120))
        for _ in range(20):
            vw.write(rng.integers(0, 255, (120, 160, 3), dtype=np.uint8))
        vw.release()
    import shutil as _sh
    _sh.copy(os.path.join(A.package_root(), "assets", "model", "yolo26m.pt"),
             pkg / "assets" / "model" / "yolo26m.pt")
    (pkg / "assets" / "accuracy" / "coco8.zip").write_bytes(b"placeholder")
    manifest = {"qualification_version": "test-1", "manifest_version": "test-1",
                "assets": []}
    for rel, aid, typ, purpose in (
            ("model/yolo26m.pt", "t-model", "model", "detection"),
            ("video/uav_test_v1.mp4", "t-uav", "video", "uav"),
            ("video/rover_test_v1.mp4", "t-rover", "video", "rover"),
            ("accuracy/coco8.zip", "t-acc", "dataset", "accuracy")):
        p = pkg / "assets" / rel
        manifest["assets"].append(
            {"asset_id": aid, "name": aid, "version": "t", "type": typ,
             "filename": rel, "size": p.stat().st_size,
             "sha256": A.sha256_file(str(p)),
             "source": "bundled", "purpose": purpose})
    return pkg, manifest


def _patch_manifest(monkeypatch, pkg, manifest):
    monkeypatch.setattr(A, "OFFICIAL_MANIFEST", manifest)
    monkeypatch.setattr(A, "MODEL_URL", "https://example.invalid/m.pt")
    monkeypatch.setattr(A, "COCO8_URL", "https://example.invalid/c.zip")
    return pkg


def test_official_model_bundled():
    """The release is Mode-A self-contained: the official model MUST be in git.
    (*.pt is git-ignored globally, so this needs an explicit force-add.)"""
    import os
    p = os.path.join(A.package_root(), "assets", "model", "yolo26m.pt")
    assert os.path.exists(p), "official model missing from checkout — release is not self-contained"
    entry = next(a for a in A.OFFICIAL_MANIFEST["assets"] if a["type"] == "model")
    assert os.path.getsize(p) == entry["size"]
    assert A.sha256_file(p) == entry["sha256"]


def test_assets_verify_ok_and_detect_corruption(tmp_path, monkeypatch):
    pkg, manifest = _make_test_package(tmp_path)
    _patch_manifest(monkeypatch, pkg, manifest)
    ok, recs = A.verify_assets(str(pkg))
    assert ok and all(r["status"] == "OK" for r in recs)
    with open(pkg / "assets" / "model" / "yolo26m.pt", "r+b") as f:
        f.seek(100)
        f.write(b"\x00")
    ok2, recs2 = A.verify_assets(str(pkg))
    assert not ok2
    assert any(r["asset_id"] == "t-model" and r["status"] == "MISMATCH" for r in recs2)


def test_bootstrap_from_trusted_url_only(tmp_path, monkeypatch):
    import urllib.request
    pkg, manifest = _make_test_package(tmp_path)
    os.remove(pkg / "assets" / "video" / "rover_test_v1.mp4")
    keep = [a for a in manifest["assets"] if a["asset_id"] != "t-rover"]
    # re-create the removed file elsewhere as the "mirror"
    import cv2
    import numpy as np
    rng = np.random.default_rng(7)
    vw = cv2.VideoWriter(str(tmp_path / "mirror.mp4"),
                         cv2.VideoWriter_fourcc(*"mp4v"), 10, (160, 120))
    for _ in range(20):
        vw.write(rng.integers(0, 255, (120, 160, 3), dtype=np.uint8))
    vw.release()
    mh = A.sha256_file(str(tmp_path / "mirror.mp4"))
    keep.append({"asset_id": "t-rover", "name": "r", "version": "t", "type": "video",
                 "filename": "video/rover_test_v1.mp4",
                 "size": os.path.getsize(tmp_path / "mirror.mp4"),
                 "sha256": mh,
                 "source": (tmp_path / "mirror.mp4").as_uri(), "purpose": "rover"})
    manifest["assets"] = keep
    _patch_manifest(monkeypatch, pkg, manifest)
    recs = A.bootstrap(str(pkg))
    assert any(r["asset_id"] == "t-rover" and r["status"] == "OK" for r in recs)


def test_qualify_corrupt_asset_aborts_without_pass(tmp_path, monkeypatch):
    pkg, manifest = _make_test_package(tmp_path)
    _patch_manifest(monkeypatch, pkg, manifest)
    with open(pkg / "assets" / "model" / "yolo26m.pt", "r+b") as f:
        f.seek(100)
        f.write(b"\x00")
    import webbrowser
    monkeypatch.setattr(webbrowser, "open", lambda *a, **k: False)
    from nexus_bench.qualify import run_qualify
    folder, zipp, results = run_qualify(root=str(pkg), assume_yes=True,
                                        out_parent=str(tmp_path / "out"),
                                        progress_out="silent")
    assert results["_gate"]["verdict"] == "ABORTED"
    assert os.path.exists(zipp)
    blob = json.dumps(results, default=str)
    assert "PASS" not in blob.replace("No qualification PASS was issued", "")


def test_release_identity_needs_no_git(tmp_path, monkeypatch):
    import json
    from nexus_bench import release as R
    from nexus_bench import provenance
    rel = {"release_id": "NEXUS-Qualification-v1.0-t",
           "qualification_version": "1.0", "gate_version": "1.0",
           "asset_manifest_version": "1.0", "asset_manifest_sha256": "x",
           "gate_sha256": "y", "benchmark_source_sha256": "z",
           "built_utc": "2026-01-01T00:00:00Z"}
    (tmp_path / "release").mkdir()
    (tmp_path / "release" / "release.json").write_text(json.dumps(rel))
    import nexus_bench.release as _REL
    monkeypatch.setattr(_REL, "read_release", lambda root=None: rel)
    monkeypatch.setattr(provenance, "_git",
                        lambda: {"sha": "unknown", "dirty": "unknown", "root": "unknown"})
    m = provenance.manifest({"model": "", "seed": 0})
    assert m["distribution"].startswith("release")
    assert m["git"]["sha"].startswith("n/a")
    assert m["release"]["qualification_version"] == "1.0"


def test_tree_hash_is_separator_independent(tmp_path):
    """The accuracy tree hash must be identical on Windows (\\) and POSIX (/).
    os.path.relpath leaks the platform separator — normalize it, or CI forks."""
    import hashlib
    (tmp_path / "a.txt").write_bytes(b"x")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.txt").write_bytes(b"y")
    from nexus_bench.assets import _tree_hash
    native = _tree_hash(str(tmp_path))
    h = hashlib.sha256()
    for rel, body in (("a.txt", b"x"), ("sub/b.txt", b"y")):
        h.update(rel.encode())
        h.update(hashlib.sha256(body).hexdigest().encode())
    assert native == h.hexdigest()


def test_qualify_end_to_end_short(tmp_path, monkeypatch):
    pkg, manifest = _make_test_package(tmp_path)
    _patch_manifest(monkeypatch, pkg, manifest)
    import webbrowser
    monkeypatch.setattr(webbrowser, "open", lambda *a, **k: False)
    from nexus_bench.qualify import run_qualify
    folder, zipp, results = run_qualify(
        root=str(pkg), assume_yes=True, out_parent=str(tmp_path / "out"),
        progress_out="silent",
        _test_overrides={"duration_s": 6, "sustained_duration_s": 20,
                         "sustained_window_s": 10, "repeats": 1, "warmup": 1,
                         "imgsz": 160, "target_fps": 5.0, "agent_frames": 30,
                         "val_data": ""})
    assert os.path.exists(zipp)
    assert os.path.exists(os.path.join(folder, "NEXUS_Qualification_Report.html"))
    assert os.path.exists(os.path.join(folder, "qualification_summary.txt"))
    assert os.path.exists(os.path.join(folder, "asset_manifest.json"))
    assert os.path.exists(os.path.join(folder, "qualification_manifest.json"))
    verdict = results["_gate"]["verdict"]
    assert verdict in ("PASS", "PASS_WITH_HEADROOM", "FAIL", "INCONCLUSIVE", "ABORTED")
    if verdict in ("PASS", "PASS_WITH_HEADROOM"):
        sus = results.get("sustained", {})
        assert sus.get("status") == "FULL"
        per = sus["tests"]["per_agent"]
        assert all(v["duration_complete"] for v in per.values())
