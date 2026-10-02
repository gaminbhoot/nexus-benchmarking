#!/usr/bin/env python3
"""Build the immutable seller release identity: release/release.json.

Run at release time (NOT by the seller). Records qualification/gate versions,
asset manifest hash, gate config hash, and benchmark source hash. Changing the
model, video, gate threshold, or benchmark code => bump versions and rebuild.
"""
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nexus_bench import assets as A
from nexus_bench import release as R


def main():
    root = A.package_root()
    manifest = A.OFFICIAL_MANIFEST
    msha = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    rel = {
        "release_id": f"NEXUS-Qualification-v{R.QUALIFICATION_VERSION}-"
                      f"{time.strftime('%Y%m%d')}",
        "qualification_version": R.QUALIFICATION_VERSION,
        "gate_version": R.GATE_VERSION,
        "asset_manifest_version": manifest["manifest_version"],
        "asset_manifest_sha256": msha,
        "gate_sha256": R.gate_sha(root),
        "requirements_sha256": R.requirements_sha(root),
        "benchmark_source_sha256": R.source_tree_sha(root),
        "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    os.makedirs(os.path.join(root, "release"), exist_ok=True)
    with open(R.release_path(root), "w", encoding="utf-8") as f:
        json.dump(rel, f, indent=2)
    print(json.dumps(rel, indent=2))


if __name__ == "__main__":
    main()
