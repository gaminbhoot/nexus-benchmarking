#!/usr/bin/env python3
"""Generate the OFFICIAL NEXUS qualification video assets (deterministic).

Seeded motion sequences, versioned as v1. After generation, record the SHA-256
hashes into nexus_bench/assets.py OFFICIAL_MANIFEST. Regenerating with the same
seed + OpenCV version MUST reproduce byte-identical files; any difference in a
shipped package fails verification (QUALIFICATION INVALID).
"""
import hashlib
import os
import sys

import cv2
import numpy as np

VERSION = "1.0"
W, H, FPS = 640, 480, 30

def make_sequence(path, seconds, seed, n_targets=6):
    rng = np.random.default_rng(seed)
    base = rng.integers(40, 90, (H, W), dtype=np.uint8)  # textured ground
    P = rng.uniform([20, 20], [W - 120, H - 120], (n_targets, 2))
    V = rng.uniform(-90, 90, (n_targets, 2))  # px/sec
    S = rng.uniform(35, 80, n_targets)
    C = rng.integers(150, 255, (n_targets, 3), dtype=np.uint8)
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    n = int(seconds * FPS)
    for f in range(n):
        img = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
        t = f / FPS
        for i in range(n_targets):
            x = (P[i][0] + V[i][0] * t) % (W - 90)
            y = (P[i][1] + V[i][1] * t) % (H - 90)
            cv2.rectangle(img, (int(x), int(y)),
                          (int(x + S[i]), int(y + S[i])),
                          tuple(int(c) for c in C[i]), -1)
        vw.write(img)
    vw.release()
    return n

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    outdir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "assets", "video")
    os.makedirs(outdir, exist_ok=True)
    specs = [("uav_test_v1.mp4", 20, 1001, 7), ("rover_test_v1.mp4", 20, 2002, 5)]
    print(f"opencv {cv2.__version__}")
    for name, secs, seed, tgt in specs:
        path = os.path.join(outdir, name)
        n = make_sequence(path, secs, seed, tgt)
        size = os.path.getsize(path)
        print(f"{name}: {n} frames, {size} bytes, sha256={sha256(path)}")
    print("Pin these hashes into nexus_bench/assets.py OFFICIAL_MANIFEST.")

if __name__ == "__main__":
    main()
