"""3D mapping: point-cloud ops, ORB features, synthetic visual odometry."""
import time
import numpy as np

from nexus_bench.monitor import Monitor
from nexus_bench.stats import summarize

def run(cfg):
    n_pts = 20000
    rng = np.random.default_rng(cfg.get("seed", 0))
    cloud = rng.random((n_pts, 3)).astype(np.float32)
    out = {"module": "mapping3d", "config": cfg, "tests": {}, "errors": []}
    with Monitor() as mon:
        try:  # voxel downsample proxy + transform (repeatable)
            ts = []
            for _ in range(max(3, cfg.get("repeats", 10))):
                t0 = time.perf_counter()
                q = np.floor(cloud * 20).astype(np.int32)
                _, idx = np.unique(q, axis=0, return_index=True)
                down = cloud[idx]
                theta = 0.01
                R = np.array([[np.cos(theta), -np.sin(theta), 0],
                              [np.sin(theta), np.cos(theta), 0], [0, 0, 1]], dtype=np.float32)
                _ = down @ R.T + 0.05
                ts.append((time.perf_counter() - t0) * 1000)
            d = summarize(ts); d["downsampled_pts"] = int(len(idx))
            out["tests"]["voxel_downsample_transform_ms"] = d
        except Exception as e:
            out["errors"].append(f"pointcloud: {e}")
        try:  # nearest-neighbor proxy on subset (ponytail: O(n^2) on 2k subset, fine for bench)
            sub = cloud[:2000]
            t0 = time.perf_counter()
            d2 = ((sub[:, None, :] - sub[None, :, :]) ** 2).sum(-1)
            np.fill_diagonal(d2, np.inf)
            nn = d2.min(axis=1)
            dt = (time.perf_counter() - t0) * 1000
            out["tests"]["nearest_neighbor_2k_ms"] = {"total_ms": round(dt, 1), "mean_nn_dist": round(float(nn.mean()), 4)}
        except Exception as e:
            out["errors"].append(f"nn: {e}")
        try:  # ORB feature extraction on synthetic frames
            import cv2
            img = (rng.random((480, 640)) * 255).astype(np.uint8)
            orb = cv2.ORB_create(nfeatures=500)
            ts = []
            for _ in range(max(3, cfg.get("repeats", 10))):
                t0 = time.perf_counter()
                kp, des = orb.detectAndCompute(img, None)
                ts.append((time.perf_counter() - t0) * 1000)
            d = summarize(ts); d["keypoints"] = len(kp)
            out["tests"]["orb_features_ms"] = d
        except Exception as e:
            out["errors"].append(f"orb: {e} (opencv-contrib missing?)")
        try:  # synthetic visual odometry: chained noisy transforms, drift report
            pose = np.eye(4, dtype=np.float64); traj = [pose[:3, 3].copy()]
            for _ in range(100):
                step = np.eye(4); step[:3, 3] = [0.1, 0.01, 0]
                step[:3, :3] += rng.normal(0, 1e-4, (3, 3))
                pose = pose @ step; traj.append(pose[:3, 3].copy())
            traj = np.array(traj)
            out["tests"]["visual_odometry_100steps"] = {
                "final_xyz": [round(float(v), 3) for v in traj[-1]],
                "path_length": round(float(np.abs(np.diff(traj, axis=0)).sum()), 3)}
        except Exception as e:
            out["errors"].append(f"vo: {e}")
        out["telemetry"] = mon.summary()
    return out
