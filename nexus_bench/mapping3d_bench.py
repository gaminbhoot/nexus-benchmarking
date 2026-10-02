"""3D mapping: a REAL two-view visual-odometry pipeline + labelled microbenchmarks.

VO pipeline (all real, OpenCV): seeded 3D scene -> two camera views with KNOWN
relative pose -> ORB detect -> brute-force match + ratio test -> essential matrix
(RANSAC) -> recoverPose -> rotation/translation error vs ground truth ->
triangulate -> mean reprojection error. Point-cloud voxel work is kept but
labelled `preprocessing_microbench`, not odometry.
"""
import time
import numpy as np

from nexus_bench import statuses as S
from nexus_bench.monitor import Monitor
from nexus_bench.stats import summarize

def _render_view(pts3d, K, R, t, H=480, W=640, seed=0):
    """Project 3D points; draw textured patches so ORB has real features to find."""
    import cv2
    rng = np.random.default_rng(seed)
    img = np.zeros((H, W, 3), dtype=np.uint8)
    P = K @ np.hstack([R, t.reshape(3, 1)])
    uv = (P @ np.vstack([pts3d.T, np.ones(len(pts3d))]))
    uv = (uv[:2] / uv[2]).T
    tex = rng.integers(80, 255, (24, 24), dtype=np.uint8)
    for (u, v) in uv.astype(int):
        if 12 < u < W - 12 and 12 < v < H - 12:
            patch = tex.astype(np.int32) + rng.integers(-20, 20, (24, 24))
            img[v - 12:v + 12, u - 12:u + 12] = np.clip(patch, 0, 255).astype(np.uint8)[:, :, None].repeat(3, 2)
    return img, uv

def _vo_pair(seed=0, n_pts=300):
    rng = np.random.default_rng(seed)
    pts3d = rng.uniform([-2, -1.5, 4], [2, 1.5, 8], (n_pts, 3))
    K = np.array([[500., 0, 320.], [0, 500., 240.], [0, 0, 1.]])
    R0 = np.eye(3)
    t0 = np.zeros(3)
    ang = np.deg2rad(4.0)
    R1 = np.array([[np.cos(ang), 0, np.sin(ang)], [0, 1, 0], [-np.sin(ang), 0, np.cos(ang)]])
    t1 = np.array([0.35, 0.02, 0.0])
    img0, _ = _render_view(pts3d, K, R0, t0, seed=seed)
    img1, _ = _render_view(pts3d, K, R1, t1, seed=seed + 1)
    return img0, img1, K, R1, t1

def _vo_once(img0, img1, K, R_gt, t_gt):
    import cv2
    t = {}
    t0 = time.perf_counter()
    orb = cv2.ORB_create(nfeatures=1500)
    g0 = cv2.cvtColor(img0, cv2.COLOR_RGB2GRAY)
    g1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    k0, d0 = orb.detectAndCompute(g0, None)
    k1, d1 = orb.detectAndCompute(g1, None)
    t["detect_ms"] = (time.perf_counter() - t0) * 1000
    if d0 is None or d1 is None or len(d0) < 16 or len(d1) < 16:
        return {"ok": False, "reason": "too few features"}
    t0 = time.perf_counter()
    matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(d0, d1, k=2)
    good = [m for m, n_ in matches if m.distance < 0.75 * n_.distance]
    t["match_ms"] = (time.perf_counter() - t0) * 1000
    if len(good) < 16:
        return {"ok": False, "reason": f"only {len(good)} good matches"}
    p0 = np.float32([k0[m.queryIdx].pt for m in good])
    p1 = np.float32([k1[m.trainIdx].pt for m in good])
    t0 = time.perf_counter()
    E, mask = cv2.findEssentialMat(p0, p1, K, method=cv2.RANSAC, threshold=1.0)
    if E is None:
        return {"ok": False, "reason": "essential matrix failed"}
    _, R, tvec, mask2 = cv2.recoverPose(E, p0, p1, K, mask=mask)
    t["estimate_ms"] = (time.perf_counter() - t0) * 1000
    inl = (int(mask.ravel().sum()), int(mask2.ravel().sum())) if mask is not None else (0, 0)
    # Rotation error (deg) + translation direction error (deg, scale-free).
    cos_r = np.clip((np.trace(R @ R_gt.T) - 1) / 2, -1, 1)
    rot_err = float(np.rad2deg(np.arccos(cos_r)))
    te, tg = tvec.ravel() / (np.linalg.norm(tvec) + 1e-9), t_gt / (np.linalg.norm(t_gt) + 1e-9)
    trans_err = float(np.rad2deg(np.arccos(np.clip(abs(te @ tg), -1, 1))))
    t0 = time.perf_counter()
    P0 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P1 = K @ np.hstack([R, tvec])
    X = cv2.triangulatePoints(P0, P1, p0.T, p1.T)
    X = (X[:3] / X[3]).T
    proj = (P1 @ np.vstack([X.T, np.ones(len(X))]))
    reproj = float(np.mean(np.linalg.norm((proj[:2] / proj[2]).T - p1, axis=1)))
    t["triangulate_ms"] = (time.perf_counter() - t0) * 1000
    return {"ok": True, "kp": (len(k0), len(k1)), "matches": len(good),
            "inliers_E": inl[0], "rot_err_deg": round(rot_err, 2),
            "trans_dir_err_deg": round(trans_err, 2),
            "reproj_px": round(reproj, 2), "stage_ms": {k: round(v, 2) for k, v in t.items()}}

def run(cfg):
    rng_seed = cfg.get("seed", 0)
    repeats = max(3, cfg.get("repeats", 10))
    out = {"module": "mapping3d", "config": {**cfg, "method": "two-view ORB->E->pose->triangulate vs GT"},
           "tests": {}, "errors": [], "status": S.FULL}
    with Monitor() as mon:
        try:
            img0, img1, K, R_gt, t_gt = _vo_pair(seed=rng_seed)
            total, succ, rot, tr, rp = [], 0, [], [], []
            stage_acc = {}
            for i in range(repeats):
                t0 = time.perf_counter()
                r = _vo_once(img0, img1, K, R_gt, t_gt)
                total.append((time.perf_counter() - t0) * 1000)
                if r.get("ok"):
                    succ += 1
                    rot.append(r["rot_err_deg"]); tr.append(r["trans_dir_err_deg"]); rp.append(r["reproj_px"])
                    for k, v in r["stage_ms"].items():
                        stage_acc.setdefault(k, []).append(v)
                elif i == 0:
                    out["errors"].append(f"vo attempt failed: {r.get('reason')}")
            d = summarize(total)
            d["success_rate"] = round(succ / repeats, 2)
            if rot:
                d["mean_rot_err_deg"] = round(float(np.mean(rot)), 2)
                d["mean_trans_dir_err_deg"] = round(float(np.mean(tr)), 2)
                d["mean_reproj_px"] = round(float(np.mean(rp)), 2)
                d["stage_means_ms"] = {k: round(float(np.mean(v)), 2) for k, v in stage_acc.items()}
                d["note"] = ("errors vs KNOWN ground-truth pose — small errors mean the "
                             "pipeline is geometrically sound on this scene")
            out["tests"]["visual_odometry_two_view_ms"] = d
        except Exception as e:
            out["errors"].append(f"vo: {e}")
        try:  # labelled preprocessing microbenchmark (NOT odometry)
            rng = np.random.default_rng(rng_seed)
            cloud = rng.random((20000, 3)).astype(np.float32)
            ts = []
            for _ in range(max(3, repeats)):
                t0 = time.perf_counter()
                q = np.floor(cloud * 20).astype(np.int32)
                _, idx = np.unique(q, axis=0, return_index=True)
                _ = cloud[idx]
                ts.append((time.perf_counter() - t0) * 1000)
            d = summarize(ts)
            d["label"] = "preprocessing_microbench (voxel hashing only)"
            out["tests"]["pointcloud_voxel_hash_ms"] = d
        except Exception as e:
            out["errors"].append(f"pointcloud: {e}")
        out["telemetry"] = mon.summary()
    return out
