"""DeepSORT-style multi-object tracking benchmark (real pipeline, numpy Kalman).

Per track: constant-velocity Kalman filter over (x, y, aspect, h). Association:
matching cascade over track age using appearance cosine distance gated by IoU,
then IoU fallback for unmatched — same structure as DeepSORT, with greedy
assignment (no scipy dependency) instead of Hungarian. Labelled honestly:
`deepsort_style_greedy` — swap in linear_sum_assignment if scipy is available.
"""
import time
import numpy as np

from nexus_bench import statuses as S
from nexus_bench.monitor import Monitor
from nexus_bench.reid_embed import embed_crops, load_embedder
from nexus_bench.profiles import resolve_device
from nexus_bench.stats import summarize

# --- Kalman filter (DeepSORT motion model, numpy) ---
_CHI2INV95 = {1: 3.8415, 2: 5.9915, 3: 7.8147, 4: 9.4877}

class KalmanFilter:
    def __init__(self):
        ndim, dt = 4, 1.0
        self._motion = np.eye(2 * ndim)
        for i in range(ndim):
            self._motion[i, ndim + i] = dt
        self._update = np.eye(ndim, 2 * ndim)
        self._std_w = np.array([1. / 20, 1. / 20, 1. / 160, 1. / 160,
                                1. / 40, 1. / 40, 1. / 320, 1. / 320])
        self._std_v = np.array([1. / 20, 1. / 20, 1. / 160, 1. / 160])

    def initiate(self, m):
        mean = np.r_[m, np.zeros_like(m)]
        std = np.concatenate([self._std_w[:4] * np.abs(m[3]), self._std_w[4:] * np.abs(m[3])])
        return mean, np.diag(std ** 2)

    def predict(self, mean, cov):
        std = np.concatenate([self._std_w[:4] * mean[3], self._std_w[4:] * mean[3]])
        return self._motion @ mean, self._motion @ cov @ self._motion.T + np.diag(std ** 2)

    def update(self, mean, cov, m):
        std = self._std_v * mean[3]
        H = self._update
        S = H @ cov @ H.T + np.diag(std ** 2)
        K = cov @ H.T @ np.linalg.inv(S)
        y = m - H @ mean
        return mean + K @ y, (np.eye(len(mean)) - K @ H) @ cov

def _tlwh_to_xyah(tlwh):
    x, y, w, h = tlwh
    return np.array([x + w / 2, y + h / 2, w / max(h, 1e-6), h])

def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0

def _greedy_match(cost, max_cost):
    """Greedy assignment. O(n^2 log n); Hungarian (scipy) is optimal — this is labelled."""
    pairs = sorted(((cost[i, j], i, j) for i in range(cost.shape[0])
                    for j in range(cost.shape[1]) if cost[i, j] <= max_cost))
    ri, cj, matches = set(), set(), []
    for _, i, j in pairs:
        if i not in ri and j not in cj:
            ri.add(i); cj.add(j); matches.append((i, j))
    return matches, [i for i in range(cost.shape[0]) if i not in ri], \
        [j for j in range(cost.shape[1]) if j not in cj]

class Track:
    def __init__(self, mean, cov, tid, feat):
        self.mean, self.cov, self.id = mean, cov, tid
        self.feats = [feat]
        self.hits, self.age, self.time_since_update = 1, 1, 0
        self.confirmed = False

    def predict(self, kf):
        self.mean, self.cov = kf.predict(self.mean, self.cov)
        self.age += 1; self.time_since_update += 1

    def update(self, kf, det_xyah, feat):
        self.mean, self.cov = kf.update(self.mean, self.cov, det_xyah)
        self.feats.append(feat)
        self.hits += 1; self.time_since_update = 0
        if self.hits >= 3:
            self.confirmed = True

    def tlbr(self):
        x, y, a, h = self.mean[:4]
        w = a * h
        return (x - w / 2, y - h / 2, x + w / 2, y + h / 2)

class Tracker:
    def __init__(self, max_age=30, max_cos=0.25, min_iou=0.5):
        self.kf = KalmanFilter()
        self.tracks, self._next = [], 0
        self.max_age, self.max_cos, self.min_iou = max_age, max_cos, min_iou

    def step(self, dets_xywh, feats):
        for t in self.tracks:
            t.predict(self.kf)
        N, M = len(self.tracks), len(dets_xywh)
        matches, u_trk, u_det = [], list(range(N)), list(range(M))
        if N and M:
            cost = np.ones((N, M))
            for i, t in enumerate(self.tracks):
                tb = t.tlbr()
                for j, (b, f) in enumerate(zip(dets_xywh, feats)):
                    x, y, w, h = b
                    iou_gate = _iou(tb, (x, y, x + w, y + h)) >= 0.3
                    cos = 1 - float(t.feats[-1] @ f)
                    cost[i, j] = cos if (cos <= self.max_cos and iou_gate) else 1e5
            confirmed = [i for i, t in enumerate(self.tracks) if t.confirmed]
            cascade = sorted(confirmed, key=lambda i: self.tracks[i].time_since_update)
            matched_t, matched_d = set(), set()
            for age in sorted(set(self.tracks[i].time_since_update for i in cascade)):
                ii = [i for i in cascade if self.tracks[i].time_since_update == age
                      and i not in matched_t]
                jj = [j for j in range(M) if j not in matched_d]
                if not ii or not jj:
                    continue
                sub = cost[np.ix_(ii, jj)]
                m, _, _ = _greedy_match(sub, self.max_cos)
                for a, b in m:
                    matches.append((ii[a], jj[b]))
                    matched_t.add(ii[a]); matched_d.add(jj[b])
            u_trk = [i for i in range(N) if i not in matched_t]
            u_det = [j for j in range(M) if j not in matched_d]
            # IoU fallback for recent unmatched tracks.
            ii = [i for i in u_trk if self.tracks[i].time_since_update <= 1]
            if ii and u_det:
                iou_cost = np.array([[1 - _iou(self.tracks[i].tlbr(),
                                               (dets_xywh[j][0], dets_xywh[j][1],
                                                dets_xywh[j][0] + dets_xywh[j][2],
                                                dets_xywh[j][1] + dets_xywh[j][3]))
                                      for j in u_det] for i in ii])
                m, _, u = _greedy_match(iou_cost, 1 - self.min_iou)
                for a, b in m:
                    matches.append((ii[a], u_det[b]))
                matched_d2 = {u_det[b] for _, b in m}
                u_det = [j for j in u_det if j not in matched_d2]
                u_trk = [i for i in u_trk if i not in {ii[a] for a, _ in m}]
        for ti, di in matches:
            self.tracks[ti].update(self.kf, _tlwh_to_xyah(np.array(dets_xywh[di], float)), feats[di])
        for j in u_det:
            m, c = self.kf.initiate(_tlwh_to_xyah(np.array(dets_xywh[j], float)))
            self.tracks.append(Track(m, c, self._next, feats[j]))
            self._next += 1
        self.tracks = [t for i, t in enumerate(self.tracks)
                       if not (i in u_trk and t.time_since_update > self.max_age)]
        return [(t.id, t.tlbr()) for t in self.tracks if t.confirmed]

# --- Seeded scenario with ground truth: linear motion + crossing + occlusion gap ---
def make_scenario(n_targets=6, n_frames=60, W=640, H=480, seed=0):
    rng = np.random.default_rng(seed)
    P = rng.uniform([50, 50], [W - 150, H - 150], (n_targets, 2))
    V = rng.uniform(-3, 3, (n_targets, 2))
    S = rng.uniform(40, 70, n_targets)
    frames = []
    for f in range(n_frames):
        dets, gts = [], []
        for i in range(n_targets):
            x, y = P[i] + V[i] * f
            x = min(max(x, 0), W - S[i]); y = min(max(y, 0), H - S[i])
            if 20 <= f < 28 and i % 2 == 0:
                continue  # occlusion gap: no detection
            jx, jy = rng.normal(0, 1.5, 2)
            dets.append([x + jx, y + jy, S[i], S[i]])
            gts.append(i)
        frames.append((dets, gts))
    return frames

def _crop_for(feat_rng, gid, size=(128, 64, 3)):
    """Deterministic per-identity crop texture so appearance is informative but seeded."""
    rng = np.random.default_rng(10_000 + gid)
    base = rng.integers(0, 255, size, dtype=np.uint8).astype(np.int32)
    tint = np.zeros_like(base); tint[:, :, gid % 3] = 60
    return np.clip(base + tint, 0, 255).astype(np.uint8)

def run(cfg):
    dev = resolve_device(cfg.get("device", "auto"))
    n_frames = max(20, cfg.get("repeats", 20) * 2)
    out = {"module": "tracking",
           "config": {**cfg, "resolved_device": dev,
                      "method": "deepsort_style_greedy (Kalman + appearance cascade; greedy, not Hungarian)"},
           "tests": {}, "errors": [], "status": S.FULL}
    with Monitor() as mon:
        try:
            model, tag = load_embedder(dev, (cfg.get("reid_model") or "").strip())
            out["config"]["embedder"] = tag
            frames = make_scenario(n_frames=n_frames, seed=cfg.get("seed", 0))
            for _ in range(min(2, cfg.get("warmup", 1))):
                embs, _, _ = embed_crops(model, [_crop_for(None, 0)], dev)
            tr = Tracker()
            assoc_ts, emb_ts, switches, total = [], [], 0, 0
            last_id_of = {}
            for dets, gts in frames:
                crops = [_crop_for(None, g) for g in gts]
                t0 = time.perf_counter()
                feats = []
                if crops:
                    embs, _, _ = embed_crops(model, crops, dev)
                    feats = [e for e in embs]
                emb_ts.append((time.perf_counter() - t0) * 1000)
                t0 = time.perf_counter()
                live = tr.step(dets, feats)
                assoc_ts.append((time.perf_counter() - t0) * 1000)
                # ID-switch count vs ground truth on confirmed tracks.
                idmap = {}
                for tid, tb in live:
                    cx = (tb[0] + tb[2]) / 2
                    best, bestd = None, 1e9
                    for (bx, by, bw, bh), g in zip(dets, gts):
                        d = abs((bx + bw / 2) - cx)
                        if d < bestd:
                            best, bestd = g, d
                    if best is not None and bestd < 40:
                        idmap[best] = tid
                for g, tid in idmap.items():
                    total += 1
                    if g in last_id_of and last_id_of[g] != tid:
                        switches += 1
                    last_id_of[g] = tid
            a = summarize(assoc_ts); a["id_switches"] = switches
            a["gt_assignments"] = total
            a["switch_rate"] = round(switches / total, 4) if total else None
            out["tests"]["association_per_frame_ms"] = a
            e = summarize(emb_ts); e["note"] = "embedding inference supporting association"
            out["tests"]["embedding_per_frame_ms"] = e
            out["tests"]["live_tracks"] = len(tr.tracks)
        except Exception as e:
            out["errors"].append(str(e))
        out["telemetry"] = mon.summary()
    return out
