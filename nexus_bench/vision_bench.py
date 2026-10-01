"""Computer vision: detect + centroid tracking, Re-ID embedding, optical flow, decode, video pipeline."""
import time
import numpy as np

from nexus_bench.monitor import Monitor
from nexus_bench.stats import summarize

def _frames(n=20, h=480, w=640, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.integers(0, 255, (h, w, 3), dtype=np.uint8) for _ in range(n)]

class CentroidTracker:
    """Minimal multi-object tracker (no extra deps)."""
    def __init__(self):
        self.next_id = 0; self.tracks = {}
    def update(self, boxes):
        ids = []
        for b in boxes:
            cx = (b[0] + b[2]) / 2
            best, bestd = None, 1e9
            for tid, (pcx, _) in self.tracks.items():
                d = abs(pcx - cx)
                if d < bestd:
                    best, bestd = tid, d
            if best is not None and bestd < 50:
                self.tracks[best] = (cx, 0); ids.append(best)
            else:
                self.tracks[self.next_id] = (cx, 0); ids.append(self.next_id); self.next_id += 1
        return ids

def run(cfg):
    import cv2
    n = max(5, cfg.get("repeats", 20))
    frames = _frames(n)
    out = {"module": "vision", "config": cfg, "tests": {}, "errors": []}
    with Monitor() as mon:
        try:  # detection + tracking pipeline (synthetic boxes move linearly)
            tr = CentroidTracker()
            ts = []
            for i, f in enumerate(frames):
                t0 = time.perf_counter()
                gray = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
                _, th = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
                cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                boxes = [cv2.boundingRect(c) for c in cnts[:8]]
                boxes = [(x, y, x + w, y + h) for x, y, w, h in boxes]
                tr.update(boxes)
                ts.append((time.perf_counter() - t0) * 1000)
            d = summarize(ts); d["active_tracks"] = tr.next_id
            out["tests"]["detect_track_ms"] = d
        except Exception as e:
            out["errors"].append(f"detect_track: {e}")
        try:  # Re-ID proxy: normalized crop embedding distance
            rng = np.random.default_rng(1)
            embs = rng.random((50, 128)); embs /= (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9)
            t0 = time.perf_counter()
            dists = ((embs[:, None, :] - embs[None, :, :]) ** 2).sum(-1)
            dt = (time.perf_counter() - t0) * 1000
            out["tests"]["reid_50x128_cosine_ms"] = {"total_ms": round(dt, 2), "mean_pairwise": round(float(dists.mean()), 4)}
        except Exception as e:
            out["errors"].append(f"reid: {e}")
        try:  # optical flow
            ts = []
            prev = cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY)
            for f in frames[1:]:
                gray = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
                t0 = time.perf_counter()
                _ = cv2.calcOpticalFlowFarneback(prev, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                ts.append((time.perf_counter() - t0) * 1000)
                prev = gray
            out["tests"]["optical_flow_ms"] = summarize(ts)
        except Exception as e:
            out["errors"].append(f"optflow: {e}")
        try:  # full video pipeline: decode->resize->detect->annotate
            ts = []
            for f in frames:
                t0 = time.perf_counter()
                small = cv2.resize(f, (320, 240))
                gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
                _ = cv2.Canny(gray, 100, 200)
                ts.append((time.perf_counter() - t0) * 1000)
            out["tests"]["video_pipeline_ms"] = summarize(ts)
        except Exception as e:
            out["errors"].append(f"pipeline: {e}")
        out["telemetry"] = mon.summary()
    return out
