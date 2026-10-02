"""Real Re-ID workload: a torch person-embedding CNN run on image crops.

This is a lightweight reference embedding network (not OSNet/ResNet-scale), so it
is labelled `reid_reference_embedder`, NOT as your production Re-ID model. Pass
--reid-model <state_dict.pt> to benchmark your actual embedding weights instead:
if the file loads and produces a (N,D) embedding, it is used and reported as
`reid_custom_weights`.
"""
import time
import numpy as np
import torch
import torch.nn as nn

from nexus_bench import statuses as S
from nexus_bench.monitor import Monitor
from nexus_bench.profiles import resolve_device
from nexus_bench.stats import summarize

EMB_DIM = 128

class TinyEmbedNet(nn.Module):
    def __init__(self, dim=EMB_DIM):
        super().__init__()
        def block(c_in, c_out):
            return nn.Sequential(nn.Conv2d(c_in, c_out, 3, padding=1, stride=2),
                                 nn.BatchNorm2d(c_out), nn.ReLU(inplace=True))
        self.feat = nn.Sequential(block(3, 32), block(32, 64), block(64, 128))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, dim)

    def forward(self, x):
        x = self.feat(x)
        x = self.pool(x).flatten(1)
        x = self.fc(x)
        return x / (x.norm(dim=1, keepdim=True) + 1e-9)

def load_embedder(dev, weights_path=""):
    """Returns (model, tag). Custom weights must be a state_dict compatible with TinyEmbedNet."""
    model = TinyEmbedNet().to(dev).eval()
    tag = "reid_reference_embedder (lightweight CNN, NOT production Re-ID)"
    if weights_path:
        try:
            sd = torch.load(weights_path, map_location=dev)
            sd = sd.get("state_dict", sd)
            model.load_state_dict(sd, strict=True)
            tag = f"reid_custom_weights ({weights_path})"
        except Exception as e:
            raise RuntimeError(f"--reid-model could not be loaded as TinyEmbedNet state_dict: {e}")
    return model, tag

def _preprocess(crops):
    import cv2
    arr = np.stack([cv2.resize(c, (64, 32)) for c in crops]).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(0, 3, 1, 2)

@torch.no_grad()
def embed_crops(model, crops, dev):
    """Real workload: crop preprocess (CPU) + embedding inference (device). Returns (emb, pre_ms, inf_ms)."""
    t0 = time.perf_counter()
    x = _preprocess(crops).to(dev)
    pre_ms = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    emb = model(x).cpu().numpy()
    if dev == "cuda":
        torch.cuda.synchronize()
    inf_ms = (time.perf_counter() - t0) * 1000
    return emb, pre_ms, inf_ms

def run(cfg):
    dev = resolve_device(cfg.get("device", "auto"))
    out = {"module": "reid", "config": {**cfg, "resolved_device": dev},
           "tests": {}, "errors": [], "status": S.FULL}
    with Monitor() as mon:
        try:
            model, tag = load_embedder(dev, (cfg.get("reid_model") or "").strip())
            out["config"]["embedder"] = tag
            rng = np.random.default_rng(cfg.get("seed", 0))
            gallery = [rng.integers(0, 255, (128, 64, 3), dtype=np.uint8) for _ in range(32)]
            queries = [rng.integers(0, 255, (128, 64, 3), dtype=np.uint8) for _ in range(8)]
            for _ in range(cfg.get("warmup", 3)):
                embed_crops(model, gallery[:8], dev)
            pre_ts, inf_ts = [], []
            g_emb, _, _ = embed_crops(model, gallery, dev)
            for _ in range(cfg.get("repeats", 10)):
                q_emb, pre_ms, inf_ms = embed_crops(model, queries, dev)
                pre_ts.append(pre_ms); inf_ts.append(inf_ms)
                _ = ((q_emb[:, None, :] - g_emb[None, :, :]) ** 2).sum(-1).argmin(1)
            d = summarize(inf_ts)
            d["preprocess_mean_ms"] = round(sum(pre_ts) / len(pre_ts), 3)
            d["gallery_size"] = len(gallery)
            d["note"] = "embedding inference + nearest-gallery match per query batch"
            out["tests"]["reid_embed_8query_vs_32gallery_ms"] = d
        except Exception as e:
            out["errors"].append(str(e))
        out["telemetry"] = mon.summary()
    return out
