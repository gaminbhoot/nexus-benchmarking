"""Concurrency model for multi-agent inference — explicit and crash-free.

Observed (faulthandler trace, torch THPVariable_float): two threads inside
Ultralytics predict concurrently segfault this stack. Concurrent same-process
forwards are therefore NEVER run unguarded.

Modes (cfg concurrency, reported verbatim in every result):
  separate-contexts  one YOLO object per agent (deployed VRAM cost is real),
                     forwards serialized by a shared lock. Safe everywhere.
  serialized          a single shared model behind a lock (central scheduler).
  separate-lockfree   separate objects, no lock — CUDA ONLY. Refuses on
                     mps/cpu with an explicit error instead of risking a crash.
"""
import threading

from nexus_bench.yolo_util import load_weights, precision_kwargs


class ModelContexts:
    def __init__(self, model_path, want_fp16, mode="separate-contexts", agents=("uav", "rover")):
        self.mode = mode if mode in ("separate-contexts", "serialized",
                                     "separate-lockfree") else "separate-contexts"
        self.prec_rec = None
        self.pk = {}
        self.lock = threading.Lock()
        self.models = {}
        if model_path:
            import os
            if os.path.exists(model_path):
                from ultralytics import YOLO
                if self.mode == "serialized":
                    self.models["shared"] = YOLO(model_path)
                else:
                    for a in agents:
                        self.models[a] = YOLO(model_path)
        _, self.prec_rec = load_weights(model_path or "", want_fp16=want_fp16)
        if want_fp16 and model_path and self.models:
            self.pk = precision_kwargs(True)

    def configure(self, dev):
        if self.mode == "separate-lockfree" and dev != "cuda":
            raise RuntimeError(
                f"concurrency=separate-lockfree refused on {dev}: concurrent same-process "
                f"Ultralytics forwards segfault this stack (observed). Use "
                f"separate-contexts (locked) or serialized.")

    def predict(self, agent, small, imgsz, dev):
        """One batched forward under this model's concurrency contract."""
        yolo = self.models["shared"] if self.mode == "serialized" else self.models[agent]
        if self.mode == "separate-lockfree":
            return yolo.predict(small, imgsz=imgsz, device=dev, verbose=False, **self.pk)[0]
        with self.lock:
            return yolo.predict(small, imgsz=imgsz, device=dev, verbose=False, **self.pk)[0]
