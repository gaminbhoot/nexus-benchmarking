"""Paced frame streams: virtual-time replay with bounded queues.

A producer thread acquires frames (video file, image dir, synthetic generator)
paced at target_fps in REAL time, stamping capture time. The consumer sees what
production sees: queue wait, drops on overrun, acquisition (read+decode) latency
measured independently in the producer. No virtual-time cheating.
"""
import os
import queue
import threading
import time

import numpy as np

_SENTINEL = object()

class PacedSource:
    def __init__(self, cfg, target_fps=30.0, queue_size=8, seed_key="video"):
        self.cfg = cfg
        self.target_fps = target_fps or 30.0
        self.q = queue.Queue(maxsize=queue_size)
        self.emitted = 0
        self.dropped_producer = 0
        self.acquire_ms = []
        vp = (cfg.get("video") or "").strip()
        dp = (cfg.get("image_dir") or "").strip()
        if vp and os.path.exists(vp):
            self.tag = f"video:{vp}"
        elif dp and os.path.isdir(dp):
            self.tag = f"image_dir:{dp}"
        else:
            self.tag = "synthetic_moving_squares (labelled)"
        self._stop = threading.Event()
        self._thread = None
        self._exhausted = threading.Event()

    def _frames(self):
        import cv2
        vp = (self.cfg.get("video") or "").strip()
        dp = (self.cfg.get("image_dir") or "").strip()
        if vp and os.path.exists(vp):
            cap = cv2.VideoCapture(vp)
            self.tag = f"video:{vp}"
            try:
                while not self._stop.is_set():
                    ok, f = cap.read()
                    if not ok:
                        break
                    yield cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            finally:
                cap.release()
            return
        if dp and os.path.isdir(dp):
            files = sorted(p for p in os.listdir(dp)
                           if p.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")))
            self.tag = f"image_dir:{dp} ({len(files)} files)"
            for f in files:
                if self._stop.is_set():
                    break
                img = cv2.imread(os.path.join(dp, f))
                if img is not None:
                    yield cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return
        from nexus_bench.vision_bench import _moving_squares
        n = self.cfg.get("agent_frames", 120)
        self.tag = "synthetic_moving_squares (labelled)"
        yield from _moving_squares(n, seed=self.cfg.get("seed", 0))

    def _produce(self):
        period = 1.0 / self.target_fps
        t0 = time.perf_counter()
        it = self._frames()
        while not self._stop.is_set():
            ta = time.perf_counter()
            try:
                frame = next(it)
            except StopIteration:
                break
            self.acquire_ms.append((time.perf_counter() - ta) * 1000)  # read+decode
            self.emitted += 1
            due = t0 + self.emitted * period
            now = time.perf_counter()
            if now < due:
                time.sleep(due - now)  # pace to target_fps: real-time arrival
            try:
                self.q.put_nowait((frame, time.time(), self.emitted))
            except queue.Full:
                self.dropped_producer += 1
        self._exhausted.set()

    def start(self):
        self._thread = threading.Thread(target=self._produce, daemon=True)
        self._thread.start()
        return self

    def get(self, timeout=5.0):
        try:
            return self.q.get(timeout=timeout)
        except queue.Empty:
            return None if self._exhausted.is_set() and self.q.empty() else "retry"

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    @property
    def real_pixels(self):
        return self.tag.startswith("video:") or self.tag.startswith("image_dir:")
