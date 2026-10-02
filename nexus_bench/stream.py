"""Paced frame streams with conservation accounting and latency taxonomy.

Arrival model: camera/input clock -> acquisition/decode (producer thread) ->
bounded queue -> processing (consumer). Slow processing NEVER slows the input
clock: the producer keeps pacing and counts producer_drops when the queue is
full. Every frame is counted exactly once:

  generated = delivered + producer_drops + queue_drops
              + processing_failures + unfinished

Drop definition (used identically in every module):
  drop_pct = 100 * (generated - delivered) / generated

Per-frame timestamps: t_acquire_start -> t_decode_done (= t_capture) ->
t_enqueue -> t_dequeue -> stages -> t_output.
  acquisition_latency = t_decode_done - t_acquire_start
  queue_wait          = t_dequeue - t_enqueue
  processing_latency  = t_output - t_dequeue
  total input->output = t_output - t_capture
"""
import os
import queue
import threading
import time

import numpy as np

_SENTINEL = object()

class FrameAccount:
    """Conservation counters for one stream. All ints, all exact."""
    def __init__(self):
        self.generated = 0
        self.acquired = 0
        self.enqueued = 0
        self.dequeued = 0
        self.processed = 0
        self.delivered = 0
        self.producer_drops = 0
        self.queue_drops = 0
        self.processing_failures = 0
        self.unfinished = 0

    def finalize(self, remaining_in_queue):
        self.unfinished = (self.dequeued - self.processed - self.processing_failures
                           + remaining_in_queue)

    def as_dict(self):
        return {k: getattr(self, k) for k in
                ("generated", "acquired", "enqueued", "dequeued", "processed",
                 "delivered", "producer_drops", "queue_drops",
                 "processing_failures", "unfinished")}

    def check_conservation(self):
        d = self.as_dict()
        lhs = d["generated"]
        rhs = (d["delivered"] + d["producer_drops"] + d["queue_drops"]
               + d["processing_failures"] + d["unfinished"])
        return lhs == rhs, {"generated": lhs, "accounted": rhs}

def drop_pct(generated, delivered):
    """THE drop definition. Same function, every module."""
    if not generated:
        return 0.0
    return round(100.0 * (generated - delivered) / generated, 2)

class PacedSource:
    def __init__(self, cfg, target_fps=30.0, queue_size=8, source_keys=("video",),
                 loop=False):
        """source_keys: ordered fallback chain, e.g. ("uav_video", "video").
        First existing source wins; the choice is recorded, never silent.
        loop=True: short clips rewind at EOF until the consumer stops, so a
        20 s clip covers a 600 s run. replay_count + source_duration_s prove it."""
        self.cfg = cfg
        self.target_fps = target_fps or 30.0
        self.q = queue.Queue(maxsize=queue_size)
        self.acct = FrameAccount()
        self.acquire_ms = []
        self.source_keys = tuple(source_keys)
        self.source_used = None   # which key won, e.g. "uav_video"
        self.source_kind = "synthetic"
        self.source_detail = ""
        self.loop = loop
        self.replay_count = 0
        self.source_duration_s = None  # measured span of one pass (video only)
        self._stop = threading.Event()
        self._thread = None
        self._exhausted = threading.Event()
        self._resolve()

    def _resolve(self):
        import cv2  # noqa (ensures decode backend present)
        for key in self.source_keys:
            vp = (self.cfg.get(key) or "").strip()
            if vp and os.path.exists(vp):
                self.source_used, self.source_kind = key, "video"
                self.source_detail = vp
                return
        dp = (self.cfg.get("image_dir") or "").strip()
        if dp and os.path.isdir(dp):
            files = sorted(p for p in os.listdir(dp)
                           if p.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")))
            if files:
                self.source_used, self.source_kind = "image_dir", "image_dir"
                self.source_detail = f"{dp} ({len(files)} files)"
                return
        tried = [f"{k}={self.cfg.get(k) or ''!r}" for k in self.source_keys]
        self.source_used, self.source_kind = None, "synthetic"
        self.source_detail = (f"synthetic_moving_squares (labelled fallback; tried {tried}"
                              + (" + image_dir" if not dp else "") + ")")

    @property
    def real_pixels(self):
        return self.source_kind in ("video", "image_dir")

    @property
    def tag(self):
        return self.source_detail

    def _frames(self):
        import cv2
        if self.source_kind == "video":
            while True:
                cap = cv2.VideoCapture(self.source_detail)
                try:
                    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                    fps = cap.get(cv2.CAP_PROP_FPS) or 0
                    if n_frames and fps:
                        self.source_duration_s = round(n_frames / fps, 2)
                    while not self._stop.is_set():
                        ok, f = cap.read()
                        if not ok:
                            break
                        yield cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                finally:
                    cap.release()
                if not self.loop or self._stop.is_set():
                    return
                self.replay_count += 1  # rewind: short clip, long qualification
            return
        if self.source_kind == "image_dir":
            base = self.source_detail.split(" (")[0]
            while True:
                files = sorted(f for f in os.listdir(base)
                               if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")))
                for f in files:
                    if self._stop.is_set():
                        break
                    img = cv2.imread(os.path.join(base, f))
                    if img is not None:
                        yield cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                if not self.loop or self._stop.is_set():
                    return
                self.replay_count += 1
            return
        from nexus_bench.vision_bench import _moving_squares
        if self.loop:
            seed = self.cfg.get("seed", 0)
            while not self._stop.is_set():
                yield from _moving_squares(300, seed=seed)
                self.replay_count += 1
                seed += 1
            return
        yield from _moving_squares(self.cfg.get("agent_frames", 600),
                                   seed=self.cfg.get("seed", 0))

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
            decode_ms = (time.perf_counter() - ta) * 1000
            self.acct.generated += 1
            self.acct.acquired += 1
            self.acquire_ms.append(decode_ms)
            due = t0 + self.acct.generated * period
            now = time.perf_counter()
            if now < due:
                time.sleep(due - now)  # real-time arrival, independent of processing
            meta = {"seq": self.acct.generated,
                    "t_acquire_start": ta,
                    "t_decode_done": ta + decode_ms / 1000.0,
                    "t_capture": time.time(),
                    "t_enqueue": time.time()}
            try:
                self.q.put_nowait((frame, meta))
                self.acct.enqueued += 1
            except queue.Full:
                self.acct.producer_drops += 1
        self._exhausted.set()

    def start(self):
        self._thread = threading.Thread(target=self._produce, daemon=True)
        self._thread.start()
        return self

    def get(self, timeout=5.0):
        """Returns (frame, meta) with meta['t_dequeue'] stamped, None when
        exhausted+empty, or 'retry' when still producing."""
        try:
            frame, meta = self.q.get(timeout=timeout)
            meta["t_dequeue"] = time.time()
            self.acct.dequeued += 1
            return frame, meta
        except queue.Empty:
            if self._exhausted.is_set():
                return None
            return "retry"

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        self.acct.finalize(self.q.qsize())
