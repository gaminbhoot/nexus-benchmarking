"""Independent background telemetry monitor (thread). Never fails the benchmark."""
import subprocess
import threading
import time

class Monitor:
    def __init__(self, interval_s=0.5):
        self.interval_s = interval_s
        self.samples = []
        self._stop = threading.Event()
        self._thread = None

    def _sample_once(self):
        s = {"t": time.time()}
        try:
            import psutil
            p = psutil.Process()
            s["cpu_pct"] = psutil.cpu_percent(interval=None)
            s["ram_pct"] = psutil.virtual_memory().percent
            s["proc_rss_mb"] = round(p.memory_info().rss / 1e6, 1)
        except Exception:
            pass
        try:
            import torch
            if torch.cuda.is_available():
                s["vram_used_mb"] = round(torch.cuda.memory_allocated() / 1e6, 1)
                try:
                    total = torch.cuda.get_device_properties(0).total_memory / 1e6
                    s["vram_pct"] = round(100 * torch.cuda.memory_allocated() / total, 1)
                except Exception:
                    pass
        except Exception:
            pass
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,temperature.gpu,power.draw,memory.used,memory.total",
                                  "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
            if out.returncode == 0 and out.stdout.strip():
                u, temp, pwr, mu, mt = [x.strip() for x in out.stdout.strip().splitlines()[0].split(",")]
                s["gpu_util_pct"] = float(u)
                s["gpu_temp_c"] = float(temp)
                s["gpu_power_w"] = float(pwr)
                s["vram_pct"] = round(100 * float(mu) / float(mt), 1)
        except Exception:
            pass
        return s

    def _loop(self):
        import psutil
        psutil.cpu_percent(interval=None)
        while not self._stop.is_set():
            self.samples.append(self._sample_once())
            time.sleep(self.interval_s)

    def __enter__(self):
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *a):
        self.stop()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def summary(self):
        out = {"n": len(self.samples)}
        for k in ("cpu_pct", "ram_pct", "proc_rss_mb", "gpu_util_pct",
                  "gpu_temp_c", "gpu_power_w", "vram_pct"):
            vals = [s[k] for s in self.samples if k in s]
            if vals:
                out[k] = {"mean": round(sum(vals) / len(vals), 2), "max": round(max(vals), 2)}
        return out
