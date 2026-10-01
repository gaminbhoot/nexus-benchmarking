"""Safety limits + graceful abort. Never modifies firmware/clocks."""
import time
from dataclasses import dataclass, field

@dataclass
class Limits:
    max_temp_c: float = 95.0
    max_ram_pct: float = 90.0
    max_duration_s: float = 600.0
    max_vram_pct: float = 92.0
    poll_s: float = 1.0

@dataclass
class Guard:
    limits: Limits = field(default_factory=Limits)
    reason: str = ""
    _t0: float = field(default_factory=time.time, init=False)

    def ok(self, telemetry=None):
        t = telemetry or {}
        if time.time() - self._t0 > self.limits.max_duration_s:
            self.reason = "max_duration exceeded"
            return False
        temp = t.get("cpu_temp_c") or t.get("gpu_temp_c")
        if temp and temp >= self.limits.max_temp_c:
            self.reason = f"sustained high temp {temp:.1f}C"
            return False
        if (t.get("ram_pct") or 0) >= self.limits.max_ram_pct:
            self.reason = f"RAM {t.get('ram_pct'):.0f}% over limit"
            return False
        if (t.get("vram_pct") or 0) >= self.limits.max_vram_pct:
            self.reason = f"VRAM {t.get('vram_pct'):.0f}% over limit"
            return False
        return True
