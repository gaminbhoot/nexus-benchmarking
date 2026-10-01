"""Comms: MAVLink-style frame parse, telemetry throughput, simulated link latency. Stdlib only."""
import random
import struct
import time

from nexus_bench.monitor import Monitor
from nexus_bench.stats import summarize

HDR = struct.Struct("<BBHI")  # sysid, compid, msgid, seq

def _frame(seq, payload_len=64):
    return HDR.pack(1, 1, 76, seq) + bytes(payload_len)

def run(cfg):
    seed = cfg.get("seed", 0)
    n = max(100, cfg.get("repeats", 20) * 50)
    out = {"module": "comms", "config": cfg, "tests": {}, "errors": []}
    with Monitor() as mon:
        try:  # MAVLink-style parse throughput
            frames = [_frame(i) for i in range(n)]
            t0 = time.perf_counter(); parsed = 0
            for f in frames:
                sysid, compid, msgid, seq = HDR.unpack_from(f)
                if sysid == 1 and len(f) > HDR.size:
                    parsed += 1
            dt = (time.perf_counter() - t0) * 1000
            out["tests"]["mavlink_parse"] = {"msgs": parsed, "total_ms": round(dt, 2),
                                             "kmsgs_per_s": round(parsed / dt, 2) if dt else 0}
        except Exception as e:
            out["errors"].append(f"mavlink: {e}")
        try:  # telemetry encode/process pipeline
            ts = []
            for i in range(n):
                t0 = time.perf_counter()
                pkt = {"seq": i, "lat": 47.6 + i * 1e-6, "lon": 7.5, "alt": 100.0, "battery": 95.0}
                _ = str(pkt).encode()
                ts.append((time.perf_counter() - t0) * 1000)
            out["tests"]["telemetry_process_ms"] = summarize(ts)
        except Exception as e:
            out["errors"].append(f"telemetry: {e}")
        try:  # simulated network latency (seeded, repeatable)
            rng = random.Random(seed)
            lat = [5 + rng.expovariate(1 / 8) for _ in range(min(n, 2000))]
            d = summarize(lat); d["unit"] = "ms (simulated)"
            out["tests"]["sim_link_latency_ms"] = d
        except Exception as e:
            out["errors"].append(f"simlink: {e}")
        try:
            rate = cfg.get("repeats", 20)
            out["tests"]["ros2_note"] = "ROS2 messaging probed only if rclpy installed"
            try:
                import rclpy  # noqa
                out["tests"]["ros2_note"] = "rclpy present (pub/sub benchmark not run by default; enable explicitly)"
            except Exception:
                pass
        except Exception as e:
            out["errors"].append(f"ros2: {e}")
        out["telemetry"] = mon.summary()
    return out
