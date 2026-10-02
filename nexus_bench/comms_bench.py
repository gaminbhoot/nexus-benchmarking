"""Comms: REAL local measurements + labelled simulation.

Real: MAVLink v2 framing implemented here (header + CRC-16/MCRF4XX incl. message
CRC-extra, e.g. HEARTBEAT msgid 0 / extra 50): serialize, parse, CRC-validate,
corrupted-frame rejection rate, routing throughput. Plus loopback UDP RTT/jitter/
loss over real sockets (localhost = stack cost; point it at the vehicle for link
numbers). The old RNG latency draw is kept ONLY as `sim_link_model` and never
presented as a measurement.
"""
import socket
import struct
import threading
import time

from nexus_bench import statuses as S
from nexus_bench.monitor import Monitor
from nexus_bench.stats import summarize

# --- MAVLink v2 (real framing, no pymavlink dependency) ---
_STX = 0xFD

def _crc16_mcrf4xx(data, crc=0xFFFF):
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = (crc >> 1) ^ 0x8408 if crc & 1 else crc >> 1
    return crc & 0xFFFF

_CRC_EXTRA = {0: 50}  # HEARTBEAT; extend per common.xml when adding messages

def mav_pack(msgid, payload, seq, sysid=1, compid=1):
    header = bytes([len(payload), 0, 0, seq & 0xFF, sysid, compid]) + struct.pack("<I", msgid)[:3]
    crc = _crc16_mcrf4xx(payload, _crc16_mcrf4xx(header))
    crc = _crc16_mcrf4xx(bytes([_CRC_EXTRA.get(msgid, 0)]), crc)
    return bytes([_STX]) + header + payload + struct.pack("<H", crc)

def mav_parse(buf):
    """Returns (msgid, payload, seq, consumed) or raises ValueError. Real CRC check."""
    if len(buf) < 12 or buf[0] != _STX:
        raise ValueError("bad sync/length")
    plen = buf[1]
    total = 12 + plen
    if len(buf) < total:
        raise ValueError("truncated")
    msgid = buf[7] | (buf[8] << 8) | (buf[9] << 16)
    payload = bytes(buf[10:10 + plen])
    rx_crc = struct.unpack("<H", bytes(buf[10 + plen:total]))[0]
    header = bytes(buf[1:10])
    crc = _crc16_mcrf4xx(payload, _crc16_mcrf4xx(header))
    crc = _crc16_mcrf4xx(bytes([_CRC_EXTRA.get(msgid, 0)]), crc)
    if crc != rx_crc:
        raise ValueError("CRC mismatch")
    return msgid, payload, buf[4], total

def _heartbeat(seq):
    # Wire order per common.xml: type, autopilot, base_mode, custom_mode, system_status, mavlink_version.
    payload = struct.pack("<BBBI BB", 2, 0, 0, 0, 4, 3)
    return mav_pack(0, payload, seq)

def _loopback_rtt(n=200, size=64, host="127.0.0.1", port=0):
    """Real UDP echo RTT over actual sockets. Localhost measures stack cost."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    srv.bind((host, 0))
    srv.settimeout(2.0)
    rport = srv.getsockname()[1]
    stop = threading.Event()

    def echo():
        while not stop.is_set():
            try:
                data, addr = srv.recvfrom(65535)
                srv.sendto(data, addr)
            except socket.timeout:
                pass
            except OSError:
                break

    t = threading.Thread(target=echo, daemon=True)
    t.start()
    cli = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    cli.settimeout(2.0)
    rtts, lost = [], 0
    try:
        for i in range(n):
            msg = bytes([i & 0xFF]) + b"x" * (size - 1)
            t0 = time.perf_counter()
            cli.sendto(msg, (host, rport))
            try:
                data, _ = cli.recvfrom(65535)
                assert data == msg
                rtts.append((time.perf_counter() - t0) * 1000)
            except socket.timeout:
                lost += 1
    finally:
        stop.set(); t.join(timeout=2)
        cli.close(); srv.close()
    return rtts, lost

def run(cfg):
    n = max(500, cfg.get("repeats", 20) * 50)
    out = {"module": "comms", "config": cfg, "tests": {}, "errors": [], "status": S.FULL}
    with Monitor() as mon:
        try:  # real MAVLink serialize throughput
            t0 = time.perf_counter()
            frames = [_heartbeat(i) for i in range(n)]
            dt = (time.perf_counter() - t0) * 1000
            out["tests"]["mavlink_serialize"] = {"msgs": n, "total_ms": round(dt, 2),
                                                 "kmsgs_per_s": round(n / dt, 2) if dt else 0,
                                                 "bytes_per_msg": len(frames[0])}
        except Exception as e:
            out["errors"].append(f"mavlink_serialize: {e}")
        try:  # real parse + CRC validation + corruption rejection
            t0 = time.perf_counter()
            parsed = sum(1 for f in frames if mav_parse(bytearray(f))[0] == 0)
            dt = (time.perf_counter() - t0) * 1000
            bad = bytearray(frames[0]); bad[12] ^= 0xFF
            rejected = False
            try:
                mav_parse(bad)
            except ValueError:
                rejected = True
            out["tests"]["mavlink_parse_crc"] = {"msgs": parsed, "total_ms": round(dt, 2),
                                                 "kmsgs_per_s": round(parsed / dt, 2) if dt else 0,
                                                 "corruption_rejected": rejected}
        except Exception as e:
            out["errors"].append(f"mavlink_parse: {e}")
        try:  # real loopback RTT
            rtts, lost = _loopback_rtt(n=min(n, 500))
            d = summarize(rtts) if rtts else {"n": 0}
            d["unit"] = "ms RTT, UDP loopback (stack cost, not link latency)"
            d["lost"] = lost
            out["tests"]["loopback_udp_rtt_ms"] = d
        except Exception as e:
            out["errors"].append(f"loopback: {e}")
        try:  # telemetry encode pipeline
            ts = []
            for i in range(min(n, 2000)):
                t0 = time.perf_counter()
                pkt = {"seq": i, "lat": 47.6 + i * 1e-6, "lon": 7.5, "alt": 100.0, "battery": 95.0}
                _ = str(pkt).encode()
                ts.append((time.perf_counter() - t0) * 1000)
            out["tests"]["telemetry_process_ms"] = summarize(ts)
        except Exception as e:
            out["errors"].append(f"telemetry: {e}")
        try:
            import random
            rng = random.Random(cfg.get("seed", 0))
            sim = [5 + rng.expovariate(1 / 8) for _ in range(min(n, 2000))]
            d = summarize(sim)
            d["unit"] = "ms SIMULATED (model only — not a measurement)"
            out["tests"]["sim_link_model_ms"] = d
        except Exception as e:
            out["errors"].append(f"simlink: {e}")
        try:
            import rclpy  # noqa
            out["tests"]["ros2"] = "rclpy present (pub/sub benchmark not run by default)"
        except Exception:
            out["tests"]["ros2"] = "unavailable (rclpy not installed)"
        out["telemetry"] = mon.summary()
    return out
