"""Purchase-gate engine: workload-specific criteria + headroom, fail-closed.

Only modules with status FULL are eligible. Anything else (PARTIAL, FALLBACK,
UNSUPPORTED, FAILED, INCONCLUSIVE, ABORTED, NOT_RUN) fails its checks with an
explicit reason — a gate can end INCONCLUSIVE but never PASS on missing evidence.
"""
from nexus_bench import statuses as S

DEFAULT_GATE = {
    "req_fps": 15,
    "workloads": {
        # Each workload names the module + test selection rule and thresholds.
        "uav": {"module": "pipeline", "min_fps": 15, "max_p95_ms": 66.7,
                "max_p99_ms": 120.0, "max_miss_pct": 5.0, "max_drop_pct": 5.0},
        "rover": {"module": "pipeline", "min_fps": 15, "max_p95_ms": 66.7,
                  "max_p99_ms": 120.0, "max_miss_pct": 5.0, "max_drop_pct": 5.0},
        "dual": {"module": "integrated", "min_fps_each": 10.0, "max_drop": 0,
                 "min_ticks": 10},
    },
    "headroom": {"min_fps_margin": 1.2, "max_vram_occupied_pct": 85.0,
                 "max_miss_pct": 2.0},
    "resources": {"forbid_throttle_likely": True, "forbid_oom": True},
}

def _check(name, cond, detail, checks):
    checks.append({"check": name, "pass": bool(cond), "detail": detail})
    return bool(cond)

def _eligible(res):
    return isinstance(res, dict) and res.get("status") in S.GATE_ELIGIBLE

def evaluate(results, gate=None):
    gate = gate or DEFAULT_GATE
    req = gate.get("req_fps", 15)
    checks, overall = [], True
    w = gate.get("workloads", {})

    for wname in ("uav", "rover"):
        crit = w.get(wname, {})
        res = results.get("pipeline", {})
        ok = _eligible(res)
        _check(f"{wname}.evidence_full", ok,
               f"pipeline status={res.get('status')}", checks)
        if not ok:
            overall = False
            continue
        t = (res.get("tests") or {}).get("end_to_end_ms", {})
        fps = t.get("throughput_fps") or 0
        overall &= _check(f"{wname}.fps>={crit.get('min_fps', req)}", fps >= crit.get("min_fps", req),
                          f"{fps} fps", checks)
        for key, lim in (("p95_ms", "max_p95_ms"), ("p99_ms", "max_p99_ms")):
            v = t.get(key)
            c = _check(f"{wname}.{key}<={crit.get(lim, 1e9)}",
                       v is not None and v <= crit.get(lim, 1e9), f"{v}ms", checks)
            overall &= c
        for key, lim in (("deadline_miss_pct", "max_miss_pct"), ("drop_pct", "max_drop_pct")):
            v = t.get(key, 0)
            overall &= _check(f"{wname}.{key}<={crit.get(lim, 100)}", v <= crit.get(lim, 100),
                              f"{v}%", checks)

    dc = w.get("dual", {})
    res = results.get("integrated", {})
    ok = _eligible(res)
    _check("dual.evidence_full", ok, f"integrated status={res.get('status')}", checks)
    if not ok:
        overall = False
    else:
        per = (res.get("tests") or {}).get("per_agent", {})
        for agent in ("uav", "rover"):
            fps = (per.get(agent) or {}).get("fps", 0)
            overall &= _check(f"dual.{agent}.fps>={dc.get('min_fps_each', 10)}",
                              fps >= dc.get("min_fps_each", 10),
                              f"{fps} fps", checks)
            drops = (per.get(agent) or {}).get("drops", 0)
            overall &= _check(f"dual.{agent}.drops<={dc.get('max_drop', 0)}",
                              drops is not None and drops <= dc.get("max_drop", 0),
                              f"{drops}", checks)
        ticks = ((res.get("tests") or {}).get("fusion") or {}).get("ticks", 0)
        overall &= _check(f"dual.fusion_ticks>={dc.get('min_ticks', 10)}",
                          ticks >= dc.get("min_ticks", 10), f"{ticks}", checks)

    # resources: VRAM occupancy, throttling, OOM — veto-grade.
    mem = results.get("memory", {})
    if _eligible(mem):
        occ = None
        t = (mem.get("tests") or {})
        if isinstance(t.get("vram_total_mb"), (int, float)) and isinstance(t.get("vram_peak_reserved_mb"), (int, float)):
            occ = 100 * t["vram_peak_reserved_mb"] / t["vram_total_mb"]
        lim = gate.get("headroom", {}).get("max_vram_occupied_pct", 85)
        overall &= _check("resources.vram_headroom", occ is not None and occ <= lim,
                          f"peak reserved {occ if occ is None else round(occ,1)}% vs {lim}%", checks)
    else:
        _check("resources.vram_headroom", False,
               f"memory status={mem.get('status')} (no VRAM evidence)", checks)
        overall = False
    th = ((results.get("thermal", {}) or {}).get("tests") or {}).get("throttling")
    if gate.get("resources", {}).get("forbid_throttle_likely", True):
        overall &= _check("resources.no_throttle", th != "likely", f"thermal={th}", checks)
    oom = any("out of memory" in str(e).lower() or str(e).startswith("VRAM OOM")
              for r in results.values() if isinstance(r, dict) for e in r.get("errors", []))
    if gate.get("resources", {}).get("forbid_oom", True):
        overall &= _check("resources.no_oom", not oom, "OOM observed" if oom else "clean", checks)

    # accuracy preservation gate: any measured delta beyond tolerance vetoes.
    acc = ((results.get("accuracy", {}) or {}).get("tests") or {}).get("agreement", {})
    tol = gate.get("accuracy", {}).get("max_map_drop", 0.05)
    if acc:
        drop = acc.get("map50_drop_vs_fp32")
        if drop is not None:
            overall &= _check("accuracy.map_drop", drop <= tol, f"drop={drop}", checks)
    else:
        overall &= _check("accuracy.measured", False, "no accuracy evidence (INCONCLUSIVE, not pass)", checks)

    states = [r.get("status") for r in results.values()
              if isinstance(r, dict) and not str(r.get("module", "")).startswith("_")]
    hard_fail = any(s in (S.FAILED, S.ABORTED) for s in states if s)
    if not overall or hard_fail:
        verdict = S.FAIL if (hard_fail or _decisive_fail(checks)) else S.INCONCLUSIVE
    else:
        verdict = S.PASS_WITH_HEADROOM if _has_headroom(results, gate) else S.PASS
    return {"verdict": verdict, "req_fps": req, "checks": checks,
            "states": {k: v for k, v in (S.gate_counts(results) or {}).items()}}

def _decisive_fail(checks):
    return any(not c["pass"] and ("evidence_full" in c["check"] or "no_oom" in c["check"]
                                  or "no_throttle" in c["check"]) for c in checks)

def _has_headroom(results, gate):
    try:
        t = (results.get("pipeline", {}).get("tests") or {}).get("end_to_end_ms", {})
        fps = t.get("throughput_fps") or 0
        margin = gate.get("headroom", {}).get("min_fps_margin", 1.2)
        miss = max(t.get("deadline_miss_pct", 0), t.get("drop_pct", 0))
        return fps >= gate.get("req_fps", 15) * margin and miss <= gate.get("headroom", {}).get("max_miss_pct", 2.0)
    except Exception:
        return False
