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

def _check(name, cond, detail, checks, kind="measured"):
    """kind: 'measured' (a real number violated a limit -> decisive FAIL) or
    'missing' (no evidence -> INCONCLUSIVE, never a measured failure)."""
    checks.append({"check": name, "pass": bool(cond), "detail": detail, "kind": kind})
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
        sus = results.get("sustained", {})
        # Prefer sustained per-agent evidence (long-run truth) over the short
        # pipeline probe; the pipeline is the fallback, not a duplicate vote.
        sus_per = ((sus.get("tests") or {}).get("per_agent") or {}).get(wname) \
            if _eligible(sus) else None
        if sus_per is not None:
            fps = sus_per.get("steady_fps", 0) or 0
            _check(f"{wname}.evidence_sustained", True, "sustained FULL per-agent", checks,
                   kind="measured")
            overall &= _check(f"{wname}.fps>={crit.get('min_fps', req)}", fps >= crit.get("min_fps", req),
                              f"sustained steady {fps} fps", checks)
            p95 = sus_per.get("max_p95_ms")
            overall &= _check(f"{wname}.max_p95_ms<={crit.get('max_p95_ms', 1e9)}",
                              p95 is not None and p95 <= crit.get("max_p95_ms", 1e9),
                              f"sustained worst-window p95 {p95}ms", checks)
            overall &= _check(f"{wname}.max_drop_pct<={crit.get('max_drop_pct', 100)}",
                              (sus_per.get("max_window_drop_pct", 0) or 0) <= crit.get("max_drop_pct", 100),
                              f"worst-window drop {sus_per.get('max_window_drop_pct')}%", checks)
            continue
        ok = _eligible(res)
        _check(f"{wname}.evidence_full", ok,
               f"pipeline status={res.get('status')}", checks, kind="missing")
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
    _check("dual.evidence_full", ok, f"integrated status={res.get('status')}", checks, kind="missing")
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

    # --- sustained qualification (preferred evidence over burst runs) ---
    sus = results.get("sustained", {})
    sc = gate.get("sustained", {})
    if _eligible(sus):
        per = ((sus.get("tests") or {}).get("per_agent") or {})
        for agent in ("uav", "rover"):
            a = per.get(agent) or {}
            fps = a.get("steady_fps", a.get("final_fps", 0)) or 0
            overall &= _check(f"sustained.{agent}.steady_fps>={sc.get('min_fps_each', 10)}",
                              fps >= sc.get("min_fps_each", 10), f"{fps} fps", checks)
            # Worst windows decide, never the median alone.
            overall &= _check(f"sustained.{agent}.min_window_fps>={sc.get('min_fps_each', 10)}",
                              (a.get("min_window_fps", 0) or 0) >= sc.get("min_fps_each", 10),
                              f"worst window {a.get('min_window_fps')} fps", checks)
            overall &= _check(f"sustained.{agent}.final_window_fps>={sc.get('min_fps_each', 10)}",
                              (a.get("final_window_fps", 0) or 0) >= sc.get("min_fps_each", 10),
                              f"final window {a.get('final_window_fps')} fps", checks)
            overall &= _check(f"sustained.{agent}.max_window_p95<={sc.get('max_p95_ms', 1e9)}",
                              a.get("max_p95_ms") is not None and
                              a.get("max_p95_ms") <= sc.get("max_p95_ms", 1e9),
                              f"worst window p95 {a.get('max_p95_ms')}ms", checks)
            if sc.get("max_p99_ms") is not None:
                overall &= _check(f"sustained.{agent}.max_window_p99<={sc.get('max_p99_ms')}",
                                  a.get("max_window_p99_ms") is not None and
                                  a.get("max_window_p99_ms") <= sc.get("max_p99_ms"),
                                  f"worst window p99 {a.get('max_window_p99_ms')}ms", checks)
            overall &= _check(f"sustained.{agent}.drop_pct<={sc.get('max_drop_pct', 5)}",
                              (a.get("drop_pct", 0) or 0) <= sc.get("max_drop_pct", 5),
                              f"{a.get('drop_pct')}%", checks)
            overall &= _check(f"sustained.{agent}.max_window_drop<={sc.get('max_drop_pct', 5)}",
                              (a.get("max_window_drop_pct", 0) or 0) <= sc.get("max_drop_pct", 5),
                              f"worst window {a.get('max_window_drop_pct')}%", checks)
            overall &= _check(f"sustained.{agent}.max_window_miss<={sc.get('max_miss_pct', 5)}",
                              (a.get("max_window_miss_pct", 0) or 0) <= sc.get("max_miss_pct", 5),
                              f"worst window {a.get('max_window_miss_pct')}%", checks)
            overall &= _check(f"sustained.{agent}.degradation<={sc.get('max_degradation_pct', 15)}",
                              (a.get("degradation_pct", 0) or 0) <= sc.get("max_degradation_pct", 15),
                              f"{a.get('degradation_pct')}%", checks)
            overall &= _check(f"sustained.{agent}.no_workload_oom",
                              not a.get("oom_events"), f"oom={a.get('oom_events')}", checks)
            # Duration proof: actual execution, not configured intent.
            overall &= _check(f"sustained.{agent}.actual_duration",
                              a.get("duration_complete") is True,
                              f"actual {a.get('actual_duration_s')}s vs requested "
                              f"{a.get('requested_duration_s')}s", checks)
            overall &= _check(f"sustained.{agent}.conservation",
                              a.get("conservation_ok") is True,
                              "frame accounting conserved" if a.get("conservation_ok") is True
                              else "frame accounting FAILED — drops cannot be trusted", checks)
        dur = (sus.get("config") or {}).get("duration_s", 0)
        min_dur = sc.get("min_duration_s", 600)
        overall &= _check(f"sustained.duration>={min_dur}",
                          dur >= min_dur, f"{dur}s", checks)
        mg = ((sus.get("tests") or {}).get("memory_growth") or {})
        if mg.get("leak_suspected"):
            overall &= _check("sustained.no_leak", False,
                              f"RSS growth {mg.get('growth')}MB (leak suspected)", checks)
        else:
            _check("sustained.no_leak", True, f"RSS growth {mg.get('growth')}MB", checks)
    elif "sustained" in gate.get("require_modules", ["sustained"]):
        overall &= _check("sustained.evidence_full", False,
                          f"sustained status={sus.get('status')} (no long-run evidence)", checks, kind="missing")
    mem = results.get("memory", {})
    # Prefer the sustained workload's peak (deployment truth) over the ladder's.
    sus_evo = ((sus.get("tests") or {}).get("resource_evolution") or {})
    occ, occ_src = None, None
    if isinstance(sus_evo.get("vram_pct"), dict) and sus_evo["vram_pct"].get("max") is not None:
        occ, occ_src = sus_evo["vram_pct"]["max"], "sustained peak vram_pct"
    if occ is None and _eligible(mem):
        t = (mem.get("tests") or {})
        if isinstance(t.get("vram_total_mb"), (int, float)) and isinstance(t.get("vram_peak_reserved_mb"), (int, float)):
            occ = 100 * t["vram_peak_reserved_mb"] / t["vram_total_mb"]
            occ_src = "memory ladder peak reserved"
    lim = gate.get("headroom", {}).get("max_vram_occupied_pct", 85)
    if occ is not None:
        overall &= _check("resources.vram_headroom", occ <= lim,
                          f"{occ_src}: {round(occ,1)}% vs {lim}%", checks)
    else:
        _check("resources.vram_headroom", False,
               f"memory status={mem.get('status')}, sustained status={sus.get('status')} (no VRAM evidence)", checks,
               kind="missing")
        overall = False
    th = ((results.get("thermal", {}) or {}).get("tests") or {}).get("throttling")
    if th == "none_detected":
        _check("resources.no_throttle", True, "thermal=none_detected", checks)
    elif th == "likely":
        overall &= _check("resources.no_throttle", False, "thermal=likely — throttling evidence", checks)
    else:
        # unknown / slowdown_cause_unknown / missing: fail CLOSED to INCONCLUSIVE,
        # never silently equivalent to "not throttling".
        overall &= _check("resources.no_throttle", False,
                          f"thermal={th} — no certifiable evidence (INCONCLUSIVE, not pass)", checks,
                          kind="missing")
    oom = any(("out of memory" in str(e).lower() or "oom" in str(e).lower())
              and "capacity_probe_oom" not in str(e).lower()
              for r in results.values() if isinstance(r, dict) for e in r.get("errors", []))
    if gate.get("resources", {}).get("forbid_oom", True):
        overall &= _check("resources.no_oom", not oom, "OOM observed" if oom else "clean", checks)

    # hardware identity: the profile may pin the expected machine.
    exp = gate.get("expected_hardware", {}) or {}
    prof = results.get("_profile", {}) or {}
    if exp:
        gpu_name = ((prof.get("gpu") or {}).get("name") or "").lower()
        want_gpu = str(exp.get("gpu_contains", "") or "").lower()
        match = (not want_gpu) or (want_gpu in gpu_name)
        overall &= _check("hardware.gpu_match", match,
                          f"expected ~{exp.get('gpu_contains')!r}, tested {gpu_name or 'unknown'}"
                          + ("" if match else " — TARGET HARDWARE DOES NOT MATCH TEST HARDWARE"),
                          checks)
        want_vram = exp.get("min_vram_gb")
        if want_vram:
            total = None
            try:
                total = float(str((prof.get("gpu") or {}).get("vram_total", "")).split()[0])
                total_gb = total / 1024.0
            except Exception:
                total_gb = None
            overall &= _check("hardware.vram_class", total_gb is not None and total_gb >= want_vram - 0.5,
                              f"tested {total_gb}GB vs expected >={want_vram}GB", checks)

    # provenance: dirty tree policy + power coverage honesty.
    prov = results.get("_provenance", {}) or {}
    git = (prov.get("git") or {})
    is_release = str(prov.get("distribution", "")).startswith("release")
    if gate.get("provenance", {}).get("forbid_dirty", False) and not is_release:
        clean = git.get("dirty") is False
        overall &= _check("provenance.clean_tree", clean,
                          "UNRELEASED / DIRTY SOURCE — exact inputs not reproducible"
                          if not clean else "clean tree", checks, kind="missing")
    else:
        _check("provenance.tree", True,
               "UNRELEASED / DIRTY SOURCE" if git.get("dirty") else "clean tree", checks)

    # accuracy preservation gate: candidate-vs-baseline mAP comparison required.
    acc_tests = ((results.get("accuracy", {}) or {}).get("tests") or {})
    comp = acc_tests.get("map_comparison", {})
    tol = gate.get("accuracy", {}).get("max_map_drop", 0.05)
    cand = comp.get("candidate") if isinstance(comp, dict) else None
    if isinstance(comp.get("map50_drop_abs"), (int, float)):
        overall &= _check("accuracy.map50_drop_abs", comp["map50_drop_abs"] <= tol,
                          f"candidate={cand} drop={comp['map50_drop_abs']} vs tol={tol}", checks)
    else:
        overall &= _check("accuracy.measured", False,
                          "no candidate-vs-baseline mAP comparison (INCONCLUSIVE, not pass)", checks,
                          kind="missing")

    # deployment identity: accuracy, inference, sustained must qualify the SAME
    # model + backend + precision + resolution + batch. Mixed identities fail.
    def _dep_id(res):
        c = (res.get("config") or {})
        fp = ((results.get("_provenance") or {}).get("fingerprints") or {}).get("model", "?")
        eff = c.get("effective_precision") or c.get("precision", {}).get("requested", "?") \
            if isinstance(c.get("precision"), dict) else c.get("precision", "?")
        be = c.get("resolved_device", "?")
        return (f"model={str(fp)[:12]} backend={be} prec={eff} "
                f"imgsz={c.get('imgsz', '?')} batch={c.get('batch', '?')}")
    dep_ids = {m: _dep_id(results[m]) for m in ("inference", "sustained", "accuracy", "backends")
               if isinstance(results.get(m), dict) and _eligible(results[m])}
    if len(set(dep_ids.values())) > 1:
        overall &= _check("deployment.identity_match", False,
                          f"mixed deployment identities: {dep_ids} — accuracy of one "
                          f"backend cannot qualify another", checks)
    elif dep_ids:
        _check("deployment.identity_match", True, f"single identity: {next(iter(dep_ids.values()))}",
               checks)

    # power condition: battery runs are marked, strict gate requires AC.
    pw = ((results.get("_power") or {}).get("signals") or {})
    ac = (pw.get("ac_connected") or {}).get("value")
    if gate.get("power", {}).get("require_ac", True):
        if ac is True:
            _check("power.ac", True, "AC connected", checks)
        elif ac is False:
            overall &= _check("power.ac", False,
                              "TEST CONDITION: BATTERY POWER — strict qualification "
                              "requires AC (INCONCLUSIVE, not equivalent)", checks,
                              kind="missing")
        else:
            overall &= _check("power.ac", False,
                              "AC state unknown — cannot certify power condition", checks,
                              kind="missing")

    # sustained thermal evidence: the 10-minute workload's own telemetry counts,
    # so a missing short thermal module is not the only path.
    sus_evo = ((sus.get("tests") or {}).get("resource_evolution") or {})
    st = sus_evo.get("gpu_temp_c") if isinstance(sus_evo.get("gpu_temp_c"), dict) else None
    if st and st.get("max") is not None and _eligible(sus):
        _check("sustained.thermal_telemetry", True,
               f"sustained max temp {st['max']}C over {len(st.get('per_window', []))} windows",
               checks)
        if (st["max"] or 0) >= gate.get("sustained", {}).get("max_temp_c", 90):
            overall &= _check("sustained.temp_limit", False,
                              f"sustained max {st['max']}C over limit", checks)

    states = [r.get("status") for r in results.values()
              if isinstance(r, dict) and not str(r.get("module", "")).startswith("_")]
    hard_fail = any(s in (S.FAILED, S.ABORTED) for s in states if s)
    if not overall or hard_fail:
        verdict = S.FAIL if (hard_fail or _decisive_fail(checks)) else S.INCONCLUSIVE
    else:
        verdict = S.PASS_WITH_HEADROOM if _has_headroom(results, gate) else S.PASS
    return {"verdict": verdict, "req_fps": req, "checks": checks,
            "states": {k: v for k, v in (S.gate_counts(results) or {}).items()},
            "criteria": {"workloads": w, "sustained": gate.get("sustained", {}),
                         "headroom": gate.get("headroom", {}),
                         "accuracy": gate.get("accuracy", {})}}

def _decisive_fail(checks):
    # Only MEASURED violations (or crashed modules) decide FAIL. Missing evidence
    # never fails a machine — it yields INCONCLUSIVE.
    return any(not c["pass"] and c.get("kind", "measured") == "measured" for c in checks)

def _has_headroom(results, gate):
    try:
        t = (results.get("pipeline", {}).get("tests") or {}).get("end_to_end_ms", {})
        fps = t.get("throughput_fps") or 0
        margin = gate.get("headroom", {}).get("min_fps_margin", 1.2)
        miss = max(t.get("deadline_miss_pct", 0), t.get("drop_pct", 0))
        return fps >= gate.get("req_fps", 15) * margin and miss <= gate.get("headroom", {}).get("max_miss_pct", 2.0)
    except Exception:
        return False

_STATUS_RANK = {S.FAILED: 0, S.ABORTED: 1, S.INCONCLUSIVE: 2, S.UNSUPPORTED: 3,
                S.NOT_RUN: 4, S.PARTIAL: 5, S.FALLBACK: 6, S.FULL: 7}

def aggregate_runs(results):
    """Fold module__runN repetitions into a conservative worst-case aggregate.

    Policy (documented): for purchase qualification the WORST critical metric
    across runs decides — min FPS/throughput, max p95/p99/miss/drop, worst
    status. Individual runs are preserved; the aggregate carries the verdict.
    """
    from collections import defaultdict
    groups = defaultdict(list)
    for k, v in results.items():
        if k.startswith("_") or not isinstance(v, dict):
            continue
        base = k.split("__run")[0] if "__run" in k else None
        if base:
            groups[base].append((k, v))
    for base, runs in groups.items():
        runs = [v for _, v in sorted(runs)]
        # Worst run = worst status first, then worst throughput (min fps).
        def _worst_key(v):
            fps = 1e18
            for m in (v.get("tests") or {}).values():
                if isinstance(m, dict):
                    for fk in ("throughput_fps", "steady_fps", "final_fps", "batch_ips"):
                        if isinstance(m.get(fk), (int, float)):
                            fps = min(fps, m[fk])
            return (_STATUS_RANK.get(v.get("status", S.NOT_RUN), -1), fps)
        worst = min(runs, key=_worst_key)
        agg = {"module": base, "tests": {}, "errors": [],
               "status": min((v.get("status", S.NOT_RUN) for v in runs),
                             key=lambda s: _STATUS_RANK.get(s, -1)),
               "runs": [k for k, _ in sorted(groups[base])],
               "aggregation": "worst-case across runs (min fps, max latency/miss/drop)"}
        nums = defaultdict(list)
        for v in runs:
            agg["errors"].extend(v.get("errors", []))
            for tname, m in (v.get("tests") or {}).items():
                if isinstance(m, dict):
                    for mk, mv in m.items():
                        if isinstance(mv, (int, float)):
                            nums[(tname, mk)].append(mv)
        for (tname, mk), vals in nums.items():
            agg["tests"].setdefault(tname, {})
            if any(k in mk for k in ("p95", "p99", "miss", "drop", "max_", "degradation", "growth")):
                agg["tests"][tname][mk] = round(max(vals), 3)
            elif any(k in mk for k in ("fps", "ips", "throughput", "matched_rate", "success")):
                agg["tests"][tname][mk] = round(min(vals), 3)
            else:
                agg["tests"][tname][mk] = round(sum(vals) / len(vals), 3)
        # Non-numeric structures (window series, accounting, configs) come from
        # the worst run intact — aggregation must never silently drop evidence.
        for tname, m in (worst.get("tests") or {}).items():
            for mk, mv in (m.items() if isinstance(m, dict) else []):
                if not isinstance(mv, (int, float)) and mk not in agg["tests"].get(tname, {}):
                    agg["tests"].setdefault(tname, {})[mk] = mv
        if worst.get("config"):
            agg["config"] = worst["config"]
        results[base] = agg
    return results
