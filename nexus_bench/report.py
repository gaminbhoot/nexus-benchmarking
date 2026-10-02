"""Results: JSON + CSV + HTML report + feasibility verdicts.

Feasibility is driven by TAIL latency (p95), deadline misses, drops, measured
throughput vs the required FPS, and resource evidence — never by median alone.
Every verdict carries machine-readable reasons shown in the HTML report.
"""
import base64
import csv
import io
import json
from datetime import datetime, timezone
from pathlib import Path

STATUSES = {"ok": "Runs within available resources",
            "limited": "Runs with performance limitations",
            "exceeds limits": "Exceeds available memory/resource limits",
            "unsupported": "Unsupported on the detected platform (no real workload ran)"}

_OOM_WORDS = ("out of memory", "oom", "cuda error", "cublas")

def _deadline_ms(res, req_fps):
    for src in (res.get("config") or {},):
        if isinstance(src.get("deadline_ms"), (int, float)):
            return float(src["deadline_ms"])
    for m in (res.get("tests") or {}).values():
        if isinstance(m, dict) and isinstance(m.get("deadline_ms"), (int, float)):
            return float(m["deadline_ms"])
    return 1000.0 / req_fps if req_fps else None

def _assess(mod, res, req_fps):
    """Returns (verdict, [reasons]). Only measured numbers count."""
    from nexus_bench import statuses as S
    reasons = []
    errs = [str(e) for e in res.get("errors", [])]
    workload_oom = any(("out of memory" in e.lower() or "oom" in e.lower())
                       and "capacity_probe_oom" not in e.lower() for e in errs)
    if workload_oom:
        return "exceeds limits", ["out-of-memory during workload measurement"]
    if res.get("status") in (S.UNSUPPORTED, S.NOT_RUN) or (
            res.get("errors") and not res.get("tests")):
        return "unsupported", [f"status={res.get('status')}: no real workload executed"]
    if res.get("status") in (S.PARTIAL, S.FALLBACK):
        part = [f"status={res.get('status')}: partial evidence only (not gate-grade)"]
    else:
        part = []
    tests = res.get("tests") or {}
    deadline = _deadline_ms(res, req_fps)
    worst_miss, worst_drop, fps_vals, tail_slow, med_slow = 0, 0, [], False, False
    for name, m in tests.items():
        if not isinstance(m, dict):
            continue
        if isinstance(m.get("deadline_miss_pct"), (int, float)):
            worst_miss = max(worst_miss, m["deadline_miss_pct"])
        if isinstance(m.get("drop_pct"), (int, float)):
            worst_drop = max(worst_drop, m["drop_pct"])
        if isinstance(m.get("throughput_fps"), (int, float)):
            fps_vals.append(m["throughput_fps"])
        p95, med = m.get("p95_ms"), m.get("median_ms")
        if deadline and isinstance(p95, (int, float)) and p95 > deadline:
            tail_slow = True
            reasons.append(f"{name}: p95 {p95:.1f}ms > deadline {deadline:.1f}ms")
        if deadline and isinstance(med, (int, float)) and med > deadline:
            med_slow = True
    if res.get("module") == "thermal" and (tests.get("throttling") == "likely"):
        return "limited", ["thermal throttling evidence under sustained load"]
    if worst_miss > 20 or worst_drop > 20:
        return "exceeds limits", [f"deadline-miss {worst_miss}% / drops {worst_drop}% — cannot hold {req_fps} FPS"]
    if fps_vals and max(fps_vals) < req_fps / 2:
        return "exceeds limits", [f"measured {max(fps_vals)} FPS vs required {req_fps} FPS"]
    limited_reasons = list(reasons) + part
    if 5 <= worst_miss <= 20 or 5 <= worst_drop <= 20:
        limited_reasons.append(f"deadline-miss {worst_miss}% / drops {worst_drop}%")
    if fps_vals and max(fps_vals) < req_fps:
        limited_reasons.append(f"measured {max(fps_vals)} FPS vs required {req_fps} FPS")
    if res.get("errors"):
        limited_reasons.append(f"{len(res['errors'])} non-fatal error(s); see log")
    if tail_slow or med_slow or limited_reasons or part:
        if not limited_reasons and med_slow:
            limited_reasons.append("median latency exceeds deadline")
        return "limited", limited_reasons or ["tail latency exceeds deadline"]
    return "ok", [f"p95 within {deadline:.1f}ms deadline; misses/drops <5%"] if deadline else ["within measured budgets"]

def feasibility(results, req_fps=15):
    verdicts, reasons = {}, {}
    for mod, res in results.items():
        if mod.startswith("_"):
            continue
        v, r = _assess(mod, res, req_fps)
        verdicts[mod], reasons[mod] = v, r
    return verdicts, reasons

def _flat(results):
    rows = []
    for mod, res in results.items():
        if mod.startswith("_"):
            continue
        for test, m in (res.get("tests") or {}).items():
            row = {"module": mod, "test": test}
            if isinstance(m, dict):
                for k in ("mean_ms", "median_ms", "p95_ms", "p99_ms",
                          "throughput_ips", "throughput_fps", "per_image_ms",
                          "total_ms", "avg_ips", "kmsgs_per_s", "degradation_pct",
                          "deadline_miss_pct", "drop_pct"):
                    if k in m and isinstance(m[k], (int, float)):
                        row[k] = round(m[k], 3)
            elif isinstance(m, (int, float, str)):
                row["value"] = m
            rows.append(row)
        for e in res.get("errors", []) or []:
            rows.append({"module": mod, "test": "ERROR", "error": str(e)[:300]})
    return rows

def write_all(results, outdir, req_fps=15):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    verdicts, reasons = feasibility(results, req_fps)
    results["_meta"] = {"timestamp_utc": stamp, "req_fps": req_fps,
                        "feasibility": verdicts, "feasibility_reasons": reasons,
                        "gate": results.get("_gate", {"verdict": "not-evaluated"}),
                        "provenance": results.get("_provenance", {}),
                        "method": ("tail-latency + deadline-miss + drop-rate + measured "
                                   "throughput vs required FPS; medians never decide alone")}
    jp = out / f"results_{stamp}.json"
    jp.write_text(json.dumps(results, indent=2, default=str))
    rows = _flat(results)
    cp = out / f"results_{stamp}.csv"
    with open(cp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["module", "test", "mean_ms", "median_ms",
                                          "p95_ms", "p99_ms", "throughput_ips",
                                          "throughput_fps", "per_image_ms", "total_ms",
                                          "avg_ips", "kmsgs_per_s", "degradation_pct",
                                          "deadline_miss_pct", "drop_pct", "value", "error"])
        w.writeheader(); w.writerows(rows)
    hp = out / f"report_{stamp}.html"
    hp.write_text(_html(results, rows, stamp))
    log = out / f"errors_{stamp}.log"
    log.write_text("\n".join(f"[{m}] {e}" for m, r in results.items()
                             if not m.startswith("_") for e in r.get("errors", [])) or "no errors")
    return {"json": str(jp), "csv": str(cp), "html": str(hp), "log": str(log)}

def _chart(rows):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        pts = [(f"{r['module']}/{r['test']}"[:38], r["median_ms"], r.get("p95_ms")) for r in rows
               if r.get("median_ms") is not None]
        if not pts:
            return ""
        pts = pts[:20]
        labels = [p[0] for p in pts][::-1]
        y = range(len(labels))
        fig, ax = plt.subplots(figsize=(9, max(3, len(pts) * 0.4)))
        ax.barh(list(y), [p[1] for p in pts][::-1], label="median")
        ax.barh(list(y), [(p[2] or p[1]) for p in pts][::-1], alpha=0.35, label="p95")
        ax.set_yticks(list(y), labels, fontsize=8)
        ax.set_xlabel("ms"); ax.legend(); fig.tight_layout()
        buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig)
        return '<img src="data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode() + '"/>'
    except Exception:
        return "<p>(chart unavailable)</p>"

def _chart(rows):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        pts = [(f"{r['module']}/{r['test']}"[:38], r["median_ms"], r.get("p95_ms")) for r in rows
               if r.get("median_ms") is not None]
        if not pts:
            return ""
        pts = pts[:20]
        labels = [p[0] for p in pts][::-1]
        y = range(len(labels))
        fig, ax = plt.subplots(figsize=(9, max(3, len(pts) * 0.4)))
        ax.barh(list(y), [p[1] for p in pts][::-1], label="median")
        ax.barh(list(y), [(p[2] or p[1]) for p in pts][::-1], alpha=0.35, label="p95")
        ax.set_yticks(list(y), labels, fontsize=8)
        ax.set_xlabel("ms"); ax.legend(); fig.tight_layout()
        buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig)
        return '<img src="data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode() + '"/>'
    except Exception:
        return "<p>(chart unavailable)</p>"

def _line_chart(series, title, ylabel, window_s=60):
    """Time-series chart. series: {label: [values per window]}. Chronological."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 3.2))
        for label, vals in series.items():
            xs = [i * window_s / 60.0 for i, v in enumerate(vals) if v is not None]
            ys = [v for v in vals if v is not None]
            if ys:
                ax.plot(xs, ys, marker="o", markersize=3, label=label)
        ax.set_title(title); ax.set_xlabel("minutes"); ax.set_ylabel(ylabel)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3); fig.tight_layout()
        buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig)
        return '<img src="data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode() + '"/>'
    except Exception:
        return "<p>(chart unavailable)</p>"

def _sustained_section(results):
    sus = results.get("sustained", {})
    if not sus or not isinstance(sus.get("tests"), dict):
        return "<p>Sustained qualification was not run in this report.</p>"
    t = sus["tests"]
    window_s = (sus.get("config") or {}).get("sustained_window_s", 60)
    per = t.get("per_agent", {})
    charts = ""
    fps_series, p95_series = {}, {}
    for agent, a in per.items():
        if isinstance(a, dict):
            if a.get("windows_fps"):
                fps_series[f"{agent} FPS"] = a["windows_fps"]
            if a.get("windows_p95_ms"):
                p95_series[f"{agent} p95 (ms)"] = a["windows_p95_ms"]
    if fps_series:
        charts += "<h3>FPS vs time</h3>" + _line_chart(fps_series, "Throughput per window", "FPS", window_s)
    if p95_series:
        charts += "<h3>p95 latency vs time</h3>" + _line_chart(p95_series, "p95 latency per window", "ms", window_s)
    evo = t.get("resource_evolution", {}) or {}
    for key, title, unit in (("vram_pct", "VRAM vs time", "%"),
                             ("gpu_temp_c", "GPU temperature vs time", "C"),
                             ("gpu_util_pct", "GPU utilization vs time", "%"),
                             ("gpu_power_w", "GPU power vs time", "W")):
        entry = evo.get(key) or {}
        if isinstance(entry, dict) and entry.get("per_window"):
            charts += f"<h3>{title}</h3>" + _line_chart({key: entry["per_window"]}, title, unit, window_s)
    qd = t.get("queue_depth_max_per_window", {}) or {}
    if any(qd.values()):
        charts += "<h3>Queue depth vs time</h3>" + _line_chart(
            {k: v for k, v in qd.items() if v}, "Max queue depth per window", "frames", window_s)
    rows = ""
    for agent, a in per.items():
        if not isinstance(a, dict):
            continue
        rows += (f"<tr><td>{agent}</td><td>{a.get('initial_fps')}</td>"
                 f"<td>{a.get('steady_fps')}</td><td>{a.get('min_fps')}</td>"
                 f"<td>{a.get('final_fps')}</td><td>{a.get('degradation_pct')}%</td>"
                 f"<td>{a.get('max_p95_ms')}</td><td>{a.get('drop_pct')}%</td>"
                 f"<td>{a.get('deadline_miss_pct')}%</td><td>{a.get('oom_events')}</td></tr>")
    mg = t.get("memory_growth", {}) or {}
    table = (f"<table><tr><th>Agent</th><th>First FPS</th><th>Steady FPS</th><th>Worst FPS</th>"
             f"<th>Final FPS</th><th>Degradation</th><th>Max p95</th><th>Drops</th>"
             f"<th>Misses</th><th>OOMs</th></tr>{rows}</table>"
             f"<p>Memory growth: {mg.get('growth')} MB "
             f"({'LEAK SUSPECTED' if mg.get('leak_suspected') else 'no leak signal'}). "
             f"Fusion ticks: {(t.get('fusion') or {}).get('ticks')}. "
             f"Limitations: {'; '.join((sus.get('config') or {}).get('limitations', []))}</p>")
    return table + charts

def _summary_rows(results):
    """Metric / Result / Limit / Status. Limits read from the gate criteria
    stored in the report — one source of truth, never a second hardcoded copy."""
    gate = results.get("_gate", {}) or {}
    checks = {c["check"]: c for c in gate.get("checks", [])}
    crit = gate.get("criteria", {}) or {}
    sus_crit = crit.get("sustained", {}) or {}
    wl = crit.get("workloads", {}) or {}
    rows = []

    def row(metric, result, limit, check_prefixes):
        """4-state row: PASS / FAIL / INCONCLUSIVE (criterion exists, evidence
        missing) / NOT TESTED (no criterion ran). Never red for missing data."""
        c = None
        if isinstance(check_prefixes, str):
            check_prefixes = (check_prefixes,)
        for prefix in check_prefixes:
            c = checks.get(next((k for k in checks if k.startswith(prefix)), ""), None)
            if c:
                break
        if c is None:
            status = "NOT TESTED"
        elif not c.get("pass") and c.get("kind") == "missing":
            status = "INCONCLUSIVE"
        else:
            status = "PASS" if c.get("pass") else "FAIL"
        color = {"PASS": "green", "FAIL": "red"}.get(status, "#b26a00")
        rows.append((metric, result, limit,
                     f"<span style='font-weight:bold;color:{color}'>{status}</span>"))

    sus = (results.get("sustained", {}) or {}).get("tests", {}) or {}
    per = sus.get("per_agent", {}) or {}
    for agent in ("uav", "rover"):
        a = per.get(agent) or {}
        lim = sus_crit.get("min_fps_each", 10)
        row(f"{agent.upper()} steady FPS", a.get("steady_fps", "—"), f">={lim}",
            f"sustained.{agent}.steady_fps")
        dlim = sus_crit.get("max_drop_pct", 5)
        dv = a.get("drop_pct")
        row(f"{agent.upper()} drops", f"{dv}%" if dv is not None else "—", f"<={dlim}%",
            f"sustained.{agent}.drop_pct")
    pipe = ((results.get("pipeline", {}) or {}).get("tests", {}) or {}).get("end_to_end_ms", {}) or {}
    if pipe:
        plim = (wl.get("uav", {}) or {}).get("max_p95_ms", 66.7)
        pv = pipe.get("p95_ms")
        row("Pipeline p95", f"{pv} ms" if pv is not None else "—", f"<={plim} ms",
            ("uav.p95_ms", "uav.max_p95_ms"))
    else:
        row("Pipeline p95", "—", "—", "no.such.check")
    evo = sus.get("resource_evolution", {}) or {}
    vram = (evo.get("vram_pct") or {}) if isinstance(evo.get("vram_pct"), dict) else {}
    if vram.get("max") is not None:
        vlim = (crit.get("headroom", {}) or {}).get("max_vram_occupied_pct", 85)
        row("Peak VRAM", f"{vram['max']}%", f"<={vlim}%", "resources.vram_headroom")
    else:
        row("Peak VRAM", "—", "—", "resources.vram_headroom")
    tmp = (evo.get("gpu_temp_c") or {}) if isinstance(evo.get("gpu_temp_c"), dict) else {}
    if tmp.get("max") is not None:
        row("Max GPU temp", f"{tmp['max']} C", "no throttle evidence", "resources.no_throttle")
    else:
        row("Max GPU temp", "—", "—", "resources.no_throttle")
    comp = ((results.get("accuracy", {}) or {}).get("tests", {}) or {}).get("map_comparison", {}) or {}
    if comp.get("map50_drop_abs") is not None:
        alim = (crit.get("accuracy", {}) or {}).get("max_map_drop", 0.05)
        row("Accuracy drop (mAP50)", comp["map50_drop_abs"], f"<={alim}",
            "accuracy.map50_drop_abs")
    else:
        row("Accuracy drop (mAP50)", "—", "—", "accuracy.map50_drop_abs")
    ooms = sum((a.get("oom_events") or 0) for a in per.values() if isinstance(a, dict))
    row("Workload OOMs", ooms, "0", "resources.no_oom")
    if per:
        actual = min((a.get("actual_duration_s") or 0) for a in per.values() if isinstance(a, dict))
        req = (results.get("sustained", {}) or {}).get("config", {}).get("sustained_duration_s", "?")
        row("Actual sustained duration", f"{actual} s", f">={req} s",
            "sustained.uav.actual_duration")
    return "".join(f"<tr><td>{m}</td><td>{r}</td><td>{l}</td><td>{s}</td></tr>"
                   for m, r, l, s in rows)

def _deployment_block(results):
    cfgs = {}
    for m in ("inference", "sustained", "accuracy", "backends"):
        r = results.get(m)
        if isinstance(r, dict) and isinstance(r.get("config"), dict):
            c = r["config"]
            fp = ((results.get("_provenance") or {}).get("fingerprints") or {}).get("model", "?")
            eff = c.get("effective_precision") or (
                c.get("precision", {}).get("requested") if isinstance(c.get("precision"), dict)
                else c.get("precision"))
            cfgs[m] = (f"model={str(fp)[:16]}… backend={c.get('resolved_device', '?')} "
                       f"precision={eff} resolution={c.get('imgsz', '?')} "
                       f"batch={c.get('batch', '?')}")
    rows = "".join(f"<tr><td>{m}</td><td>{v}</td></tr>" for m, v in cfgs.items())
    notes = ("<p>Accuracy, inference, sustained, and gate results must all refer to ONE "
             "deployment identity above. A mismatch fails the gate.</p>" if len(set(cfgs.values())) > 1
             else "<p>Single deployment identity across modules.</p>")
    return f"<table><tr><th>Module</th><th>Deployment identity</th></tr>{rows}</table>{notes}"

def _assets_block(results):
    recs = results.get("_assets", {}) or {}
    manifest = recs.get("manifest", [])
    if not manifest:
        return "<p>Official asset verification was not run for this report.</p>"
    rows = "".join(f"<tr><td>{r.get('asset_id')}</td><td>{r.get('status')}</td>"
                   f"<td>{r.get('expected_sha256', '')[:16]}…</td></tr>" for r in manifest)
    bad = [r for r in manifest if r.get("status") != "OK"]
    flag = ("<p><b>QUALIFICATION INVALID — official assets were modified or corrupted. "
            "No gate verdict in this report may be used.</b></p>" if bad else
            f"<p>Qualification v{recs.get('qualification_version', '?')}: all official "
            "assets verified.</p>")
    return flag + f"<table><tr><th>Asset</th><th>Status</th><th>SHA-256</th></tr>{rows}</table>"

def _provenance_block(results, prov):
    rel = prov.get("release")
    if rel:
        return (f"<p>Release package: NEXUS Qualification v{rel.get('qualification_version', '?')} "
                f"({rel.get('release_id', '?')}). Source identity: release manifest "
                f"{str(rel.get('source_tree_sha256', ''))[:16]}…. "
                f"Git: not applicable in packaged distribution.</p>"
                f"<pre>{json.dumps(prov, indent=2, default=str)[:2000]}</pre>")
    return f"<pre>{json.dumps(prov, indent=2, default=str)[:4000]}</pre>"

def _limitations_block(results):
    items = ["Depth-estimation network: NOT TESTED (occupancy-grid proxy in rover agent)",
             "Mission planner: representative allocation proxy only",
             "Physical wireless link: NOT TESTED (loopback stack cost only)"]
    sus = ((results.get("sustained", {}) or {}).get("config") or {}).get("limitations", [])
    items.extend(x for x in sus if x not in items)
    return "<ul>" + "".join(f"<li>{x}</li>" for x in items) + "</ul>"

def plain_summary(results):
    """Plain-text summary a seller can send by email/WhatsApp. No jargon."""
    gate = results.get("_gate", {}) or {}
    verdict = gate.get("verdict", "INCONCLUSIVE")
    prof = results.get("_profile", {}) or {}
    gpu = (prof.get("gpu") or {}).get("name", "unknown GPU")
    ram = (prof.get("memory") or {}).get("ram_total_gb", "?")
    cfg = results.get("_config", {}) or {}
    dur = (results.get("sustained", {}) or {}).get("config", {}).get("duration_s") \
        or cfg.get("duration_s", "?")
    fails = [c for c in gate.get("checks", []) if not c["pass"]]
    not_tested = sorted(k for k, v in results.items()
                        if not k.startswith("_") and isinstance(v, dict)
                        and "__run" not in k and v.get("status") != "FULL")
    lines = [
        "NEXUS Hardware Qualification",
        f"Machine GPU: {gpu}",
        f"RAM: {ram} GB",
        f"Test duration: {dur} seconds",
        f"NEXUS target: {cfg.get('req_fps', 15)} FPS",
        "",
        f"RESULT: {verdict}",
        "",
    ]
    if verdict == "PASS" or verdict == "PASS_WITH_HEADROOM":
        lines.append("Reason: all required workloads held the target throughput for the "
                     "full run with no workload OOM, VRAM within limits, no throttling "
                     "evidence, and accuracy within tolerance.")
    elif verdict == "FAIL":
        lines.append("Reason:")
        for c in fails[:6]:
            lines.append(f"- {c['check']}: {c['detail']}")
    else:
        lines.append("Reason: the evidence was insufficient to certify the hardware. "
                     "The benchmark ran, but a strict purchase decision needs more proof.")
        for c in fails[:6]:
            lines.append(f"- {c['check']}: {c['detail']}")
    if not_tested:
        lines.append("")
        lines.append(f"Not fully tested: {', '.join(sorted(set(not_tested)))}")
    lines += ["",
              "Detailed report: NEXUS_Qualification_Report.html",
              "This summary cannot overrule the detailed report: only FULL-status "
              "measurements count toward qualification."]
    return "\n".join(lines)

def _html(results, rows, stamp):
    feas = results["_meta"]["feasibility"]
    why = results["_meta"]["feasibility_reasons"]
    gate = results["_meta"].get("gate", {})
    prov = results["_meta"].get("provenance", {})
    gate_rows = "".join(
        f"<tr><td>{c['check']}</td><td>{'PASS' if c['pass'] else 'FAIL'}</td>"
        f"<td>{c['detail']}</td></tr>" for c in gate.get("checks", [])) or "<tr><td colspan=3>no gate evaluated</td></tr>"
    feas_rows = "".join(
        f"<tr><td>{m}</td><td>{v}</td><td>{STATUSES.get(v, v)}</td>"
        f"<td>{'; '.join(why.get(m, []))}</td></tr>" for m, v in feas.items())
    det = "".join(f"<tr><td>{r.get('module')}</td><td>{r.get('test')}</td>"
                  f"<td>{r.get('median_ms', '')}</td><td>{r.get('p95_ms', '')}</td>"
                  f"<td>{r.get('p99_ms', '')}</td>"
                  f"<td>{r.get('throughput_fps', r.get('throughput_ips', ''))}</td>"
                  f"<td>{r.get('deadline_miss_pct', '')}</td><td>{r.get('drop_pct', '')}</td>"
                  f"<td>{r.get('error', r.get('value', ''))}</td></tr>" for r in rows)
    prof = results.get("_profile", {})
    gate_verdict = gate.get("verdict", "not-evaluated")
    states = json.dumps(gate.get("states", {}), default=str)
    color = {"PASS": "green", "PASS_WITH_HEADROOM": "green", "FAIL": "red"}.get(gate_verdict, "#b26a00")
    why_bullets = "".join(f"<li><b>{c['check']}</b>: {c['detail']}</li>"
                          for c in gate.get("checks", []) if not c["pass"]) or \
        "<li>All required workloads held the target throughput with no workload OOM, " \
        "VRAM within limits, no throttling evidence, and accuracy within tolerance.</li>"
    hw_fail = [c for c in gate.get("checks", []) if not c["pass"] and c["check"].startswith("hardware.")]
    hw_banner = ("<div style='background:#ffe0e0;padding:12px;border:2px solid red'>"
                 "<b>TARGET HARDWARE DOES NOT MATCH TEST HARDWARE.</b> This report must not "
                 "be presented as a qualification of a different machine.</div>"
                 if hw_fail else "")
    dirty = (prov.get("git") or {}).get("dirty")
    dirty_banner = ("<div style='background:#fff3cd;padding:12px;border:2px solid #b26a00'>"
                    "<b>UNRELEASED / DIRTY SOURCE.</b> The benchmark tree had uncommitted "
                    "changes; results are not reproducible from the commit SHA alone.</div>"
                    if dirty else "")
    # NOT TESTED = every module whose own status is not FULL evidence:
    not_tested = sorted(k for k, v in results.items()
                        if not k.startswith("_") and isinstance(v, dict)
                        and "__run" not in k and v.get("status") != "FULL")
    power = results.get("_power", {}) or {}
    power_rows = "".join(
        f"<tr><td>{k}</td><td>{v.get('value')}</td><td>{v.get('how')}</td></tr>"
        for k, v in (power.get("signals") or {}).items())
    machine = f"{(prof.get('cpu') or {}).get('brand', '?')} / " \
              f"{(prof.get('gpu') or {}).get('name', (prof.get('gpu') or {}).get('note', '?'))} / " \
              f"{(prof.get('memory') or {}).get('ram_total_gb', '?')} GB RAM"
    rel = prov.get("release") or {}
    import hashlib as _hl
    gate_fp = _hl.sha256(json.dumps(gate.get("checks", []), sort_keys=True, default=str).encode()).hexdigest()[:12]
    stamps = (f"NEXUS Qualification v{rel.get('qualification_version', '?')} · "
              f"benchmark {prov.get('benchmark_version', '?')} · gate-evidence {gate_fp}")
    dur = (results.get("sustained", {}) or {}).get("config", {}).get("sustained_duration_s", "—")
    # deployment identity: the exact qualified configuration, one block.
    dep = _deployment_block(results)
    assets_html = _assets_block(results)
    prov_block = _provenance_block(results, prov)
    limitations = _limitations_block(results)
    return f"""<html><head><title>NEXUS Hardware Qualification {stamp}</title>
<style>body{{font-family:sans-serif;max-width:1100px;margin:auto;padding:20px}}
table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ccc;padding:4px 8px;font-size:13px}}
.hero{{text-align:center;padding:24px;border:3px solid {color};margin:16px 0}}
.hero h1{{font-size:44px;margin:8px;color:{color}}}</style>
</head><body>
<div class="hero"><div>NEXUS HARDWARE QUALIFICATION</div>
<div style="font-size:14px">{machine} · sustained {dur}s · target {results['_meta']['req_fps']} FPS<br/>{stamps}</div>
<h1>{gate_verdict}</h1></div>
{hw_banner}{dirty_banner}
<h2>Why?</h2><ul>{why_bullets}</ul>
<h2>Summary</h2>
<table><tr><th>Metric</th><th>Result</th><th>Limit</th><th>Status</th></tr>{_summary_rows(results)}</table>
<h2>Sustained qualification</h2>{_sustained_section(results)}
<h2>Deployment identity</h2>{dep}
<h2>Official assets</h2>{assets_html}
<h2>LIMITATIONS — what this report does NOT certify</h2>{limitations}
<h2>Purchase gate checks</h2>
<table><tr><th>Check</th><th>Result</th><th>Detail</th></tr>{gate_rows}</table>
<p>Only FULL-status measurements count. States: {states}. NOT TESTED / non-FULL: {', '.join(not_tested) or 'none'}</p>
<h2>Power (measured vs unavailable — never invented)</h2>
<table><tr><th>Signal</th><th>Value</th><th>How</th></tr>{power_rows or '<tr><td colspan=3>no power section</td></tr>'}</table>
<p>{(power.get('coverage') or {}).get('note', '')}</p>
<h2>Provenance</h2>{prov_block}
<h2>Platform</h2><pre>{json.dumps(prof, indent=2, default=str)[:3000]}</pre>
<h2>Feasibility (required: {results['_meta']['req_fps']} FPS; p95 + misses decide)</h2>
<table><tr><th>Module</th><th>Verdict</th><th>Meaning</th><th>Why</th></tr>{feas_rows}</table>
<h2>Latency (median + p95, ms)</h2>{_chart(rows)}
<h2>Detailed measurements</h2>
<table><tr><th>Module</th><th>Test</th><th>Median</th><th>P95</th><th>P99</th><th>FPS/IPS</th><th>Miss%</th><th>Drop%</th><th>Error/Value</th></tr>{det}</table>
<p>A nice-looking report never overrules missing evidence: PASS requires every
critical check above to hold on FULL-status measurements.</p>
</body></html>"""
