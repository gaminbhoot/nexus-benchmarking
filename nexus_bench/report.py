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
    errs = " ".join(res.get("errors", [])).lower()
    if any(k in errs for k in _OOM_WORDS):
        return "exceeds limits", ["out-of-memory during measurement"]
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
    prov_json = json.dumps(prov, indent=2, default=str)[:2500]
    return f"""<html><head><title>NEXUS benchmark {stamp}</title>
<style>body{{font-family:sans-serif;max-width:1100px;margin:auto;padding:20px}}
table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ccc;padding:4px 8px;font-size:13px}}</style>
</head><body><h1>NEXUS Benchmark Report — {stamp} UTC</h1>
<h2>Purchase gate: {gate_verdict}</h2>
<table><tr><th>Check</th><th>Result</th><th>Detail</th></tr>{gate_rows}</table>
<p>Only modules with status FULL count toward the gate. States: {states}</p>
<h2>Provenance</h2><pre>{prov_json}</pre>
<h2>Platform</h2><pre>{json.dumps(prof, indent=2, default=str)[:3000]}</pre>
<h2>Feasibility (required: {results['_meta']['req_fps']} FPS; p95 + misses decide)</h2>
<table><tr><th>Module</th><th>Verdict</th><th>Meaning</th><th>Why</th></tr>{feas_rows}</table>
<h2>Latency (median + p95, ms)</h2>{_chart(rows)}
<h2>Detailed measurements</h2>
<table><tr><th>Module</th><th>Test</th><th>Median</th><th>P95</th><th>P99</th><th>FPS/IPS</th><th>Miss%</th><th>Drop%</th><th>Error/Value</th></tr>{det}</table>
<p>Verdicts use tail latency and measured throughput only. A "limited" module may show a
fine median — check its p95/miss columns. Conclusions reflect measured numbers only.</p>
</body></html>"""
