"""Results: JSON + CSV + HTML report + feasibility verdicts."""
import base64
import csv
import io
import json
from datetime import datetime, timezone
from pathlib import Path

def _flat(results):
    rows = []
    for mod, res in results.items():
        if mod.startswith("_"):
            continue
        for test, m in (res.get("tests") or {}).items():
            row = {"module": mod, "test": test}
            if isinstance(m, dict):
                for k in ("mean_ms", "median_ms", "p95_ms", "p99_ms", "throughput_ips",
                          "total_ms", "avg_ips", "kmsgs_per_s", "degradation_pct"):
                    if k in m and isinstance(m[k], (int, float)):
                        row[k] = round(m[k], 3)
            rows.append(row)
        for e in res.get("errors", []) or []:
            rows.append({"module": mod, "test": "ERROR", "error": str(e)[:300]})
    return rows

def feasibility(results, req_fps=15):
    """Per-module verdict from measured numbers only. No real-time suitability claims."""
    verdicts = {}
    for mod, res in results.items():
        if mod.startswith("_"):
            continue
        errs = " ".join(res.get("errors", [])).lower()
        if any(k in errs for k in ("oom", "exceed", "out of memory")):
            verdicts[mod] = "exceeds limits"
            continue
        if res.get("errors") and not res.get("tests"):
            verdicts[mod] = "unsupported"
            continue
        slow = False
        for m in (res.get("tests") or {}).values():
            if isinstance(m, dict) and m.get("median_ms") and m["median_ms"] > 1000 / req_fps:
                slow = True
        verdicts[mod] = "limited" if (slow or res.get("errors")) else "ok"
    return verdicts

STATUSES = {"ok": "Runs within available resources",
            "limited": "Runs with performance limitations",
            "exceeds limits": "Exceeds available memory/resource limits",
            "unsupported": "Unsupported on the detected platform"}

def write_all(results, outdir, req_fps=15):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    results["_meta"] = {"timestamp_utc": stamp, "req_fps": req_fps,
                        "feasibility": feasibility(results, req_fps)}
    jp = out / f"results_{stamp}.json"
    jp.write_text(json.dumps(results, indent=2, default=str))
    rows = _flat(results)
    cp = out / f"results_{stamp}.csv"
    with open(cp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["module", "test", "mean_ms", "median_ms",
                                          "p95_ms", "p99_ms", "throughput_ips",
                                          "total_ms", "avg_ips", "kmsgs_per_s",
                                          "degradation_pct", "error"])
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
        pts = [(f"{r['module']}/{r['test']}"[:38], r["median_ms"]) for r in rows
               if r.get("median_ms") is not None]
        if not pts:
            return ""
        pts = pts[:20]
        fig, ax = plt.subplots(figsize=(9, max(3, len(pts) * 0.35)))
        ax.barh([p[0] for p in pts][::-1], [p[1] for p in pts][::-1])
        ax.set_xlabel("median ms"); fig.tight_layout()
        buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig)
        return '<img src="data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode() + '"/>'
    except Exception:
        return "<p>(chart unavailable)</p>"

def _html(results, rows, stamp):
    feas = results["_meta"]["feasibility"]
    feas_rows = "".join(f"<tr><td>{m}</td><td>{v}</td><td>{STATUSES.get(v, v)}</td></tr>"
                        for m, v in feas.items())
    det = "".join(f"<tr><td>{r.get('module')}</td><td>{r.get('test')}</td>"
                  f"<td>{r.get('median_ms', '')}</td><td>{r.get('p95_ms', '')}</td>"
                  f"<td>{r.get('throughput_ips', '')}</td><td>{r.get('error', '')}</td></tr>"
                  for r in rows)
    prof = results.get("_profile", {})
    return f"""<html><head><title>NEXUS benchmark {stamp}</title>
<style>body{{font-family:sans-serif;max-width:1000px;margin:auto;padding:20px}}
table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ccc;padding:4px 8px;font-size:13px}}</style>
</head><body><h1>NEXUS Benchmark Report — {stamp} UTC</h1>
<h2>Platform</h2><pre>{json.dumps(prof, indent=2, default=str)[:2000]}</pre>
<h2>Feasibility (req: {results['_meta']['req_fps']} FPS)</h2>
<table><tr><th>Module</th><th>Verdict</th><th>Meaning</th></tr>{feas_rows}</table>
<h2>Latency (median ms)</h2>{_chart(rows)}
<h2>Detailed measurements</h2>
<table><tr><th>Module</th><th>Test</th><th>Median</th><th>P95</th><th>IPS</th><th>Error</th></tr>{det}</table>
<p>Full data: JSON + CSV alongside this file. Conclusions reflect measured numbers only.</p>
</body></html>"""
