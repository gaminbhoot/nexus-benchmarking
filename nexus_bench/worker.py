"""Isolated benchmark worker: runs ONE module in a fresh process.

Usage: python -m nexus_bench.worker <module> <cfg.json> <out.json>
Always writes a result envelope (even on failure) so the controller can
distinguish FAILED from lost. Fresh process = fresh CUDA context/allocator.
"""
import importlib
import json
import sys
import traceback

from nexus_bench import statuses as S

def main():
    mod_name, cfg_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    try:
        dotted = mod_name if mod_name.startswith("nexus_bench.") else f"nexus_bench.{mod_name}"
        mod = importlib.import_module(dotted)
    except Exception as e:
        _write(out_path, {"module": mod_name, "status": S.FAILED, "tests": {},
                          "errors": [f"worker import: {e}"]})
        return 1
    try:
        res = mod.run(cfg)
        st = res.get("status")
        if st not in (S.FULL, S.PARTIAL, S.FALLBACK, S.UNSUPPORTED, S.FAILED,
                      S.INCONCLUSIVE, S.ABORTED, S.NOT_RUN):
            # Fail closed: a module that never established a valid measurement
            # status must never be treated as a successful FULL measurement.
            res["errors"] = list(res.get("errors", [])) + [
                f"worker: no valid status established (got {st!r}) -> FAILED"]
            res["status"] = S.FAILED
        _write(out_path, res)
        return 0
    except KeyboardInterrupt:
        _write(out_path, {"module": mod_name, "status": S.ABORTED, "tests": {},
                          "errors": ["cancelled"]})
        return 130
    except Exception as e:
        _write(out_path, {"module": mod_name, "status": S.FAILED, "tests": {},
                          "errors": [f"worker exception: {e}", traceback.format_exc(limit=3)]})
        return 1

def _write(path, res):
    import os
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(res, f, default=str)
    os.replace(tmp, path)

if __name__ == "__main__":
    raise SystemExit(main())
