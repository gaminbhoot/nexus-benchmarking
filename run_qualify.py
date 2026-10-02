"""One-binary entry point for PyInstaller (see nexus_qualify.spec)."""
from nexus_bench.qualify import VERDICT_EXIT, run_qualify
import sys

if __name__ == "__main__":
    extended = "--extended" in sys.argv
    _folder, _zipp, results = run_qualify(extended=extended)
    sys.exit(VERDICT_EXIT.get((results.get("_gate") or {}).get("verdict"), 3))
