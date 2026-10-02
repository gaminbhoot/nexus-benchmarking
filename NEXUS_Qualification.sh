#!/bin/bash
# NEXUS Hardware Qualification — seller entry point (Linux).
# One-time setup uses the PINNED runtime (requirements.lock); afterwards fully offline.
set -euo pipefail
cd "$(dirname "$0")"
LOG="nexus_launcher.log"
: > "$LOG"

if ! command -v python3 >/dev/null 2>&1; then
  echo "NEXUS Qualification needs Python 3.10+, which was not found."
  echo "Install it, then run again. Internet is needed ONCE for setup."
  exit 1
fi
if [ ! -x ".venv/bin/python" ]; then
  echo "Preparing qualification runtime (one-time setup, pinned versions)..."
  python3 -m venv .venv >>"$LOG" 2>&1
  ./.venv/bin/python -m pip install -q -r requirements.lock >>"$LOG" 2>&1
  ./.venv/bin/python -m pip install -q --no-deps -e . >>"$LOG" 2>&1
fi
./.venv/bin/python -m nexus_bench.cli qualify 2>&1 | tee -a "$LOG"
echo "Log saved to $LOG"
