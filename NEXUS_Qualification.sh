#!/bin/bash
# NEXUS Hardware Qualification — seller entry point (Linux).
# No choices, no files, no commands: runs the official qualification.
set -euo pipefail
cd "$(dirname "$0")"
LOG="nexus_launcher.log"
: > "$LOG"

if ! command -v python3 >/dev/null 2>&1; then
  echo "NEXUS Qualification needs Python 3.10+, which was not found."
  echo "Install it from https://www.python.org/downloads/ and run again."
  exit 1
fi
if [ ! -x ".venv/bin/python" ]; then
  echo "Preparing qualification runtime (one-time setup)..."
  python3 -m venv .venv >>"$LOG" 2>&1
fi
./.venv/bin/python -m pip install -q -e . >>"$LOG" 2>&1
./.venv/bin/python -m nexus_bench.cli qualify 2>&1 | tee -a "$LOG"
echo "Log saved to $LOG"
