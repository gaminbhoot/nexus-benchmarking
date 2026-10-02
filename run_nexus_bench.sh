#!/bin/bash
# NEXUS Hardware Qualification launcher (Linux/macOS) — developer fallback.
# Seller entry: NEXUS_Qualification.command (macOS) / NEXUS_Qualification.sh (Linux).
set -euo pipefail
cd "$(dirname "$0")"
LOG="nexus_launcher.log"

say() { echo "$1" | tee -a "$LOG"; }

: > "$LOG"
if ! command -v python3 >/dev/null 2>&1; then
  say "Python 3 was not found. Install Python 3.10+ from https://www.python.org/downloads/ and run again."
  exit 1
fi
say "Python: $(python3 --version)"

if [ ! -x ".venv/bin/python" ]; then
  say "Creating an isolated environment (.venv)..."
  if ! python3 -m venv .venv >>"$LOG" 2>&1; then
    say "Could not create a virtual environment (need python3-venv). See $LOG."
    exit 1
  fi
fi
say "Installing dependencies (visible errors are not hidden)..."
if ! ./.venv/bin/python -m pip install -e ".[dev]" >>"$LOG" 2>&1; then
  say "Dependency install failed — check connection, then run again. Details: $LOG"
  exit 1
fi
say "Starting the NEXUS qualification wizard..."
if ! ./.venv/bin/python -m nexus_bench.cli wizard 2>&1 | tee -a "$LOG"; then
  say "The wizard exited with an error. Details: $LOG"
  exit 1
fi
say "Done. Full log: $LOG"
