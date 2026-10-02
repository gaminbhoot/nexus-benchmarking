#!/bin/bash
# NEXUS Hardware Qualification — seller entry point (Linux).
# One-time setup uses the PINNED runtime (requirements.lock); afterwards fully offline.
set -euo pipefail
cd "$(dirname "$0")"
LOG="$PWD/nexus_launcher.log"
: > "$LOG"

echo "============================================================"
echo "       NEXUS HARDWARE QUALIFICATION"
echo "============================================================"
echo "Log file for this run: $LOG"

if [ ! -f "requirements.lock" ]; then
  echo "WHAT HAPPENED: The benchmark files are not all here."
  echo "WHAT TO DO: extract the WHOLE zip first, then re-run from the new folder."
  exit 1
fi
if ! command -v python3 >/dev/null 2>&1; then
  echo "WHAT HAPPENED: No Python found. Install Python 3.10-3.13 (64-bit), then re-run."
  exit 1
fi
echo "[1/3] Checking this computer..."
python3 -m nexus_bench.cli doctor 2>&1 | tee -a "$LOG"
if [ ! -x ".venv/bin/python" ]; then
  echo "[2/3] One-time setup: downloading the pinned runtime (~2 GB once, do NOT close)..."
  python3 -m venv .venv >>"$LOG" 2>&1
  ./.venv/bin/python -m pip install -q -r requirements.lock >>"$LOG" 2>&1
  ./.venv/bin/python -m pip install -q --no-deps -e . >>"$LOG" 2>&1
else
  echo "[2/3] Runtime already prepared, skipping setup."
fi
echo "[3/3] Starting the 10-minute qualification. Do not close this window."
./.venv/bin/python -m nexus_bench.cli qualify 2>&1 | tee -a "$LOG"
echo "Finished. Full log: $LOG"
