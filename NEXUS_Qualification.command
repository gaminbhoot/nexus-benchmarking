#!/bin/bash
# NEXUS Hardware Qualification — seller entry point (macOS, double-clickable).
# One-time setup uses the PINNED runtime (requirements.lock); afterwards fully offline.
set -euo pipefail
cd "$(dirname "$0")"
LOG="$HOME/Desktop/NEXUS_Qualification.log"
exec > >(tee "$LOG") 2>&1

echo "============================================================"
echo "       NEXUS HARDWARE QUALIFICATION"
echo "============================================================"
echo "Log file for this run: $LOG"

fail() { echo "WHAT HAPPENED: $1"; echo "WHAT TO DO: $2"; exit 1; }

[ -f "requirements.lock" ] || fail "The benchmark files are not all here." "Extract the WHOLE zip first, then re-run."
command -v python3 >/dev/null || fail "No Python found." "Install Python 3.10-3.14 from python.org."
python3 -c "import sys; assert sys.version_info >= (3,10)" 2>/dev/null || fail "Python is too old." "Install Python 3.10-3.14."
echo "[1/3] Checking this computer..."
python3 -m nexus_bench.cli doctor || fail "Environment check found a FAIL line (above)." "Fix it, then re-run."

if [ -x ".venv/bin/python" ] && ./.venv/bin/python -c "import sys, torch, ultralytics, cv2, numpy, psutil, yaml, nexus_bench; assert sys.version_info >= (3,10)" >>"$LOG" 2>&1; then
  echo "[2/3] Runtime already prepared and verified, skipping setup."
else
  [ -d ".venv" ] && { echo "Existing runtime is incomplete or invalid. Recreating it..."; rm -rf ".venv"; }
  echo "[2/3] One-time setup: downloading the pinned runtime (~2 GB once, do NOT close)..."
  python3 -m venv .venv >>"$LOG" 2>&1 || fail "venv creation failed." "Reinstall Python, then retry."
  ./.venv/bin/python -m pip install -q -r requirements.lock >>"$LOG" 2>&1 || fail "Runtime download failed." "Check internet and disk space, then re-run."
  ./.venv/bin/python -m pip install -q --no-deps -e . >>"$LOG" 2>&1 || fail "Package install failed." "See $LOG, then re-run."
  ./.venv/bin/python -c "import sys, torch, ultralytics, cv2, numpy, psutil, yaml, nexus_bench; assert sys.version_info >= (3,10)" >>"$LOG" 2>&1 || fail "Runtime verification failed." "See $LOG, then re-run."
  echo "Runtime verified."
fi
echo "[3/3] Starting the 10-minute qualification. Do not close this window."
set +e
./.venv/bin/python -m nexus_bench.cli qualify
RESULT=$?
set -e
echo "============================================================"
if [ "$RESULT" = "0" ]; then echo "RESULT: PASS (exit code 0)"; else echo "RESULT: NOT PASSED (exit code $RESULT). 0=PASS, 2=FAIL, 3=INCONCLUSIVE, 4=ABORTED."; fi
echo "Finished. Full log: $LOG"
exit "$RESULT"
