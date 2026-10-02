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

fail() { echo "WHAT HAPPENED: $1"; echo "WHAT TO DO: $2"; echo "Details: $LOG"; exit 1; }

[ -f "requirements.lock" ] || fail "The benchmark files are not all here." "Extract the WHOLE zip first, then re-run."
command -v python3 >/dev/null || fail "No Python found." "Install Python 3.10-3.14 (64-bit), then re-run."
python3 -c "import sys; assert sys.version_info >= (3,10)" 2>/dev/null || fail "Python is too old ($(python3 --version))." "Install Python 3.10-3.14 (64-bit)."
python3 -c "import sys; assert sys.maxsize > 2**32" 2>/dev/null || fail "Python is 32-bit." "Install 64-bit Python."

echo "[1/3] Checking this computer..."
python3 -m nexus_bench.cli doctor 2>&1 | tee -a "$LOG" || fail "Environment check found a FAIL line (above)." "Fix it, then re-run."

ensure_venv() {
  if [ -x ".venv/bin/python" ] && ./.venv/bin/python -c "import sys, torch, ultralytics, cv2, numpy, psutil, yaml, nexus_bench; assert sys.version_info >= (3,10)" >>"$LOG" 2>&1; then
    echo "[2/3] Runtime already prepared and verified, skipping setup."
    return 0
  fi
  [ -d ".venv" ] && { echo "Existing runtime is incomplete or invalid. Recreating it..."; rm -rf ".venv"; }
  echo "[2/3] One-time setup: downloading the pinned runtime (~2 GB once, do NOT close)..."
  python3 -m venv .venv >>"$LOG" 2>&1 || return 1
  ./.venv/bin/python -m pip install -q -r requirements.lock >>"$LOG" 2>&1 || return 1
  ./.venv/bin/python -m pip install -q --no-deps -e . >>"$LOG" 2>&1 || return 1
  ./.venv/bin/python -c "import sys, torch, ultralytics, cv2, numpy, psutil, yaml, nexus_bench; assert sys.version_info >= (3,10)" >>"$LOG" 2>&1 || return 1
  echo "Runtime verified."
}
ensure_venv || { echo; tail -n 12 "$LOG"; fail "Runtime setup failed (see last log lines above)." "Check internet and ~6 GB free disk, then re-run (broken installs are discarded automatically)."; }

echo "[3/3] Starting the 10-minute qualification. Do not close this window."
set +e
./.venv/bin/python -m nexus_bench.cli qualify 2>&1 | tee -a "$LOG"
RESULT=${PIPESTATUS[0]}
set -e
echo "============================================================"
if [ "$RESULT" = "0" ]; then echo "RESULT: PASS (exit code 0)"; else echo "RESULT: NOT PASSED (exit code $RESULT). 0=PASS, 2=FAIL, 3=INCONCLUSIVE, 4=ABORTED."; fi
echo "Full log: $LOG"
exit "$RESULT"
