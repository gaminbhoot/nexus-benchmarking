#!/bin/bash
# NEXUS Hardware Qualification launcher (Linux/macOS).
# Double-click or:  bash run_nexus_bench.sh
set -e
cd "$(dirname "$0")"
LOG="nexus_launcher.log"
exec > >(tee "$LOG") 2>&1

if ! command -v python3 >/dev/null 2>&1; then
  echo "Python 3 was not found. Install Python 3.10+ from https://www.python.org/downloads/"
  echo "and run this launcher again."
  exit 1
fi
echo "Python: $(python3 --version)"

if [ ! -d ".venv" ]; then
  echo "Creating an isolated environment (.venv)..."
  python3 -m venv .venv || { echo "Could not create a virtual environment. Install python3-venv and retry."; exit 1; }
fi
./.venv/bin/python -m pip install -q -e ".[dev]" 2>&1 | tail -n 3 || {
  echo "Dependency install had a problem. Check your internet connection and run again."
  echo "Missing pieces are listed above in plain language by pip."
  exit 1
}
echo "Starting the NEXUS qualification wizard..."
./.venv/bin/python -m nexus_bench.cli wizard
echo "Done. The report folder and ZIP are described above. Full log: $LOG"
