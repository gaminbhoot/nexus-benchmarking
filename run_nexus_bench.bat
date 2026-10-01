@echo off
REM NEXUS Hardware Qualification launcher (Windows). Double-click to run.
cd /d "%~dp0"
set LOG=nexus_launcher.log

where python >nul 2>nul
if errorlevel 1 (
  echo Python 3 was not found. Install Python 3.10+ from https://www.python.org/downloads/
  echo (tick "Add python.exe to PATH" during install) and run this launcher again.
  exit /b 1
)
python --version

if not exist ".venv\Scripts\python.exe" (
  echo Creating an isolated environment (.venv)...
  python -m venv .venv
  if errorlevel 1 (
    echo Could not create a virtual environment. Reinstall Python with default options and retry.
    exit /b 1
  )
)
.venv\Scripts\python.exe -m pip install -q -e ".[dev]" >> "%LOG%" 2>&1
if errorlevel 1 (
  echo Dependency install had a problem. Check your internet connection and run again.
  echo See %LOG% for details.
  exit /b 1
)
echo Starting the NEXUS qualification wizard...
.venv\Scripts\python.exe -m nexus_bench.cli wizard >> "%LOG%" 2>&1
echo Done. The report folder and ZIP are described above. Full log: %LOG%
pause
