@echo off
REM NEXUS Hardware Qualification — seller entry point (Windows).
REM One-time setup uses the PINNED runtime (requirements.lock); afterwards fully offline.
cd /d "%~dp0"
set LOG=nexus_launcher.log

where python >nul 2>nul
if errorlevel 1 (
  echo NEXUS Qualification needs Python 3.10+, which was not found.
  echo Install it from https://www.python.org/downloads/ (tick "Add python.exe to PATH"),
  echo then double-click this file again. Internet is needed ONCE for setup.
  pause
  exit /b 1
)
if not exist ".venv\Scripts\python.exe" (
  echo Preparing qualification runtime (one-time setup, pinned versions)...
  python -m venv .venv >> "%LOG%" 2>&1
  if errorlevel 1 (
    echo Setup failed. Details: %LOG%
    pause
    exit /b 1
  )
  .venv\Scripts\python.exe -m pip install -q -r requirements.lock >> "%LOG%" 2>&1
  if errorlevel 1 (
    echo Setup failed (no internet or blocked install^). Details: %LOG%
    pause
    exit /b 1
  )
  .venv\Scripts\python.exe -m pip install -q --no-deps -e . >> "%LOG%" 2>&1
)
.venv\Scripts\python.exe -m nexus_bench.cli qualify
echo.
echo Log saved to %LOG%
pause
