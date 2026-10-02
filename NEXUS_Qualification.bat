@echo off
REM NEXUS Hardware Qualification — seller entry point (Windows).
REM No choices, no files, no commands: runs the official qualification.
cd /d "%~dp0"
set LOG=nexus_launcher.log

where python >nul 2>nul
if errorlevel 1 (
  echo NEXUS Qualification needs Python 3.10+, which was not found.
  echo Install it from https://www.python.org/downloads/ (tick "Add python.exe to PATH"),
  echo then double-click this file again. No other setup is needed.
  pause
  exit /b 1
)
if not exist ".venv\Scripts\python.exe" (
  echo Preparing qualification runtime (one-time setup)...
  python -m venv .venv >> "%LOG%" 2>&1
  if errorlevel 1 (
    echo Setup failed. Details: %LOG%
    pause
    exit /b 1
  )
)
.venv\Scripts\python.exe -m pip install -q -e . >> "%LOG%" 2>&1
if errorlevel 1 (
  echo Setup failed (no internet or blocked install^). Details: %LOG%
  pause
  exit /b 1
)
.venv\Scripts\python.exe -m nexus_bench.cli qualify
echo.
echo Log saved to %LOG%
pause
