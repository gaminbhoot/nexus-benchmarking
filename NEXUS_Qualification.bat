@echo off
REM NEXUS Hardware Qualification — seller entry point (Windows).
REM One-time setup uses the PINNED runtime (requirements.lock); afterwards fully offline.
cd /d "%~dp0"
set LOG=%CD%\nexus_launcher.log
echo NEXUS Hardware Qualification > "%LOG%" 2>&1

echo ============================================================
echo        NEXUS HARDWARE QUALIFICATION
echo ============================================================
echo.
echo Log file for this run:
echo   %LOG%
echo.

if not exist "requirements.lock" (
  echo WHAT HAPPENED: The benchmark files are not all here.
  echo WHAT IT MEANS: You probably opened this file from INSIDE the zip
  echo   without extracting it first.
  echo WHAT TO DO: Right-click the zip ^> "Extract All", open the new folder,
  echo   and double-click NEXUS_Qualification.bat there.
  echo See also: %LOG%
  pause
  exit /b 1
)

where python >nul 2>nul
if errorlevel 1 (
  echo WHAT HAPPENED: No Python found on this computer.
  echo WHAT IT MEANS: The qualification cannot start without Python 3.10-3.13 (64-bit).
  echo WHAT TO DO: Install it from https://www.python.org/downloads/
  echo   (tick "Add python.exe to PATH"), then double-click this file again.
  echo   Internet is needed ONCE for setup.
  pause
  exit /b 1
)

echo [1/3] Checking this computer...
python -m nexus_bench.cli doctor
if errorlevel 1 (
  echo.
  echo WHAT HAPPENED: The environment check above found a FAIL line.
  echo WHAT IT MEANS: Setup cannot continue yet; nothing was tested.
  echo WHAT TO DO: Fix the FAIL line(s) shown above, then double-click again.
  echo Full details: %LOG%
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo.
  echo [2/3] One-time setup: downloading the pinned qualification runtime...
  echo        This downloads about 2 GB ONCE and can take 10+ minutes.
  echo        Do NOT close this window. Afterwards no internet is needed.
  python -m venv .venv >> "%LOG%" 2>&1
  if errorlevel 1 (
    echo WHAT HAPPENED: Could not create the isolated runtime.
    echo WHAT TO DO: Reinstall Python with default options, then retry. Details: %LOG%
    pause
    exit /b 1
  )
  .venv\Scripts\python.exe -m pip install -q -r requirements.lock >> "%LOG%" 2>&1
  if errorlevel 1 (
    echo WHAT HAPPENED: The runtime download/install failed.
    echo WHAT IT MEANS: Nothing was tested. This is usually no internet,
    echo   a blocked connection, or less than ~6 GB free disk.
    echo WHAT TO DO: Check internet and disk space, then double-click again
    echo   (setup resumes where it stopped). Details: %LOG%
    pause
    exit /b 1
  )
  .venv\Scripts\python.exe -m pip install -q --no-deps -e . >> "%LOG%" 2>&1
) else (
  echo.
  echo [2/3] Runtime already prepared, skipping setup.
)

echo.
echo [3/3] Starting the 10-minute qualification. Do not close this window.
.venv\Scripts\python.exe -m nexus_bench.cli qualify
echo.
echo Finished. The report folder/ZIP locations are shown above.
echo Full log: %LOG%
pause
