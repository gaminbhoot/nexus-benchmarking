@echo off
setlocal EnableExtensions
cd /d "%~dp0"
set "LOG=%~dp0nexus_launcher.log"
echo NEXUS Hardware Qualification > "%LOG%" 2>&1
echo ============================================================
echo        NEXUS HARDWARE QUALIFICATION
echo ============================================================
echo.
echo Log file for this run:
echo   %LOG%
echo.
if not exist "requirements.lock" goto MISSING_FILES
where python >nul 2>&1
if errorlevel 1 goto NO_PYTHON
python -c "import sys; assert sys.version_info >= (3,10), sys.version" >nul 2>&1
if errorlevel 1 goto OLD_PYTHON
python -c "import sys; assert sys.maxsize > 2**32, '32-bit'" >nul 2>&1
if errorlevel 1 goto PYTHON_32BIT
echo [1/3] Checking this computer...
python -m nexus_bench.cli doctor
if errorlevel 1 goto DOCTOR_FAILED
call :ENSURE_VENV
if errorlevel 1 goto SETUP_FAILED
echo.
echo [3/3] Starting the 10-minute qualification. Do not close this window.
".venv\Scripts\python.exe" -m nexus_bench.cli qualify
set "RESULT=%ERRORLEVEL%"
echo.
echo ==========================================================
echo Qualification exit code: %RESULT%
echo 0=PASS, 2=FAIL, 3=INCONCLUSIVE, 4=ABORTED. Report explains why.
echo Log: %LOG%
echo ==========================================================
echo.
pause
exit /b %RESULT%
:MISSING_FILES
echo WHAT HAPPENED: The benchmark files are not all here.
echo WHAT IT MEANS: You probably opened this file from INSIDE the zip
echo   without extracting it first.
echo WHAT TO DO: Right-click the zip, choose Extract All, open the new
echo   folder, and double-click NEXUS_Qualification.bat there.
echo.
pause
exit /b 1
:NO_PYTHON
echo WHAT HAPPENED: No Python found on this computer.
echo WHAT IT MEANS: The qualification cannot start without Python 3.10-3.14 (64-bit).
echo WHAT TO DO: Install it from https://www.python.org/downloads/
echo   (tick Add python.exe to PATH), then double-click this file again.
echo   Internet is needed ONCE for setup.
echo.
pause
exit /b 1
:OLD_PYTHON
echo WHAT HAPPENED: The found Python is too old.
python --version
echo WHAT TO DO: Install Python 3.10-3.14 (64-bit), then retry.
echo.
pause
exit /b 1
:PYTHON_32BIT
echo WHAT HAPPENED: The found Python is 32-bit.
echo WHAT TO DO: Install 64-bit Python (32-bit cannot load torch/CUDA), then retry.
echo.
pause
exit /b 1
:DOCTOR_FAILED
echo.
echo WHAT HAPPENED: The environment check above found a FAIL line.
echo WHAT IT MEANS: Setup cannot continue yet; nothing was tested.
echo WHAT TO DO: Fix the FAIL line(s) shown above, then double-click again.
echo.
pause
exit /b 1
:ENSURE_VENV
echo.
echo [2/3] Checking qualification runtime...
".venv\Scripts\python.exe" -c "import sys, torch, ultralytics, cv2, numpy, psutil, yaml, nexus_bench; assert sys.version_info >= (3,10)" >nul 2>&1
if errorlevel 1 goto VENV_REBUILD
echo Runtime already prepared and verified, skipping setup.
exit /b 0
:VENV_REBUILD
if exist ".venv" rmdir /s /q ".venv"
echo Existing runtime is incomplete or invalid. Recreating it...
echo One-time setup: downloading the pinned qualification runtime...
echo This downloads about 2 GB ONCE and can take 10+ minutes.
echo Do NOT close this window. Afterwards no internet is needed.
python -m venv ".venv" >>"%LOG%" 2>&1
if errorlevel 1 exit /b 1
".venv\Scripts\python.exe" -m pip install -q -r "requirements.lock" >>"%LOG%" 2>&1
if errorlevel 1 exit /b 1
".venv\Scripts\python.exe" -m pip install -q --no-deps -e . >>"%LOG%" 2>&1
if errorlevel 1 exit /b 1
".venv\Scripts\python.exe" -c "import sys, torch, ultralytics, cv2, numpy, psutil, yaml, nexus_bench; assert sys.version_info >= (3,10)" >>"%LOG%" 2>&1
if errorlevel 1 exit /b 1
echo Runtime verified.
exit /b 0
:SETUP_FAILED
echo.
echo WHAT HAPPENED: The runtime setup failed.
echo WHAT IT MEANS: Nothing was tested. This is usually no internet,
echo   a blocked connection, or less than ~6 GB free disk.
echo WHAT TO DO: Check internet and disk space, then double-click again
echo   (a broken install is discarded automatically and setup restarts clean).
echo.
echo Last log lines:
powershell -NoProfile -Command "Get-Content '%LOG%' -Tail 12" 2>nul
echo Full details: %LOG%
echo.
pause
exit /b 1
