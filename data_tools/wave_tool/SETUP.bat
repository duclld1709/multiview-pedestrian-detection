@echo off
setlocal
cd /d "%~dp0"
python --version >nul 2>&1
if errorlevel 1 (
    echo Python was not found. Install Python 3.10 or newer and enable Add Python to PATH.
    pause
    exit /b 1
)
python -m venv .venv
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto failed
echo Setup complete. Run RUN_WAVE_TOOL.bat to start.
pause
exit /b 0
:failed
echo Setup failed. See the error above.
pause
exit /b 1
