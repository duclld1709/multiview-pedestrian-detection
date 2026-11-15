@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Run SETUP.bat first.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" wave_tool.py %*
if errorlevel 1 (
    echo Wave Tool stopped with an error. See the message above.
    pause
    exit /b 1
)
