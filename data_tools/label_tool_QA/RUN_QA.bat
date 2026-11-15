@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Run SETUP.bat first.
    pause
    exit /b 1
)
set PYTHONIOENCODING=utf-8
echo Open http://127.0.0.1:8770 in your browser after the server starts.
echo Keep this window open. Press Ctrl+C to stop.
".venv\Scripts\python.exe" server.py %*
if errorlevel 1 pause
