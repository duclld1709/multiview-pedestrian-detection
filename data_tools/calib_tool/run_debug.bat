@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Hay chay run.bat truoc de cai dat.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" app.py
pause
