@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Hay chay run.bat mot lan truoc de cai dat moi truong.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -m pip install pyinstaller
".venv\Scripts\python.exe" -m PyInstaller --noconfirm --windowed --name ChArUcoCalibrator app.py
echo.
echo Xong. Chuong trinh nam o: dist\ChArUcoCalibrator\ChArUcoCalibrator.exe
echo Co the nen ca thu muc dist\ChArUcoCalibrator de mang sang may khac (khong can cai Python).
pause
