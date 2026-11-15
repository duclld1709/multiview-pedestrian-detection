@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
title ChArUco Calibrator

set "PY="
where py >nul 2>nul && set "PY=py -3"
if not defined PY (
    where python >nul 2>nul && set "PY=python"
)
if not defined PY (
    echo.
    echo [LOI] Chua cai Python tren may.
    echo Tai Python 3.10 - 3.13 tai https://www.python.org/downloads/windows/
    echo Khi cai, nho tick "Add python.exe to PATH".
    echo.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo Lan dau chay: dang tao moi truong va cai thu vien, mat vai phut...
    %PY% -m venv .venv
    if errorlevel 1 (
        echo [LOI] Khong tao duoc moi truong Python.
        pause
        exit /b 1
    )
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo [LOI] Cai thu vien that bai. Kiem tra ket noi mang roi chay lai.
        rmdir /s /q .venv
        pause
        exit /b 1
    )
)

start "" ".venv\Scripts\pythonw.exe" app.py
echo Dang mo chuong trinh... lan dau co the mat 10-20 giay.
echo Neu co loi, se hien hop thoai va ghi vao charuco_error.log
endlocal
