@echo off
setlocal
cd /d "%~dp0"
chcp 65001 >nul
set PYTHONUTF8=1

where python >nul 2>nul
if errorlevel 1 (
  echo Python not found. Please install Python 3.10+.
  pause
  exit /b 1
)

if not exist ".venv\Scripts\activate.bat" (
  echo Creating Python virtual environment...
  python -m venv .venv
)

call ".venv\Scripts\activate.bat"
python -m pip install --upgrade pip
pip install -r requirements.txt

echo Backend URL: http://127.0.0.1:8000
echo Health Check: http://127.0.0.1:8000/api/health
uvicorn backend.api.main:app --host 127.0.0.1 --port 8000 --reload
