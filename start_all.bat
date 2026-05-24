@echo off
setlocal
cd /d "%~dp0"

echo Backend URL: http://127.0.0.1:8000
echo Frontend URL: http://127.0.0.1:5173
echo Health Check: http://127.0.0.1:8000/api/health

start "AI Risk Terminal Backend" cmd /k "%~dp0start_backend.bat"
start "AI Risk Terminal Frontend" cmd /k "%~dp0start_frontend.bat"

