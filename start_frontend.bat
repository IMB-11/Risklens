@echo off
setlocal
cd /d "%~dp0frontend"

where node >nul 2>nul
if errorlevel 1 (
  echo Node.js not found. Please install Node.js 18+.
  pause
  exit /b 1
)

where npm >nul 2>nul
if errorlevel 1 (
  echo npm not found. Please install npm.
  pause
  exit /b 1
)

if not exist "node_modules" (
  echo Installing frontend dependencies...
  npm install
)

echo Frontend URL: http://127.0.0.1:5173
npm run dev

