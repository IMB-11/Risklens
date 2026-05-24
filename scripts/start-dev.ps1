$Root = Split-Path -Parent $PSScriptRoot
Write-Host "Backend URL: http://127.0.0.1:8000"
Write-Host "Frontend URL: http://127.0.0.1:5173"
Write-Host "Health Check: http://127.0.0.1:8000/api/health"
Start-Process cmd -ArgumentList "/k", (Join-Path $Root "start_backend.bat")
Start-Process cmd -ArgumentList "/k", (Join-Path $Root "start_frontend.bat")
