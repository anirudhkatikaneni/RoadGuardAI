# RoadGuardAI PowerShell Startup Script
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "        ROADGUARDAI: INTELLIGENT POTHOLE DETECTION" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
    Write-Host "[!] Virtual environment not found. Please setup .venv first." -ForegroundColor Yellow
    exit 1
}

Write-Host "[+] Launching web application on http://127.0.0.1:8080 ..." -ForegroundColor Green
Start-Process "http://127.0.0.1:8080"

& ".\.venv\Scripts\python.exe" -m uvicorn backend.main:app --host 127.0.0.1 --port 8080
