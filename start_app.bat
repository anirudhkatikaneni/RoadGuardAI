@echo off
title RoadGuardAI - Startup Launcher
echo ============================================================
echo         ROADGUARDAI: INTELLIGENT POTHOLE DETECTION
echo ============================================================
echo.

cd /d "%~dp0"

echo [*] Starting RoadGuardAI FastAPI Server on http://127.0.0.1:8080 ...
start "" http://127.0.0.1:8080

.\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8080

pause
