@echo off
cd /d "%~dp0"
echo ========================================================
echo   VASA - Vinh Accent Prosody Assessment (Mini Game)
echo   Mo trinh duyet tai: http://127.0.0.1:8000
echo ========================================================
"C:\ProgramData\anaconda3\python.exe" -m uvicorn src.backend.main:app --host 127.0.0.1 --port 8000 --reload
pause
