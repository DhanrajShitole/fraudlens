@echo off
title FraudLens Local API Server
echo ========================================================
echo Starting FraudLens FastAPI Backend on http://127.0.0.1:8000
echo Keep this window OPEN while testing the dashboard!
echo ========================================================
cd /d "C:\Users\DHANRAJ\fraudlens"
"C:\Users\DHANRAJ\AppData\Local\Programs\Python\Python313\python.exe" -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
pause

