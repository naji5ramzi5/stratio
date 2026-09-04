@echo off
cd /d "C:\Users\IRAQ SOFT\Desktop\stratocrypto\crypto"
start /B py -3.14 dashboard_server.py >nul 2>&1
timeout /t 3 >nul
start /B py -3.14 advanced_bot.py >nul 2>&1
start /B py -3.14 news_bot.py >nul 2>&1
echo Dashboard + bot system started
echo    Dashboard: http://127.0.0.1:8080
echo    Log: crypto\advanced_bot.log
pause
