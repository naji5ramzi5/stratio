@echo on
title StratoCrypto Dashboard ^& Telegram Bot
echo [🔄] Starting StratoCrypto Dashboard...
echo [🔄] Checking for Python installation...

:: Set Python Path
set PYTHON_EXE=

:: Check if py launcher exists
where py >nul 2>nul
if %ERRORLEVEL% equ 0 (
    set PYTHON_EXE=py
    goto run
)

:: Check if python exists in PATH
where python >nul 2>nul
if %ERRORLEVEL% equ 0 (
    set PYTHON_EXE=python
    goto run
)

:: Check common paths
if exist "%LocalAppData%\Programs\Python\Python313\python.exe" (
    set PYTHON_EXE="%LocalAppData%\Programs\Python\Python313\python.exe"
    goto run
)

if exist "%LocalAppData%\Programs\Python\Python38\python.exe" (
    set PYTHON_EXE="%LocalAppData%\Programs\Python\Python38\python.exe"
    goto run
)

echo [❌] Python was not found on your system!
echo Please download and install Python from: https://www.python.org/downloads/
echo Make sure to check the box "Add Python to PATH" during installation.
pause
exit /b

:run
echo [✅] Using Python: %PYTHON_EXE%
echo [🔄] Installing/Verifying requirements...
%PYTHON_EXE% -m pip install -r requirements.txt
if %ERRORLEVEL% neq 0 (
    echo [⚠️] Warning: Failed to install some requirements. Trying to start anyway...
)

echo [🚀] Starting server and bot...
%PYTHON_EXE% train.py
if %ERRORLEVEL% neq 0 (
    echo [❌] Program exited with an error. Please read the error message above.
    pause
)
:end
pause
