@echo off
REM Kuchiko - Windows Setup Script (Batch)
REM This script launches the PowerShell setup script

echo.
echo ========================================
echo    Kuchiko Setup Script (Windows)
echo ========================================
echo.

REM Check if PowerShell is available
where powershell >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Error: PowerShell is required but not found.
    echo Please run start.ps1 directly or install PowerShell.
    pause
    exit /b 1
)

REM Run the PowerShell script
powershell -ExecutionPolicy Bypass -File "%~dp0start.ps1"

pause
