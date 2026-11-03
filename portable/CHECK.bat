@echo off
setlocal
cd /d "%~dp0"
if not exist "%~dp0runtime\python.exe" goto incomplete
if not exist "%~dp0CHECK_INSTALLATION.py" goto incomplete
"%~dp0runtime\python.exe" -X utf8 -B -u "%~dp0CHECK_INSTALLATION.py" --root "%~dp0."
set "CHECK_RESULT=%ERRORLEVEL%"
echo.
if "%CHECK_RESULT%"=="0" (echo Offline self-test passed.) else (echo Offline self-test failed. See data\diagnostics\installation-check.json)
if /i not "%~1"=="--no-pause" pause
exit /b %CHECK_RESULT%
:incomplete
echo Package incomplete. Extract the entire ZIP before running CHECK.bat.
if /i not "%~1"=="--no-pause" pause
exit /b 1
