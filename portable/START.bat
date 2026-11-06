@echo off
setlocal
cd /d "%~dp0"
if not exist "%~dp0runtime\python.exe" goto incomplete
if not exist "%~dp0app\workbench.py" goto incomplete
"%~dp0runtime\python.exe" -X utf8 -B -u "%~dp0app\workbench.py" --data-dir "%~dp0data" --open-browser
if errorlevel 1 pause
exit /b
:incomplete
echo Package incomplete. Extract the entire ZIP before running START.bat.
pause
exit /b 1
