@echo off
setlocal DisableDelayedExpansion
chcp 65001 >nul
cd /d "%~dp0"
if not exist "%~dp0runtime\python.exe" goto incomplete
if not exist "%~dp0app\workbench.py" goto incomplete
"%~dp0runtime\python.exe" -X utf8 -B "%~dp0verify_manifest.py" > "%~dp0CHECK.log" 2>&1
if errorlevel 1 goto integrity_failed
"%~dp0runtime\python.exe" -X utf8 -B "%~dp0win7_environment.py" >> "%~dp0CHECK.log" 2>&1
if errorlevel 1 goto environment_failed
"%~dp0runtime\python.exe" -X utf8 -B -u "%~dp0app\workbench.py" --data-dir "%~dp0data" --self-test >> "%~dp0CHECK.log" 2>&1
set "CHECK_RESULT=%ERRORLEVEL%"
type "%~dp0CHECK.log"
echo.
if "%CHECK_RESULT%"=="0" (echo Offline self-test passed.) else (echo Offline self-test failed.)
echo Result saved to CHECK.log. No diagnosis reports or API keys are included.
pause
exit /b %CHECK_RESULT%
:integrity_failed
type "%~dp0CHECK.log"
echo Package verification failed. Open CHECK.log for the exact filename and Windows error.
echo Missing or changed file: extract a fresh ZIP. Access denied: check permissions and antivirus.
pause
exit /b 1
:environment_failed
type "%~dp0CHECK.log"
echo Win7 SP1 x64 requires KB2533623 or superseding updates and Universal CRT.
echo If Python cannot start, system components must be installed by your administrator.
pause
exit /b 1
:incomplete
echo Package incomplete. Extract the entire ZIP before running CHECK.bat.
pause
exit /b 1
