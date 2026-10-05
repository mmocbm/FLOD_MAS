@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" goto :missing
".venv\Scripts\python.exe" main_1366.py
exit /b %ERRORLEVEL%
:missing
echo Project Python environment is missing.
echo Create it with: py -3.12 -m venv .venv
echo Then install: .venv\Scripts\python.exe -m pip install -r requirements.txt
pause
exit /b 1
