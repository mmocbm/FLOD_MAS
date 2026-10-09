@echo off
cd /d "%~dp0"
.venv\Scripts\python.exe tools\pipeline_lab.py %*
