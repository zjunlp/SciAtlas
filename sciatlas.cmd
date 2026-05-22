@echo off
setlocal

set "ROOT=%~dp0"
set "LOCAL_PY=%ROOT%.venv\Scripts\python.exe"

if exist "%LOCAL_PY%" (
  "%LOCAL_PY%" "%ROOT%run_sciatlas.py" %*
) else (
  python "%ROOT%run_sciatlas.py" %*
)
