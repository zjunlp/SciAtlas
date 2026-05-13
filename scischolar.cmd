@echo off
setlocal

set "ROOT=%~dp0"
set "LOCAL_PY=%ROOT%.venv\Scripts\python.exe"

if exist "%LOCAL_PY%" (
  "%LOCAL_PY%" "%ROOT%run_scischolar.py" %*
) else (
  python "%ROOT%run_scischolar.py" %*
)
