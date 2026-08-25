@echo off
rem mylc0 UCI engine launcher.
rem
rem Point your chess GUI / engine tester at this file and pass the network:
rem     engine\mylc0.bat --weights C:\path\to\networks\gen_000500.mylc0
rem
rem Use an ABSOLUTE path for --weights: the GUI decides the working directory,
rem so a relative one may not resolve. With no --weights it falls back to
rem networks\latest.mylc0 relative to the current directory.
rem
rem The interpreter must be one that has torch + python-chess installed. If
rem "python" on the system PATH is not that one (for example because torch
rem lives in a conda environment), set MYLC0_PYTHON, either here or in the
rem environment:
rem     set "MYLC0_PYTHON=C:\Users\You\anaconda3\envs\mylc0\python.exe"
setlocal
set "SCRIPT_DIR=%~dp0"
if not defined MYLC0_PYTHON set "MYLC0_PYTHON=python"
"%MYLC0_PYTHON%" "%SCRIPT_DIR%mylc0.py" %*
