@echo off
cd /d "%~dp0"
python apply_pose.py %*
echo.
pause
