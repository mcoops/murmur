@echo off
"%~dp0murmur.exe" %*
if %errorlevel% neq 0 (
    echo.
    echo murmur exited with error code %errorlevel% — scroll up to read the error.
    pause
)
