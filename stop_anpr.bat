@echo off
REM ---------------------------------------------------------------------------
REM  Stop both ANPR services started by start_anpr.bat.
REM  Stops only Python processes running this checkout's service scripts.
REM ---------------------------------------------------------------------------
setlocal
set "ANPR_STOP_ROOT=%~dp0"

echo Stopping ANPR services ...

powershell -NoProfile -Command "$api = Join-Path $env:ANPR_STOP_ROOT 'code\api_server.py'; $rec = Join-Path $env:ANPR_STOP_ROOT 'code\rec_service.py'; Get-CimInstance Win32_Process -ErrorAction Stop | Where-Object { $_.Name -match '^python(w)?\.exe$' -and $_.CommandLine -and ($_.CommandLine.IndexOf($api, [StringComparison]::OrdinalIgnoreCase) -ge 0 -or $_.CommandLine.IndexOf($rec, [StringComparison]::OrdinalIgnoreCase) -ge 0) } | ForEach-Object { Write-Host ('  stopping PID ' + $_.ProcessId); Stop-Process -Id $_.ProcessId -Force -ErrorAction Stop }"
if errorlevel 1 (
  echo [ERROR] Could not stop the services. Close the ANPR service windows manually.
  pause
  exit /b 1
)

echo Done. Close any remaining ANPR service windows manually.
pause
endlocal
