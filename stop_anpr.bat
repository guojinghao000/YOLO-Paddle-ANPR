@echo off
REM ---------------------------------------------------------------------------
REM  Stop both ANPR services started by start_anpr.bat.
REM  Kills whatever listens on the API / recognition ports, plus any shell whose
REM  command line mentions our two service scripts (closes their windows too).
REM ---------------------------------------------------------------------------
setlocal
set "API_PORT=8000"
set "REC_PORT=8100"

echo Stopping ANPR services ...

for %%P in (%REC_PORT% %API_PORT%) do (
  for /f "tokens=5" %%I in ('netstat -ano ^| findstr /r /c:"LISTENING" ^| findstr ":%%P "') do (
    if not "%%I"=="0" (
      echo   port %%P ^-^> PID %%I
      taskkill /F /T /PID %%I >nul 2>&1
    )
  )
)

powershell -NoProfile -ExecutionPolicy Bypass -Command "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'rec_service\.py|api_server\.py' } | ForEach-Object { Write-Host ('  closing ' + $_.Name + ' PID ' + $_.ProcessId); Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"

echo Done.
endlocal
