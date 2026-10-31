@echo off
REM ---------------------------------------------------------------------------
REM  ANPR one-click launcher: split deployment
REM
REM    window 1  recognition service  (anpr-ocr env, GPU, fine-tuned PIR model)
REM    window 2  detection API        (ML env, GPU YOLO) -> calls window 1 by HTTP
REM
REM  The API refuses to serve plates when the recognition service is down, so we
REM  wait for it to report loaded=true before starting the API.
REM ---------------------------------------------------------------------------
setlocal

set "ROOT=%~dp0"
set "PY_REC=E:\CondaEnvs\anpr-ocr\python.exe"
set "PY_API=E:\CondaEnvs\ML\python.exe"
set "REC_MODEL_DIR=%ROOT%models\recognition"
set "REC_URL=http://127.0.0.1:8100"
set "API_PORT=8000"

echo ============================================================
echo  ANPR launcher (split deployment)
echo    recognition : %PY_REC%
echo    detection   : %PY_API%
echo    model dir   : %REC_MODEL_DIR%
echo ============================================================
echo.

if not exist "%PY_REC%" (
  echo [ERROR] anpr-ocr python not found: %PY_REC%
  pause & exit /b 1
)
if not exist "%PY_API%" (
  echo [ERROR] ML python not found: %PY_API%
  pause & exit /b 1
)
if not exist "%REC_MODEL_DIR%\inference.json" (
  echo [ERROR] fine-tuned PIR model not found in %REC_MODEL_DIR%
  echo         restore the model files from the repository, or export and copy them here.
  pause & exit /b 1
)

echo [1/3] starting recognition service (%REC_URL%) ...
start "ANPR recognition service" cmd /k call "%PY_REC%" "%ROOT%code\rec_service.py" --model-dir "%REC_MODEL_DIR%" --host 127.0.0.1 --port 8100

echo [2/3] waiting for the recogniser to load (up to 120 s) ...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ok=$false; for($i=0;$i -lt 60;$i++){ try{ $h=Invoke-RestMethod '%REC_URL%/health' -TimeoutSec 3; if($h.loaded){ Write-Host ('      ready: engine=' + $h.engine + ' device=' + $h.device + ' model=' + $h.model_dir); $ok=$true; break } }catch{}; Start-Sleep -Seconds 2 }; if(-not $ok){ exit 1 }"
if errorlevel 1 (
  echo [ERROR] the recognition service did not become ready.
  echo         check the other window for the reason ^(missing deps, busy GPU, bad model dir^).
  pause & exit /b 1
)

echo [3/3] starting detection API on http://127.0.0.1:%API_PORT% ...
start "ANPR API (YOLO)" cmd /k call "%PY_API%" "%ROOT%code\api_server.py" --host 127.0.0.1 --port %API_PORT% --rec-url %REC_URL%

echo.
echo   web UI      : http://127.0.0.1:%API_PORT%
echo   api health  : http://127.0.0.1:%API_PORT%/api/health
echo   rec health  : %REC_URL%/health
echo   stop        : close both windows, or run stop_anpr.bat
echo.
echo This window can be closed; the two service windows keep running.
endlocal
