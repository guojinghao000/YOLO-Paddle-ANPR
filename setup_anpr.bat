@echo off
setlocal
set "ROOT=%~dp0"
set "API_PY=%ROOT%.venv-api\Scripts\python.exe"
set "REC_PY=%ROOT%.venv-rec\Scripts\python.exe"
set "BOOTSTRAP=%ANPR_SETUP_PYTHON%"

echo ANPR setup: Windows x64, Python 3.11, NVIDIA GPU inference
echo This creates two local environments and downloads packages from the internet.
echo Dataset download is optional and is not needed for inference.
echo.

if not defined BOOTSTRAP for /f "delims=" %%I in ('py -3.11 -c "import sys; print(sys.executable)" 2^>nul') do set "BOOTSTRAP=%%I"
if not defined BOOTSTRAP goto missing_python
"%BOOTSTRAP%" -c "import struct,sys; sys.exit(0 if sys.version_info[:2] == (3,11) and struct.calcsize('P') == 8 else 1)" >nul 2>&1
if errorlevel 1 goto missing_python

if not exist "%API_PY%" (
  "%BOOTSTRAP%" -m venv "%ROOT%.venv-api"
  if errorlevel 1 goto failed
)
if not exist "%REC_PY%" (
  "%BOOTSTRAP%" -m venv "%ROOT%.venv-rec"
  if errorlevel 1 goto failed
)

echo [1/4] Installing API environment tools...
"%API_PY%" -m pip install --upgrade pip
if errorlevel 1 goto failed
echo [2/4] Installing CUDA PyTorch and API dependencies...
"%API_PY%" -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126
if errorlevel 1 goto failed
"%API_PY%" -m pip install -r "%ROOT%code\requirements-api-deploy.txt"
if errorlevel 1 goto failed
echo [3/4] Installing recognition environment tools...
"%REC_PY%" -m pip install --upgrade pip
if errorlevel 1 goto failed
echo [4/4] Installing CUDA PaddlePaddle and recognition dependencies...
"%REC_PY%" -m pip install paddlepaddle-gpu==3.3.1 --index-url https://www.paddlepaddle.org.cn/packages/stable/cu129/
if errorlevel 1 goto failed
"%REC_PY%" -m pip install -r "%ROOT%code\requirements-rec-deploy.txt"
if errorlevel 1 goto failed
"%API_PY%" -c "import torch,sys; print('PyTorch GPU:', torch.cuda.is_available()); sys.exit(0 if torch.cuda.is_available() else 1)"
if errorlevel 1 goto failed
"%REC_PY%" -c "import paddle,sys; ok = paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0; print('Paddle GPU:', ok); sys.exit(0 if ok else 1)"
if errorlevel 1 goto failed

echo.
echo Setup complete. Run start_anpr.bat and open http://127.0.0.1:8000
pause
exit /b 0

:missing_python
echo [ERROR] 64-bit Python 3.11 is required.
echo Install Python 3.11 with the Windows py launcher, or set ANPR_SETUP_PYTHON.
pause
exit /b 1

:failed
echo [ERROR] Setup failed. Check the error above and your network connection.
echo You can rerun setup_anpr.bat; existing environments will be reused.
pause
exit /b 1
