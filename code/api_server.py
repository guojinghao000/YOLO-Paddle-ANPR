"""HTTP API server for YOLO-Paddle ANPR.

FastAPI app that loads the shared inference engine (anpr_engine.ANPREngine)
once at startup and exposes:

    GET  /api/health                     -> model / engine status
    POST /api/recognize                  -> multipart image upload
    POST /api/recognize/base64           -> JSON {image: base64, ...}
    GET  /api/demo                       -> recognise a bundled sample image
    /docs, /openapi.json                 -> interactive API docs (FastAPI)
    /                                     -> static web UI (code/web)

Run:
    conda activate YOLO-PADDLE-ANPR
    pip install -r requirements-web.txt
    cd code
    python api_server.py --host 0.0.0.0 --port 8000

Then open http://127.0.0.1:8000
"""
from __future__ import annotations

import argparse
import base64
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

# NOTE: anpr_engine imports ultralytics BEFORE paddleocr (torch DLL order).
from anpr_engine import ANPREngine, PROJECT_ROOT

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("anpr.api")

WEB_DIR = Path(__file__).resolve().parent / "web"
START_TIME = time.time()

DATA_URL_PREFIX = "data:image/jpeg;base64,"


# --------------------------------------------------------------------------- #
#  Settings (env overridable so uvicorn can launch the app module directly)
# --------------------------------------------------------------------------- #
def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


class Settings:
    host: str = os.environ.get("ANPR_HOST", "127.0.0.1")
    port: int = _env_int("ANPR_PORT", 8000)
    preload: bool = os.environ.get("ANPR_PRELOAD", "1") not in ("0", "false", "False")
    warmup: bool = os.environ.get("ANPR_WARMUP", "1") not in ("0", "false", "False")
    conf: float = float(os.environ.get("ANPR_CONF", "0.5"))
    yolo_weights: str | None = os.environ.get("ANPR_YOLO_WEIGHTS") or None
    rec_model_dir: str | None = os.environ.get("ANPR_REC_MODEL_DIR") or None
    ocr_version: str = os.environ.get("ANPR_OCR_VERSION", "PP-OCRv6_small")
    device: str | None = os.environ.get("ANPR_DEVICE","gpu") or None
    infer_engine: str | None = os.environ.get("ANPR_ENGINE","transformers") or None
    rec_url: str | None = os.environ.get("ANPR_REC_URL") or None
    mkldnn: bool = os.environ.get("ANPR_MKLDNN", "0") not in ("0", "false", "False")


# --------------------------------------------------------------------------- #
#  Engine lifecycle
# --------------------------------------------------------------------------- #
_settings = Settings()
_engine: ANPREngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the (heavy) models at startup so the first request is fast.

    If preload fails the server still starts; /api/recognize then answers 503
    with the reason until the environment is fixed (or ANPR_PRELOAD=0 defers
    loading entirely to the first request).
    """
    global _engine
    log.info("Preparing engine (preload=%s, engine=%s, device=%s, ocr=%s, rec=%s)",
             _settings.preload,
             _settings.infer_engine or "paddle",
             _settings.device or "auto", _settings.ocr_version,
             _settings.rec_url or "local module")
    _engine = ANPREngine(
        weights=_settings.yolo_weights,
        rec_model_dir=_settings.rec_model_dir,
        ocr_version=_settings.ocr_version,
        device=_settings.device,
        conf=_settings.conf,
        mkldnn=_settings.mkldnn,
        infer_engine=_settings.infer_engine,
        rec_url=_settings.rec_url,
        warmup=_settings.warmup,
        preload=_settings.preload,
        raise_errors=False,
    )
    summary = _engine.model_summary()
    log.info("Engine state: loaded=%s error=%s", summary["loaded"], summary["error"])
    for note in summary.get("notes", []):
        log.warning("engine note: %s", note)
    yield
    log.info("Shutting down.")


app = FastAPI(
    title="YOLO-Paddle ANPR API",
    description="License plate detection (YOLO) + recognition (PaddleOCR).",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],            # LAN / dev friendly; tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)


def _engine_or_503() -> ANPREngine:
    if _engine is None:
        raise HTTPException(status_code=503, detail="engine not initialised yet")
    if not _engine.loaded:
        raise HTTPException(
            status_code=503,
            detail={"error": "models not loaded",
                    "reason": _engine.error or "unknown",
                    "hint": "check ANPR_YOLO_WEIGHTS / ANPR_REC_MODEL_DIR "
                            "and the console log"})
    return _engine


def _clamp_conf(conf: float) -> float:
    return min(0.95, max(0.05, float(conf)))


def _frame_response(engine: ANPREngine, result: dict, source: str | None) -> dict:
    """Shape an engine result dict into the HTTP response payload."""
    plates = result["plates"]
    for p in plates:
        if "crop_b64" in p:
            p["crop"] = (DATA_URL_PREFIX + p["crop_b64"]
                         if p.get("crop_b64") else None)
            p.pop("crop_b64", None)
    return {
        "success": True,
        "error": None,
        "source": source,
        "count": len(plates),
        "image_size": result["image_size"],
        "plates": plates,
        "timings_ms": result["timings_ms"],
        "annotated_image": (
            DATA_URL_PREFIX + result["annotated_b64"]
            if result.get("annotated_b64") else None),
        "model": engine.model_summary(),
    }


# --------------------------------------------------------------------------- #
#  Endpoints
# --------------------------------------------------------------------------- #
@app.get("/api/health")
def health():
    eng = _engine
    return {
        "status": "ok",
        "uptime_s": round(time.time() - START_TIME, 1),
        # live probe of the recognition service (cached copies are used per request)
        "engine": eng.model_summary(refresh_remote=True) if eng else {"loaded": False,
                                                                     "error": "not initialised"},
        "api": "YOLO-Paddle-ANPR",
    }


@app.post("/api/recognize")
def recognize_upload(
    file: UploadFile = File(...),
    conf: float = Form(0.5),
    annotate: bool = Form(True),
    crops: bool = Form(False),
):
    """Recognise plates in an uploaded image (JPEG/PNG/WebP/BMP)."""
    engine = _engine_or_503()
    t_req = time.perf_counter()
    data = file.file.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty upload")
    try:
        result = engine.recognize_bytes(
            data, conf=_clamp_conf(conf),
            with_annotated=annotate, with_crops=crops)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - surface inference errors cleanly
        log.exception("recognize failed for %s", file.filename)
        raise HTTPException(status_code=500, detail=f"inference error: {exc}") from exc

    payload = _frame_response(engine, result, file.filename or None)
    payload["timings_ms"]["request_ms"] = round(
        (time.perf_counter() - t_req) * 1000, 1)
    return payload


class RecognizeBase64Request(BaseModel):
    """POST /api/recognize/base64 body.

    ``image`` accepts either a raw base64 string or a data URL
    (``data:image/...;base64,...``).
    """
    image: str
    conf: float = Field(0.5, ge=0.05, le=0.95)
    annotate: bool = True
    crops: bool = False


@app.post("/api/recognize/base64")
def recognize_base64(body: RecognizeBase64Request):
    engine = _engine_or_503()
    t_req = time.perf_counter()
    raw = body.image.strip()
    if "," in raw and raw.startswith("data:"):
        raw = raw.split(",", 1)[1]
    try:
        data = base64.b64decode(raw, validate=True)
    except Exception as exc:
        raise HTTPException(status_code=400,
                            detail=f"invalid base64 image: {exc}") from exc
    try:
        result = engine.recognize_bytes(
            data, conf=_clamp_conf(body.conf),
            with_annotated=body.annotate, with_crops=body.crops)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    payload = _frame_response(engine, result, "base64")
    payload["timings_ms"]["request_ms"] = round(
        (time.perf_counter() - t_req) * 1000, 1)
    return payload


@app.get("/api/demo")
def demo(conf: float = Query(0.5, ge=0.05, le=0.95),
         annotate: bool = Query(True),
         crops: bool = Query(False)):
    """Run the pipeline on a sample image stored next to the repo (if any)."""
    engine = _engine_or_503()
    samples = [p for p in (PROJECT_ROOT / "image.png",
                           PROJECT_ROOT / "image_result.jpg") if p.exists()]
    if not samples:
        raise HTTPException(status_code=404,
                            detail="no sample image found at project root")
    path = samples[0]
    t_req = time.perf_counter()
    result = engine.recognize_file(
        path, conf=_clamp_conf(conf),
        with_annotated=annotate, with_crops=crops)
    payload = _frame_response(engine, result, path.name)
    payload["timings_ms"]["request_ms"] = round(
        (time.perf_counter() - t_req) * 1000, 1)
    return payload


# --------------------------------------------------------------------------- #
#  Web UI + entry point
# --------------------------------------------------------------------------- #
if WEB_DIR.exists():
    app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")


def main():
    parser = argparse.ArgumentParser(description="YOLO-Paddle ANPR HTTP API")
    parser.add_argument("--host", default=_settings.host,
                        help="bind address (default: %(default)s)")
    parser.add_argument("--port", type=int, default=_settings.port,
                        help="bind port (default: %(default)s)")
    parser.add_argument("--engine", default=_settings.infer_engine or "paddle",
                        choices=["paddle", "transformers"],
                        help="PaddleOCR inference backend; 'transformers' runs on "
                             "GPU via torch (needs the transformers package and "
                             "safetensors-format models)")
    parser.add_argument("--rec-url", default=_settings.rec_url,
                        help="remote recognition service (code/rec_service.py), e.g. "
                             "http://127.0.0.1:8100; when set no local recogniser is "
                             "loaded and a down service is reported as an error")
    parser.add_argument("--device", default=_settings.device,
                        help="device for both models: cpu | gpu | gpu:0 | 0")
    parser.add_argument("--no-warmup", action="store_true",
                        help="skip the startup warm-up/inference validation")
    parser.add_argument("--workers", type=int, default=1,
                        help="uvicorn workers - keep 1: each worker loads a "
                             "full copy of both models")
    args = parser.parse_args()

    # tolerate `--host 127.0.0.1:8000` (host and port in one argument)
    if args.host and args.host.count(":") == 1:
        h, _, p = args.host.partition(":")
        if p.isdigit():
            log.warning("--host %s looks like host:port -> host=%s port=%s",
                        args.host, h, p)
            args.host, args.port = h, int(p)

    # push CLI choices into the module-level settings used by the lifespan hook
    _settings.device = args.device
    _settings.infer_engine = None if args.engine in ("paddle", "", None) else args.engine
    if args.rec_url:
        _settings.rec_url = args.rec_url.rstrip("/")
    if args.no_warmup:
        _settings.warmup = False

    import uvicorn
    log.info("Starting ANPR API on http://%s:%d  (web UI served at /, engine=%s, device=%s, "
             "rec=%s)",
             args.host, args.port,
             _settings.infer_engine or "paddle", _settings.device or "auto",
             _settings.rec_url or "local module")
    if args.workers and args.workers != 1:
        # multiprocess mode needs an import string, not an app instance
        uvicorn.run("api_server:app", host=args.host, port=args.port,
                    workers=args.workers, log_level="info")
    else:
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
