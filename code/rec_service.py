"""Standalone plate-recognition service (runs in the torch-free anpr-ocr env).

Why a separate process: the fine-tuned model is a PIR export, which only the
PaddleX ``paddle`` engine can load, and ``import paddle`` is impossible in the
ML env (CUDA torch vs CUDA paddle DLL conflict).  So detection stays in the ML
env (YOLO + CUDA torch) and recognition runs here, where paddle works.

Endpoints
---------
GET  /health   -> model/engine/device status, load time, request counters
POST /api/rec  -> {"images": ["<base64 jpg>", ...], "batch": true}
                  -> {"results": [{"text": str, "score": float, "index": int}],
                      "count": int, "elapsed_ms": float}

Run
---
    E:\\CondaEnvs\\anpr-ocr\\python.exe rec_service.py \
        --model-dir code\\output_plate\\inference --port 8100

Env overrides: REC_MODEL_DIR, REC_MODEL_NAME, REC_ENGINE, REC_DEVICE,
REC_HOST, REC_PORT, REC_MKLDNN, REC_WARMUP.

The ANPR engine talks to this service when ``ANPR_REC_URL`` is set; when it is
down the API reports the failure instead of silently falling back to the
official (much weaker) model.
"""
from __future__ import annotations

import argparse
import base64
import logging
import os
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("anpr.rec_service")

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent


class RecModel:
    """Lazily-built singleton around paddleocr.TextRecognition."""

    def __init__(self, model_dir: str | None, model_name: str, engine: str | None,
                 device: str | None, mkldnn: bool, warmup: bool = True):
        self.model_dir = Path(model_dir).resolve() if model_dir else None
        self.model_name = model_name
        self.engine = engine
        self.device = device
        self.mkldnn = mkldnn
        self.warmup = warmup
        self.loaded = False
        self.error: str | None = None
        self.load_seconds = 0.0
        self.warmup_ms = 0.0
        self.requests = 0
        self.images = 0
        self.last_ms = 0.0
        self._lock = threading.RLock()
        self._rec = None

    # ------------------------------------------------------------- lifecycle
    def load(self) -> None:
        with self._lock:
            if self.loaded:
                return
            t0 = time.perf_counter()
            try:
                from paddleocr import TextRecognition   # lazy, paddle-only
                kwargs = dict(model_name=self.model_name, enable_mkldnn=self.mkldnn)
                if self.model_dir is not None:
                    if not (self.model_dir / "inference.json").exists():
                        raise FileNotFoundError(
                            f"no inference.json in {self.model_dir} (export it with "
                            f"`train_ocr.py --export-only`)")
                    kwargs["model_dir"] = str(self.model_dir)
                if self.engine:
                    kwargs["engine"] = self.engine
                if self.device:
                    kwargs["device"] = self.device
                log.info("Loading TextRecognition (name=%s, dir=%s, engine=%s, device=%s) ...",
                         self.model_name, self.model_dir or "official",
                         self.engine or "paddle", self.device or "auto")
                self._rec = TextRecognition(**kwargs)
                if self.warmup:
                    probe = np.full((48, 320, 3), 240, dtype=np.uint8)
                    cv2.putText(probe, "ANPR", (8, 34), cv2.FONT_HERSHEY_SIMPLEX,
                                1.0, (20, 20, 20), 2)
                    t1 = time.perf_counter()
                    list(self._rec.predict(probe))
                    self.warmup_ms = round((time.perf_counter() - t1) * 1000, 1)
                self.load_seconds = round(time.perf_counter() - t0, 2)
                self.loaded = True
                self.error = None
                log.info("Recogniser ready in %.1fs (warm-up %.1f ms)",
                         self.load_seconds, self.warmup_ms)
            except Exception as exc:                     # noqa: BLE001
                self.loaded = False
                self.error = f"{type(exc).__name__}: {exc}"
                log.error("Recogniser failed to load: %s", self.error)

    def ensure_loaded(self) -> None:
        if not self.loaded:
            raise RuntimeError(self.error or "recogniser not loaded")

    # ------------------------------------------------------------- inference
    def recognize_many(self, images: list[np.ndarray]) -> tuple[list[dict], float]:
        """Recognise a batch of BGR crops; returns (results, elapsed_ms)."""
        results: list[dict] = []
        t0 = time.perf_counter()
        with self._lock:                     # PaddleX predictors are not thread-safe
            self.ensure_loaded()
            for i, img in enumerate(images):
                t_img = time.perf_counter()
                try:
                    out = list(self._rec.predict(img))
                    res = out[0] if out else {}
                    text = res.get("rec_text", "")
                    if isinstance(text, (list, tuple)):
                        text = "".join(str(t) for t in text)
                    score = res.get("rec_score", 0.0)
                    if isinstance(score, (list, tuple, np.ndarray)):
                        score = float(np.mean(score)) if len(score) else 0.0
                    score = float(score)
                    if not np.isfinite(score):
                        score = 0.0
                    results.append({"index": i, "text": str(text), "score": score,
                                    "ms": round((time.perf_counter() - t_img) * 1000, 1)})
                except Exception as exc:                 # noqa: BLE001
                    log.warning("image %d failed: %s: %s", i, type(exc).__name__, exc)
                    results.append({"index": i, "text": "", "score": 0.0,
                                    "error": f"{type(exc).__name__}: {exc}",
                                    "ms": round((time.perf_counter() - t_img) * 1000, 1)})
        elapsed = round((time.perf_counter() - t0) * 1000, 1)
        self.requests += 1
        self.images += len(images)
        self.last_ms = elapsed
        return results, elapsed

    def summary(self) -> dict:
        return {
            "loaded": self.loaded,
            "error": self.error,
            "module": "TextRecognition",
            "model_name": self.model_name,
            "model_dir": str(self.model_dir) if self.model_dir else None,
            "engine": self.engine or "paddle",
            "device": self.device or "auto",
            "enable_mkldnn": self.mkldnn,
            "load_seconds": self.load_seconds,
            "warmup_ms": self.warmup_ms,
            "requests": self.requests,
            "images": self.images,
            "last_ms": self.last_ms,
        }


def decode_b64_image(raw: str) -> np.ndarray:
    if "," in raw and raw.lstrip().startswith("data:"):
        raw = raw.split(",", 1)[1]
    arr = np.frombuffer(base64.b64decode(raw, validate=True), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("could not decode image bytes")
    return img


class RecRequest(BaseModel):
    """POST /api/rec body: base64 JPEG/PNG crops, one entry per plate.

    Defined at module level on purpose: with ``from __future__ import
    annotations`` a model defined inside a function cannot be resolved by
    FastAPI, which then reads the body as a query parameter (HTTP 422).
    """
    images: list[str] = Field(..., min_length=1)
    batch: bool = True


def build_app(model: RecModel):
    app = FastAPI(title="ANPR recognition service",
                  description="Plate recognition (fine-tuned PP-OCRv6_small_rec, rec-only)",
                  version="1.0.0")
    started = time.time()

    @app.get("/health")
    def health():
        s = model.summary()
        s.update({"status": "ok" if model.loaded else "degraded",
                  "uptime_s": round(time.time() - started, 1),
                  "api": "ANPR recognition service"})
        return s

    @app.post("/api/rec")
    def rec(body: RecRequest):
        if not model.loaded:
            raise HTTPException(status_code=503,
                                detail={"error": "recogniser not loaded",
                                        "reason": model.error or "unknown"})
        crops: list[np.ndarray] = []
        for i, raw in enumerate(body.images):
            try:
                crops.append(decode_b64_image(raw))
            except Exception as exc:                     # noqa: BLE001
                raise HTTPException(
                    status_code=400,
                    detail=f"image #{i} is not a valid base64 image: "
                           f"{type(exc).__name__}: {exc}") from exc
        results, elapsed = model.recognize_many(crops)
        return {"results": results, "count": len(results), "elapsed_ms": elapsed,
                "model": {"model_name": model.model_name,
                          "model_dir": str(model.model_dir) if model.model_dir else None,
                          "engine": model.engine or "paddle",
                          "device": model.device or "auto"}}

    return app


def main():
    ap = argparse.ArgumentParser(description="ANPR plate recognition service")
    ap.add_argument("--model-dir", default=os.environ.get("REC_MODEL_DIR")
                    or str(BASE_DIR / "output_plate" / "inference"))
    ap.add_argument("--model-name", default=os.environ.get("REC_MODEL_NAME")
                    or "PP-OCRv6_small_rec")
    ap.add_argument("--engine", default=os.environ.get("REC_ENGINE") or "paddle",
                    choices=["paddle", "transformers"])
    ap.add_argument("--device", default=os.environ.get("REC_DEVICE") or "gpu:0")
    ap.add_argument("--host", default=os.environ.get("REC_HOST") or "127.0.0.1")
    ap.add_argument("--port", type=int, default=int(os.environ.get("REC_PORT") or 8100))
    ap.add_argument("--mkldnn", action="store_true",
                    help="enable oneDNN (crashes paddle CPU inference - keep off)")
    ap.add_argument("--no-warmup", action="store_true")
    args = ap.parse_args()

    print(f"python   : {os.sys.executable}")
    print(f"model dir: {args.model_dir}")
    model = RecModel(args.model_dir, args.model_name,
                     None if args.engine in ("paddle", "") else args.engine,
                     args.device, args.mkldnn, warmup=not args.no_warmup)
    model.load()                       # load before serving: /health tells the truth
    app = build_app(model)

    import uvicorn
    log.info("Serving recognition on http://%s:%d  (model_dir=%s, device=%s)",
             args.host, args.port, model.model_dir or "official", args.device)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info", workers=1)


if __name__ == "__main__":
    main()
