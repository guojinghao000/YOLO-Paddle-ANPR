"""Core inference engine for YOLO-Paddle ANPR (plate detection + recognition).

Shared by two front-ends:
  * CLI       -> recognize.py (keeps the old argparse interface)
  * HTTP API  -> api_server.py (FastAPI) + web UI in ./web

Notes that matter here (also documented in AGENTS.md):

  * IMPORT ORDER: ultralytics (torch) MUST be imported before paddleocr.
    Importing paddle first pollutes the DLL search path and crashes torch.
    Keep the two imports right below this docstring in this order.
  * Recognition uses the standalone ``TextRecognition`` module, NOT the
    ``PaddleOCR`` pipeline: YOLO already localises the plate, so the pipeline's
    own detection pass (PP-OCRv6_medium_det) only re-detected the same text
    line. Measured on a real crop: 87 ms for det+rec vs 20 ms for rec-only,
    with identical text. Output is still filtered through the plate character
    set (province abbreviations, letters and digits).
  * Two deployment shapes: all-in-one (YOLO + the local ``TextRecognition``
    module in one process) or split (YOLO here, recognition in
    ``code/rec_service.py`` under the torch-free anpr-ocr env, where the
    fine-tuned PIR model can be loaded by the ``paddle`` engine). Set
    ``ANPR_REC_URL`` / ``--rec-url`` for the split shape. There is deliberately
    no silent fallback: if that service is down the API reports the failure.
  * Models are constructed once per process (singleton) and all inference is
    serialised with an RLock - PaddleX predictors are not thread-safe.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import threading
import time
import unicodedata
import urllib.error
import urllib.request
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------- import order
from ultralytics import YOLO            # torch first  (noqa: E402)
# NOTE: paddleocr is imported lazily (see _import_text_recognition). Importing
# paddle after torch is mandatory (DLL search path), and keeping it lazy means
# the paddlex 'transformers' engine (GPU via torch) never touches paddle at all.
# ------------------------------------------------------------------------------


def _import_text_recognition():
    """Deliberate lazy import: only the paddle engine ever imports paddleocr."""
    from paddleocr import TextRecognition   # noqa: PLC0415
    return TextRecognition


log = logging.getLogger("anpr.engine")

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

# PaddleOCR rec model names, same mapping as the original recognize.py.
REC_MODEL_NAMES = {
    "PP-OCRv3": "PP-OCRv3_mobile_rec",
    "PP-OCRv4": "PP-OCRv4_mobile_rec",
    "PP-OCRv5": "PP-OCRv5_server_rec",
    "PP-OCRv6": "PP-OCRv6_medium_rec",
    "PP-OCRv6_small": "PP-OCRv6_small_rec",
}

# Characters a Chinese plate can contain; OCR output is filtered through this set:
# province abbreviations + "警/学", letters A-Z without I/O, digits 0-9.
PROVINCE_CHARS = "京津冀晋蒙辽吉黑沪苏浙皖闽赣鲁豫鄂湘粤桂琼渝川贵云藏陕甘青宁新警学"
DICT_LETTERS = "ABCDEFGHJKLMNPQRSTUVWXYZ"
DIGITS = "0123456789"
PLATE_ALLOWED = set(PROVINCE_CHARS) | set(DICT_LETTERS) | set(DIGITS)

_FONT_PATHS = (
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/msyhbd.ttc",
    "C:/Windows/Fonts/simhei.ttf",
    "C:/Windows/Fonts/Deng.ttf",
)


# --------------------------------------------------------------------------- #
#  Pure helpers
# --------------------------------------------------------------------------- #
def normalize_plate(raw: str) -> tuple[str, bool]:
    """Filter an OCR string down to valid plate characters and validate shape.

    Returns (plate, is_valid).  ``is_valid`` means the string starts with a
    province char, is followed by a dict letter and has >= 5 chars overall -
    the standard mainland layout is province + letter + 5..6 alnum.
    """
    if not raw:
        return "", False
    s = unicodedata.normalize("NFKC", raw).upper()
    keep = [ch for ch in s if ch in PLATE_ALLOWED]
    s = "".join(keep)
    if len(s) < 5:
        return s, False
    if s[0] not in PROVINCE_CHARS:
        return s, False
    if s[1] not in DICT_LETTERS:
        return s, False
    return s, True


def crop_plate(img: np.ndarray, box, pad_ratio: float = 0.05) -> np.ndarray:
    """Crop the plate region with padding, clipped to the image bounds."""
    x1, y1, x2, y2 = (int(v) for v in box)
    h, w = img.shape[:2]
    pw = int((x2 - x1) * pad_ratio)
    ph = int((y2 - y1) * pad_ratio)
    return img[max(0, y1 - ph):min(h, y2 + ph), max(0, x1 - pw):min(w, x2 + pw)]


_font_cache: dict[int, object] = {}


def _pil_font(size: int):
    try:
        from PIL import ImageFont
    except Exception:  # pragma: no cover - PIL missing
        return None
    if size not in _font_cache:
        font = None
        for path in _FONT_PATHS:
            try:
                font = ImageFont.truetype(path, size)
                break
            except Exception:
                continue
        _font_cache[size] = font
    return _font_cache[size]


def draw_text(img: np.ndarray, text: str, org, font_scale: float = 1.0,
              color=(0, 255, 0)):
    """Draw text (may contain Chinese) on a copy of img via PIL, cv2 fallback.

    Kept as a module-level function with the original signature so old code
    that imported it from recognize.py keeps working.
    """
    out = img.copy()
    try:
        from PIL import Image, ImageDraw
        pil = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil)
        font = _pil_font(int(32 * font_scale))
        if font is None:
            raise RuntimeError("no CJK font")
        draw.text((int(org[0]), int(org[1])), text, font=font,
                  fill=(color[2], color[1], color[0]),
                  stroke_width=3, stroke_fill=(0, 0, 0))
        return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
    except Exception:
        cv2.putText(out, text, (int(org[0]), int(org[1])),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)
        return out


def imread_unicode(path: Path) -> np.ndarray:
    """cv2.imread that also works for paths containing non-ASCII characters."""
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def _b64jpeg(img: np.ndarray, quality: int = 88) -> str | None:
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return None
    return base64.b64encode(buf.tobytes()).decode("ascii")


# --------------------------------------------------------------------------- #
#  Engine
# --------------------------------------------------------------------------- #
class ANPREngine:
    """One process keeps exactly one YOLO + one recognition module.

    YOLO localises plates; recognition is the standalone ``TextRecognition``
    module, so no detection pass runs twice.  All inference happens under a
    re-entrant lock: PaddleX predictors are not thread-safe, so concurrent
    HTTP requests are serialised here.
    """

    def __init__(self, weights: str | Path | None = None,
                 rec_model_dir: str | Path | None = None,
                 ocr_version: str = "PP-OCRv6_small",
                 device: str | None = None, conf: float = 0.5,
                 imgsz: int = 640, mkldnn: bool = False,
                 enhance_small_crops: bool = True,
                 infer_engine: str | None = None, warmup: bool = True,
                 rec_url: str | None = None,
                 preload: bool = True, raise_errors: bool = False):
        if ocr_version not in REC_MODEL_NAMES:
            raise ValueError(
                f"unknown --ocr-version {ocr_version!r}; "
                f"choose from {sorted(REC_MODEL_NAMES)}")
        self.weights_arg = Path(weights) if weights else None
        self.rec_model_dir_arg = Path(rec_model_dir) if rec_model_dir else None
        self.ocr_version = ocr_version
        self.rec_name = REC_MODEL_NAMES[ocr_version]
        self.device = device
        # "" / "paddle" / "default" -> PaddleX default engine (paddle inference)
        eng = (infer_engine or os.environ.get("ANPR_ENGINE") or "").strip().lower()
        self.infer_engine = None if eng in ("", "paddle", "default", "paddlepaddle") else eng
        # Remote recognition service (code/rec_service.py, anpr-ocr env): when set,
        # crops are POSTed to <rec_url>/api/rec and no local recogniser is loaded.
        # Deliberately no silent fallback - if the service is down the API reports
        # the failure instead of quietly serving the much weaker official model.
        self.rec_url = (rec_url or os.environ.get("ANPR_REC_URL") or "").strip().rstrip("/") or None
        self.rec_backend = "remote" if self.rec_url else "local"
        self.remote_health: dict | None = None
        self.warmup = bool(warmup)
        self.conf = float(conf)
        self.imgsz = int(imgsz)
        self.mkldnn = bool(mkldnn)
        self.enhance_small_crops = bool(enhance_small_crops)
        self.raise_errors = bool(raise_errors)

        self._lock = threading.RLock()
        self.loaded = False
        self.error: str | None = None
        self.load_seconds = 0.0
        self.weights_path: Path | None = None
        self.rec_model_dir: Path | None = None            # actually used (None = official)
        self.rec_model_dir_requested: Path | None = None  # what the user asked for
        self.used_engine: str | None = None
        self.used_ocr_device: str | None = None
        self.used_yolo_device: str | None = None
        self.notes: list[str] = []
        self._yolo = None
        self._rec = None
        self._yolo_dev = None

        if preload:
            self.load()

    # ------------------------------------------------------------- loading --
    @staticmethod
    def _default_yolo_weights() -> Path:
        env = os.environ.get("ANPR_YOLO_WEIGHTS")
        if env:
            return Path(env)
        bundled = PROJECT_ROOT / "models" / "detection" / "best.pt"
        if bundled.exists():
            return bundled
        return PROJECT_ROOT / "runs" / "ccpd_yolo" / "weights" / "best.pt"

    @staticmethod
    def _default_rec_model_dir() -> Path | None:
        env = os.environ.get("ANPR_REC_MODEL_DIR")
        if env:
            return Path(env)
        for cand in ("output/inference",
                     "output/best_accuracy/inference",
                     "output/latest/inference"):
            p = BASE_DIR / cand
            if (p / "inference.json").exists() and (p / "inference.pdiparams").exists():
                return p
        return None

    # ------------------------------------------------- device / engine logic --
    @staticmethod
    def _has_cuda() -> bool:
        try:
            import torch
            return bool(torch.cuda.is_available())
        except Exception:
            return False

    def _yolo_device(self) -> str | None:
        """Normalise the requested device to something ultralytics understands."""
        if not self.device:
            return None                      # None = ultralytics auto-select
        d = str(self.device).strip().lower()
        if d == "cpu":
            return "cpu"
        if d.startswith("gpu") or d.startswith("cuda"):
            if not self._has_cuda():
                log.warning("device=%s requested for YOLO but CUDA is unavailable; "
                            "using cpu", self.device)
                return "cpu"
            return d.split(":", 1)[1] if ":" in d else "0"   # gpu:1 -> "1"
        return str(self.device)

    def _ocr_device(self) -> str | None:
        """Normalise the requested device for PaddleX (gpu -> gpu:0)."""
        if not self.device:
            return None
        d = str(self.device).strip().lower()
        if d == "gpu":
            return "gpu:0"
        return str(self.device)

    @staticmethod
    def _looks_transformers(model_dir: Path | None) -> bool:
        """True when a model dir holds safetensors/HF files (transformers engine).

        PaddleX/PaddleOCR inference exports are PIR (inference.json/.pdiparams),
        which the transformers engine cannot load.
        """
        if model_dir is None:
            return False        # official models ship both formats
        if list(model_dir.glob("*.safetensors")):
            return True
        return (model_dir / "config.json").exists() and not (model_dir / "inference.json").exists()

    def _ocr_attempts(self, rec_dir: Path | None) -> list[tuple]:
        """Ordered (engine, rec_dir, device, description) attempts for the recognition module."""
        req_dev = self._ocr_device()
        eng = self.infer_engine
        attempts: list[tuple] = []

        if eng and rec_dir is not None and not self._looks_transformers(rec_dir):
            note = (f"engine={eng} cannot load the PIR export at {rec_dir} "
                    f"(it needs safetensors/HF files), so the built-in "
                    f"{self.rec_name} model is used for {eng} inference instead. "
                    f"To use that model dir, run with the default paddle "
                    f"engine (ANPR_ENGINE=paddle) on CPU.")
            self.notes.append(note)
            log.warning(note)
            attempts.append((eng, None, req_dev, f"{eng} + official rec"))
            attempts.append((None, rec_dir, req_dev, "paddle + custom rec"))
            attempts.append((None, rec_dir, "cpu", "paddle + custom rec on CPU"))
        elif eng:
            attempts.append((eng, rec_dir, req_dev, f"{eng} (requested backend)"))
            attempts.append((None, rec_dir, req_dev, "paddle fallback"))
            attempts.append((None, rec_dir, "cpu", "paddle fallback on CPU"))
        else:
            attempts.append((None, rec_dir, req_dev, "paddle (default)"))
            if (req_dev or "").lower() != "cpu":
                attempts.append((None, rec_dir, "cpu", "paddle on CPU"))

        # de-duplicate while keeping order
        seen, uniq = set(), []
        for a in attempts:
            if a[:3] not in seen:
                seen.add(a[:3])
                uniq.append(a)
        return uniq

    def _build_rec(self, rec_dir: Path | None, engine: str | None, device: str | None):
        """Build the rec-only ``TextRecognition`` module (no detection pass)."""
        kwargs = dict(
            model_name=self.rec_name,
            enable_mkldnn=self.mkldnn,
        )
        if engine:
            kwargs["engine"] = engine
        if device:
            kwargs["device"] = device
        if rec_dir is not None:
            kwargs["model_dir"] = str(rec_dir)
        log.info("Creating TextRecognition module (engine=%s, device=%s, rec=%s) ...",
                 engine or "paddle", device or "auto",
                 rec_dir or f"official {self.rec_name}")
        TextRecognition = _import_text_recognition()
        return TextRecognition(**kwargs)

    @staticmethod
    def _probe_image() -> np.ndarray:
        """Small synthetic image used to warm up / validate the backends."""
        img = np.full((64, 240, 3), 32, dtype=np.uint8)
        cv2.rectangle(img, (18, 14), (222, 50), (235, 235, 240), -1)
        return img

    def load(self):
        """Build both predictors. Safe to call multiple times / concurrently.

        Each backend (engine / device) combination is validated with a real
        one-shot inference; on failure the next combination is tried, so an
        incompatible model format or a missing GPU never takes the service down.
        """
        with self._lock:
            if self.loaded:
                return
            t0 = time.perf_counter()
            try:
                weights = self.weights_arg or self._default_yolo_weights()
                if not weights.exists():
                    raise FileNotFoundError(
                        f"YOLO weights not found: {weights}. Train a detector "
                        f"(python train.py) or point ANPR_YOLO_WEIGHTS at a "
                        f"best.pt file.")
                self.weights_path = weights.resolve()

                rec_dir = self.rec_model_dir_arg or self._default_rec_model_dir()
                self.rec_model_dir_requested = rec_dir.resolve() if rec_dir else None
                probe = self._probe_image()

                # ---- YOLO (validated with a warm-up predict) ---------------
                log.info("Loading YOLO from %s ...", self.weights_path)
                self._yolo = YOLO(str(self.weights_path))
                yolo_dev = self._yolo_device()
                candidates = [yolo_dev] if not yolo_dev or yolo_dev == "cpu" else [yolo_dev, "cpu"]
                for cand in candidates:
                    try:
                        self._yolo.predict(probe, conf=0.25, imgsz=320,
                                           device=cand, verbose=False)
                        self._yolo_dev = cand
                        self.used_yolo_device = cand or "auto"
                        log.info("YOLO ready (device=%s)", self.used_yolo_device)
                        break
                    except Exception as exc:  # noqa: BLE001
                        log.warning("YOLO warm-up failed (device=%s): %s: %s",
                                    cand or "auto", type(exc).__name__, exc)
                else:
                    raise RuntimeError("YOLO warm-up failed on every device")

                # ---- recognition: remote service, or a local module ---------
                if self.rec_url:
                    self.remote_health = self.remote_health_probe()
                    if not self.remote_health.get("loaded"):
                        reason = (self.remote_health.get("error")
                                  or self.remote_health.get("detail") or "unknown")
                        raise RuntimeError(
                            f"recognition service at {self.rec_url} is not ready: {reason}"
                            f" - start it with rec_service.py (anpr-ocr env)")
                    self.used_engine = self.remote_health.get("engine") or "paddle"
                    self.used_ocr_device = self.remote_health.get("device") or "auto"
                    remote_dir = self.remote_health.get("model_dir")
                    self.rec_model_dir = Path(remote_dir) if remote_dir else None
                    self.rec_model_dir_requested = self.rec_model_dir
                    log.info("Remote recogniser ready at %s (engine=%s, device=%s, model=%s)",
                             self.rec_url, self.used_engine, self.used_ocr_device,
                             self.rec_model_dir or f"official {self.rec_name}")
                else:
                    last_exc: Exception | None = None
                    for engine, rd, dev, why in self._ocr_attempts(self.rec_model_dir_requested):
                        try:
                            rec = self._build_rec(rd, engine, dev)
                            if self.warmup:
                                rec.predict(probe)  # validates model + engine + device
                            self._rec = rec
                            self.rec_model_dir = rd
                            self.used_engine = engine or "paddle"
                            self.used_ocr_device = dev or "auto"
                            log.info("TextRecognition ready (engine=%s, device=%s, rec=%s) [%s]",
                                     self.used_engine, self.used_ocr_device,
                                     rd or f"official {self.rec_name}", why)
                            break
                        except Exception as exc:  # noqa: BLE001
                            last_exc = exc
                            log.warning("recognition attempt failed [%s] "
                                        "(engine=%s, device=%s, rec=%s): %s: %s",
                                        why, engine or "paddle", dev or "auto",
                                        rd or "official", type(exc).__name__, exc)
                    if self._rec is None:
                        raise RuntimeError(
                            f"all recognition backend attempts failed; last error: "
                            f"{type(last_exc).__name__}: {last_exc}")

                self.load_seconds = time.perf_counter() - t0
                self.loaded = True
                self.error = None
                log.info("Engine ready in %.1fs (yolo=%s@%s, ocr=%s@%s, rec=%s)",
                         self.load_seconds, self.weights_path.name,
                         self.used_yolo_device, self.used_engine,
                         self.used_ocr_device,
                         self.rec_model_dir or f"official {self.rec_name}")
            except Exception as exc:  # noqa: BLE001 - keep server alive, report later
                self._yolo = None
                self._rec = None
                self.loaded = False
                self.error = f"{type(exc).__name__}: {exc}"
                log.error("Engine failed to load: %s", self.error)
                if self.raise_errors:
                    raise RuntimeError(self.error) from exc

    # ------------------------------------------------------------ inference --
    @staticmethod
    def _extract_detections(result) -> tuple[list, list]:
        det = result.boxes
        if det is not None and len(det):
            boxes = det.xyxy.cpu().numpy().tolist()
            scores = det.conf.cpu().numpy().tolist()
        else:
            boxes, scores = [], []
        return boxes, scores

    @staticmethod
    def _as_seq(value):
        """Safely coerce paddlex output fields (lists OR numpy arrays) to a list.

        Never do ``x or []`` on these: ``bool(ndarray)`` raises
        "truth value of an array is ambiguous" when the array has >1 element.
        """
        if value is None:
            return []
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (list, tuple)):
            return list(value)
        try:
            return list(value)
        except TypeError:
            return []

    # ------------------------------------------------- remote recogniser ----
    def remote_health_probe(self) -> dict:
        """GET <rec_url>/health; never raises - problems come back as a dict."""
        if not self.rec_url:
            return {}
        try:
            with urllib.request.urlopen(f"{self.rec_url}/health", timeout=10) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            try:
                return json.loads(exc.read().decode("utf-8"))
            except Exception:                       # noqa: BLE001
                return {"loaded": False, "error": f"HTTP {exc.code}"}
        except Exception as exc:                    # noqa: BLE001
            return {"loaded": False, "error": f"{type(exc).__name__}: {exc}"}

    def _mark_remote_down(self, reason: str) -> None:
        """Record a failed remote call so /api/health stops reporting 'ok'."""
        if not self.rec_url:
            return
        health = dict(self.remote_health or {})
        health.update({"loaded": False, "status": "unreachable", "error": reason,
                       "checked_at": time.time()})
        self.remote_health = health

    def _mark_remote_up(self) -> None:
        """After a failure, re-probe once the service answers again."""
        if self.remote_health is not None and self.remote_health.get("status") != "ok":
            self.remote_health = self.remote_health_probe()

    def _rec_row_remote(self, crop: np.ndarray) -> dict:
        """POST one crop to the recognition service (same shape as _rec_row)."""
        b64 = _b64jpeg(crop, quality=92)
        if not b64:
            raise RuntimeError("could not JPEG-encode the crop")
        payload = json.dumps({"images": [b64]}).encode("utf-8")
        req = urllib.request.Request(
            f"{self.rec_url}/api/rec", data=payload,
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:300]
            self._mark_remote_down(f"HTTP {exc.code}: {detail}")
            raise RuntimeError(f"recognition service HTTP {exc.code}: {detail}") from exc
        except Exception as exc:                    # noqa: BLE001
            self._mark_remote_down(f"{type(exc).__name__}: {exc}")
            raise RuntimeError(f"recognition service unreachable: "
                               f"{type(exc).__name__}: {exc}") from exc
        self._mark_remote_up()
        results = body.get("results") or []
        if not results:
            return {"texts": [], "scores": [], "boxes": []}
        item = results[0]
        if item.get("error"):
            raise RuntimeError(f"recognition service: {item['error']}")
        text = str(item.get("text") or "")
        try:
            score = float(item.get("score") or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        if not np.isfinite(score):
            score = 0.0
        return {"texts": [text] if text else [], "scores": [score], "boxes": []}

    def _rec_row(self, crop: np.ndarray) -> dict:
        """Run the recognition module on one crop.

        Returns ``{'texts': [...], 'scores': [...], 'boxes': []}``.  The
        ``TextRecognition`` module reports a single line under the singular
        keys ``rec_text`` / ``rec_score``; the plural pipeline-style keys are
        accepted too, so a future paddleocr version keeps working.  An empty
        decode comes back with score ``nan``, which is not JSON-safe, so
        non-finite scores are normalised to 0.0 here.

        In remote mode (``ANPR_REC_URL``) the crop goes to the recognition
        service instead; both paths return the same dict.
        """
        if self.rec_url:
            return self._rec_row_remote(crop)
        rows = self._rec.predict(crop)
        if not rows:
            return {"texts": [], "scores": [], "boxes": []}
        row = rows[0]
        try:
            get = row.get
        except AttributeError:
            row = getattr(row, "to_dict", lambda: dict(row))()
            get = row.get

        def _one(*keys):
            """First non-empty value among ``keys``, always as a list."""
            for key in keys:
                value = get(key)
                if value is None:
                    continue
                if isinstance(value, str):
                    if value:
                        return [value]
                    continue
                if isinstance(value, (int, float)):
                    return [value]
                seq = self._as_seq(value)
                if seq:
                    return seq
            return []

        texts = [str(t) for t in _one("rec_text", "rec_texts")]
        scores = []
        for s in _one("rec_score", "rec_scores"):
            try:
                f = float(s)
            except (TypeError, ValueError):
                f = 0.0
            scores.append(f if np.isfinite(f) else 0.0)
        boxes_raw = self._as_seq(get("rec_boxes"))

        def first_x(b):
            try:
                arr = np.asarray(b, dtype=float).reshape(-1)
                return float(arr[0]) if arr.size else 0.0
            except Exception:
                return 0.0

        order = sorted(range(len(texts)), key=lambda i: first_x(boxes_raw[i])
                       if i < len(boxes_raw) else 0.0)
        texts = [texts[i] for i in order]
        scores = [scores[i] for i in order] if len(scores) == len(order) else scores
        return {"texts": texts, "scores": scores, "boxes": boxes_raw}

    def _prepare_crop(self, crop: np.ndarray) -> np.ndarray:
        """Upscale very small crops: recognition accuracy drops sharply on tiny
        (< ~220 px wide) images, which is common for distant plates."""
        if not self.enhance_small_crops:
            return crop
        h, w = crop.shape[:2]
        if w <= 0:
            return crop
        if w < 220:
            scale = min(4.0, 320.0 / w)
            if scale > 1.0:
                crop = cv2.resize(crop, None, fx=scale, fy=scale,
                                  interpolation=cv2.INTER_CUBIC)
        return crop

    @staticmethod
    def _plate_from_rec(res: dict) -> tuple[str, str, float, bool]:
        """Join recognition lines left-to-right, then normalize to a plate.

        Returns (raw_text, plate, confidence, is_valid).
        """
        texts = [t for t in res["texts"] if t and any(c in PLATE_ALLOWED for c in t)]
        scores = list(res["scores"])
        if not texts:
            return "", "", 0.0, False
        raw = "".join(texts)
        raw = unicodedata.normalize("NFKC", raw).upper().replace(" ", "")
        conf = float(np.mean(scores)) if scores else 0.0
        plate, valid = normalize_plate(raw)
        return raw, plate, conf, valid

    def _draw_label(self, img: np.ndarray, label: str, x1: int, y1: int,
                    y2: int, color):
        """Draw a (possibly Chinese) label just above the box, falling back to
        inside the box when there is no room above."""
        h_img = img.shape[0]
        font_size = max(14, min(40, int((y2 - y1) * 0.9)))
        ascent = font_size + 6
        if y1 - ascent >= 4:
            draw_text(img, label, (x1, y1 - ascent), font_scale=1.0, color=color)
        else:
            yy = min(y1 + font_size, h_img - 4)
            draw_text(img, label, (x1, yy), font_scale=1.0, color=color)

    # ------------------------------------------------------------ public API --
    def model_summary(self, refresh_remote: bool = False) -> dict:
        """Engine state as a JSON-ready dict.

        ``refresh_remote=True`` re-probes the recognition service (used by
        ``/api/health``); per-request payloads reuse the cached snapshot so a
        probe is not paid on every inference call.
        """
        if self.rec_url and refresh_remote:
            self.remote_health = self.remote_health_probe()
        return {
            "loaded": bool(self.loaded),
            "error": self.error,
            "load_seconds": round(self.load_seconds, 2),
            "yolo_weights": str(self.weights_path) if self.weights_path else None,
            "yolo_device": self.used_yolo_device or "auto",
            "engine_requested": self.infer_engine or "paddle",
            "engine": self.used_engine or self.infer_engine or "paddle",
            "ocr_device": self.used_ocr_device or "auto",
            "ocr_name": self.rec_name,
            "ocr_module": "TextRecognition",
            "rec_backend": self.rec_backend,
            "rec_url": self.rec_url,
            "rec_remote": self.remote_health,
            "rec_model_dir": str(self.rec_model_dir) if self.rec_model_dir else None,
            "rec_model_dir_requested": (str(self.rec_model_dir_requested)
                                        if self.rec_model_dir_requested else None),
            "device": self.used_yolo_device or self.device or "auto",
            "imgsz": self.imgsz,
            "default_conf": self.conf,
            "notes": list(self.notes),
        }

    def recognize_bytes(self, data: bytes, conf: float | None = None,
                        with_annotated: bool = False,
                        with_crops: bool = False) -> dict:
        """Decode raw image bytes and run the full pipeline."""
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("could not decode image bytes "
                             "(is it a valid JPEG/PNG/WebP?)")
        return self.recognize(img, conf=conf, with_annotated=with_annotated,
                              with_crops=with_crops)

    def recognize_file(self, path: str | Path, conf: float | None = None,
                       with_annotated: bool = False,
                       with_crops: bool = False) -> dict:
        img = imread_unicode(Path(path))
        if img is None:
            raise ValueError(f"failed to read image: {path}")
        return self.recognize(img, conf=conf, with_annotated=with_annotated,
                              with_crops=with_crops)

    def recognize(self, image: np.ndarray, conf: float | None = None,
                  with_annotated: bool = False,
                  with_crops: bool = False) -> dict:
        """Detect + recognise plates in one BGR image.

        Returns a plain, JSON-ready dict:

            {
              "success": True,
              "plates": [{bbox, det_conf, raw_text, plate, valid,
                          rec_conf, ocr_ms, crop_b64?}],
              "annotated_b64": str | None,     # only when with_annotated
              "image_size": [w, h],
              "timings_ms": {...},
            }
        """
        t_total = time.perf_counter()
        timings: dict = {}

        plates: list[dict] = []
        crop_imgs: list[np.ndarray | None] = []
        annotated_bgr: np.ndarray | None = None

        if not self.loaded:
            self.load()
        if not self.loaded:
            raise RuntimeError(self.error or "models are not loaded")

        conf = float(conf) if conf is not None else self.conf
        conf = min(0.95, max(0.05, conf))

        with self._lock:
            # -- detection ---------------------------------------------------
            t0 = time.perf_counter()
            results = self._yolo.predict(
                image, conf=conf, imgsz=self.imgsz,
                device=self._yolo_dev, verbose=False)
            timings["detect_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            boxes, scores = self._extract_detections(results[0])
            h_img, w_img = image.shape[:2]

            # -- annotation canvas ------------------------------------------
            if with_annotated:
                scale = min(1.0, 1920.0 / max(h_img, w_img)) if max(h_img, w_img) > 1920 else 1.0
                annotated_bgr = image.copy()
                if scale < 1.0:
                    annotated_bgr = cv2.resize(
                        annotated_bgr, None, fx=scale, fy=scale,
                        interpolation=cv2.INTER_AREA)

            # -- per-plate OCR -----------------------------------------------
            ocr_total = 0.0
            for i, (box, det_conf) in enumerate(zip(boxes, scores)):
                p0 = time.perf_counter()
                x1, y1, x2, y2 = (int(v) for v in box)
                crop = crop_plate(image, box)
                crop = self._prepare_crop(crop)

                ocr_ok = True
                ocr_msg = None
                try:
                    res = self._rec_row(crop)
                    raw, plate, rec_conf, valid = self._plate_from_rec(res)
                    if not valid:
                        log.info("plate %d OCR empty/invalid: raw=%r", i, raw)
                except Exception as exc:  # noqa: BLE001 - one plate must not kill the request
                    raw, plate, rec_conf, valid = "", "", 0.0, False
                    ocr_ok = False
                    ocr_msg = f"{type(exc).__name__}: {exc}"
                    log.warning("plate %d OCR failed: %s", i, exc)

                ocr_ms = round((time.perf_counter() - p0) * 1000, 1)
                ocr_total += ocr_ms

                plate_item = {
                    "index": i,
                    "bbox": [x1, y1, x2, y2],
                    "det_conf": round(float(det_conf), 4),
                    "raw_text": raw,
                    "plate": plate,
                    "valid": valid,
                    "rec_conf": round(float(rec_conf), 4),
                    "ocr_ok": ocr_ok,
                    "ocr_error": ocr_msg,
                    "ocr_ms": ocr_ms,
                }
                plates.append(plate_item)
                crop_imgs.append(crop if with_crops else None)

                if annotated_bgr is not None:
                    color = (34, 197, 94) if plate_item["valid"] else (0, 165, 255)
                    sx = sy = 1.0
                    if scale < 1.0:
                        sx = sy = scale
                    ax1, ay1 = int(x1 * sx), int(y1 * sy)
                    ax2, ay2 = int(x2 * sx), int(y2 * sy)
                    cv2.rectangle(annotated_bgr, (ax1, ay1), (ax2, ay2), color, 2)
                    label = f"{plate or raw or '?'}  {plate_item['rec_conf']:.0%}"
                    self._draw_label(annotated_bgr, label, ax1, ay1, ay2, color)

            timings["ocr_total_ms"] = round(ocr_total, 1)
            timings["ocr_per_plate_ms"] = [p["ocr_ms"] for p in plates]

        # -- encoding happens outside the lock -------------------------------
        if with_crops:
            for p, c in zip(plates, crop_imgs):
                p["crop_b64"] = _b64jpeg(c, quality=90) if c is not None else None

        annotated_b64 = None
        if with_annotated and annotated_bgr is not None:
            annotated_b64 = _b64jpeg(annotated_bgr, quality=90)

        timings["total_ms"] = round((time.perf_counter() - t_total) * 1000, 1)

        # -- per-plate + request summary logs (server console visibility) ----
        for p in plates:
            log.info(
                "plate #%s bbox=%s det=%.3f valid=%s plate=%r raw=%r rec=%.3f "
                "ocr_ok=%s%s",
                p["index"], p["bbox"], p["det_conf"], p["valid"],
                p["plate"], p["raw_text"], p["rec_conf"], p["ocr_ok"],
                f" error={p['ocr_error']}" if p.get("ocr_error") else "",
            )
        if not plates:
            log.info("no plates detected (conf=%.2f)", conf)
        log.info("request done: %d plate(s), detect=%.1fms, ocr=%.1fms, total=%.1fms",
                 len(plates), timings["detect_ms"], timings["ocr_total_ms"],
                 timings["total_ms"])

        return {
            "success": True,
            "plates": plates,
            "annotated_b64": annotated_b64,
            "image_size": [w_img, h_img],
            "timings_ms": timings,
        }
