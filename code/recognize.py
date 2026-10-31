"""Single-image CLI for license plate detection (YOLO) + recognition (PaddleOCR).

Thin wrapper around anpr_engine.ANPREngine - the exact same pipeline is used
by the HTTP API (api_server.py).

Examples
--------
python recognize.py --source ../CCPD2019/ccpd_base/xxx.jpg
python recognize.py --source img.jpg --show
python recognize.py --source img.jpg --ocr-version PP-OCRv6_small --rec-model-dir output/best_accuracy/inference
"""
import argparse
import base64
import os
import time
from pathlib import Path

import cv2
import numpy as np

from anpr_engine import ANPREngine, PROJECT_ROOT, REC_MODEL_NAMES

DEFAULT_WEIGHTS = PROJECT_ROOT / "models" / "detection" / "best.pt"


def main():
    parser = argparse.ArgumentParser(
        description="License plate detection (YOLO) + recognition (PaddleOCR)")
    parser.add_argument("--source", type=str, required=True, help="input image path")
    parser.add_argument("--weights", type=str, default=None,
                        help="YOLO weights path")
    parser.add_argument("--ocr-version", type=str, default="PP-OCRv6_small",
                        choices=list(REC_MODEL_NAMES),
                        help="PaddleOCR rec model version")
    parser.add_argument("--rec-model-dir", type=str, default=None,
                        help="custom rec model directory (takes precedence over the "
                             "official pretrained model for --ocr-version)")
    parser.add_argument("--conf", type=float, default=0.5,
                        help="detection confidence threshold")
    parser.add_argument("--device", type=str, default=None,
                        help="inference device: 'cpu', 'gpu', 'gpu:0', '0', or empty = auto")
    parser.add_argument("--engine", type=str,
                        default=os.environ.get("ANPR_ENGINE") or "transformers",
                        choices=["paddle", "transformers"],
                        help="PaddleOCR inference backend (default: transformers = GPU via "
                             "torch with safetensors models, matching api_server.py; "
                             "'paddle' = PaddleX/PaddleOCR, CPU on Windows. A backend that "
                             "cannot be loaded falls back to the other one automatically)")
    parser.add_argument("--rec-url", type=str,
                        default=os.environ.get("ANPR_REC_URL") or None,
                        help="remote recognition service (rec_service.py), e.g. "
                             "http://127.0.0.1:8100 - no local recogniser is loaded then")
    parser.add_argument("--no-warmup", action="store_true",
                        help="skip the startup warm-up/inference validation")
    parser.add_argument("--out", type=str, default=None,
                        help="output annotated image path")
    parser.add_argument("--show", action="store_true", help="display image window")
    args = parser.parse_args()

    source = Path(args.source)
    if not source.exists():
        raise FileNotFoundError(f"Image not found: {source}")
    weights = Path(args.weights) if args.weights else ANPREngine._default_yolo_weights()
    if not weights.exists():
        raise FileNotFoundError(f"YOLO weights not found: {weights}")

    # 1. Load models (once per process; cached for the whole run)
    t0 = time.time()
    engine = ANPREngine(
        weights=weights,
        rec_model_dir=args.rec_model_dir,
        ocr_version=args.ocr_version,
        device=args.device,
        conf=args.conf,
        infer_engine=args.engine,
        rec_url=args.rec_url,
        warmup=not args.no_warmup,
        preload=True,
        raise_errors=True,          # CLI should fail loudly on bad paths
    )
    print(f"[Timing] Models loaded in {time.time() - t0:.2f}s "
          f"(yolo={engine.weights_path.name}@{engine.used_yolo_device}, "
          f"ocr={engine.used_engine}@{engine.used_ocr_device}, "
          f"rec={engine.rec_model_dir or engine.rec_name})")
    for note in engine.notes:
        print(f"[Note] {note}")

    # 2. Recognise
    result = engine.recognize_file(source, conf=args.conf,
                                   with_annotated=True, with_crops=False)

    # 3. Report
    plates = result["plates"]
    print(f"Detected {len(plates)} plate(s) in {source.name}")
    for p in plates:
        status = "OK " if p["valid"] else ("RAW" if p["plate"] else "FAIL")
        print(f"Plate {p['index'] + 1} [{status}]: "
              f"{p['plate'] or p['raw_text'] or '(no text)'} "
              f"(rec conf {p['rec_conf']:.2f}, det conf {p['det_conf']:.2f})")
        print(f"  └─ OCR: {p['ocr_ms']:.1f}ms" +
              (f", warning: {p['ocr_error']}" if p.get("ocr_error") else ""))

    timings = result["timings_ms"]
    print(f"\n[Timing] Detect: {timings['detect_ms']:.1f}ms | "
          f"OCR total: {timings['ocr_total_ms']:.1f}ms | "
          f"Total: {timings['total_ms']:.1f}ms")

    # 4. Save annotated result
    if result.get("annotated_b64"):
        out = Path(args.out) if args.out else source.with_name(source.stem + "_result.jpg")
        data = base64.b64decode(result["annotated_b64"])
        annotated = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        cv2.imwrite(str(out), annotated)
        print(f"Saved annotated image to: {out}")

        # 5. Show (if requested)
        if args.show:
            cv2.imshow("Result", annotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
