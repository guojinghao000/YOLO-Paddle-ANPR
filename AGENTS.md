# YOLO-Paddle-ANPR

License plate detection + recognition: YOLO → plate localization, PaddleOCR → character recognition.

## Dataset

CCPD2019 at `CCPD2019/` — annotations are **embedded in filenames**, not separate label files.
Filename schema: `area-tilt_degree-bbox_xy-vertices-plate-brightness-blurriness.jpg`

- `splits/train.txt` / `val.txt` / `test.txt` list image paths (relative to `CCPD2019/`).
- `tools.py` → `Stringparsing()` parses filenames into dict fields via regex.
- `tools.py` → `annotate_image()` draws bbox + vertices + plate text for debugging.

## Project state

- `code/yolo26n.pt` — pre-trained YOLO nano weights.
- `code/process_dataset.py` — converts CCPD2019 annotations to YOLO label format. Generates `dataset/data.yaml` + image lists + `.txt` labels alongside images.
- `code/train.py` — YOLO fine-tuning script.
- `code/recognize.py` — single-image CLI: YOLO detect → bbox crop (+5% padding) → PaddleOCR recognition module (`TextRecognition`, rec-only). Thin wrapper over the shared engine (below); original args kept, `--device` now defaults to auto instead of forcing GPU `0`, and `--engine` defaults to `transformers` (GPU, same as `api_server.py`; a combination that cannot load falls back to the other engine). Chinese text drawn via PIL (msyh.ttc), falls back to cv2. `--rec-model-dir` points at a custom PaddleX/PaddleOCR inference export (a directory holding `inference.json` + `inference.pdiparams`; must also set the matching `--ocr-version`, e.g. `PP-OCRv6_small`) — without it the official pretrained model for `--ocr-version` is used.
- `code/anpr_engine.py` — shared inference engine used by the CLI **and** the HTTP API: loads YOLO once per process plus the standalone recognition module (`paddleocr.TextRecognition`, rec-only — no detection pass, see the Environment notes), or POSTs each crop to `code/rec_service.py` when `ANPR_REC_URL` is set (split deployment), RLock-serialises inference (PaddleX predictors are not thread-safe), filters the recognised text through the plate charset (`PLATE_ALLOWED`: province chars + letters without I/O + digits) and validates plate shape. Returns JSON-ready dicts with per-stage timings.
- `code/api_server.py` — FastAPI HTTP API (models preloaded at startup; fail-soft → 503 with reason). Endpoints below. Serves the static web UI from `/`.
- `code/web/index.html` — single-file Chinese frontend: upload / drag&drop / Ctrl+V paste / camera snapshot / continuous video recognition.
- `code/tools.py` — filename parsing (`Stringparsing`) + annotation visualization.
- `code/make_plate_rec_dataset.py` — CCPD2019 → plate recognition dataset: decodes the 7 plate index fields in the file name (e.g. `0_0_22_10_26_29_24` → `皖AC9M60`), crops with the **same +5% padding as serving** (`anpr_engine.crop_plate`), samples 8 subsets with hard subsets over-weighted, dedupes images that appear in several subsets, and writes `train.txt` / `val.txt` / `crop_img/` / `dict.txt`. `--preview N` also writes a labelled contact sheet.
- `code/train_ocr.py` — fine-tuning entry point (see the section at the end): loads `code/configs/plate_rec.yml`, applies CLI overrides, writes a reproducible `<output>/run_config.yml` and calls `PaddleOCR/tools/train.py`; `--smoke`, `--resume`, `--export` / `--export-only`.
- `code/verify_rec_plate.py` — evaluates a model dir (or the official model) on a plate split using the **same `TextRecognition` module serving uses**: whole-plate exact match, per-position character accuracy, per-subset table, `--dump` per-sample JSON.
- `code/configs/plate_rec.yml` — fine-tuning config (upstream PP-OCRv6_small_rec architecture untouched; data paths, dict, lr, batch and eval cadence overridden).
- `code/rec_service.py` — standalone recognition service (anpr-ocr env; FastAPI + `TextRecognition` + the PIR export): `GET /health`, `POST /api/rec` (base64 crops → text/score). Loads the model before serving, serialises inference with a lock (PaddleX predictors are not thread-safe) and passes `model_name` so it matches the export's `inference.yml`.
- `code/eval_e2e.py` — end-to-end accuracy over CCPD images (YOLO + recognition) compared with the plate text in the file names; `--rec-url` scores the fine-tuned remote model, omitting it scores the local official baseline.
- `start_anpr.bat` / `stop_anpr.bat` — one-click dual-service launcher (starts the recogniser, waits until it reports `loaded=true`, then starts the API) and a precise stopper (kills whatever listens on 8000/8100 and closes the wrapper windows).
- Repo root also holds the training inputs the user provided: `PaddleOCR/` (upstream clone used for `tools/train.py` / `tools/export_model.py`), `PP-OCRv6_small_rec_pretrained.pdparams` and `train_data/` (PaddleOCR's English demo dataset, unrelated to plates).
- Trained weights: `runs/ccpd_yolo/weights/best.pt` (mAP50 0.995, mAP50-95 ~0.749, plateaued).

## Environment

- Conda-based (`.vscode/settings.json` → `ms-python.python:conda`), Python 3.10.
- Dependencies: `ultralytics`, `opencv-python`, `numpy`, `paddlepaddle`, `paddleocr`. Web service additionally needs `fastapi`, `uvicorn[standard]`, `python-multipart` → `pip install -r code/requirements-web.txt`.
- **PaddlePaddle on Windows**: GPU wheels stop at 2.6.2, but `paddlex>=3.7` requires paddle 3.x API (`AnalysisConfig.set_optimization_level`) → must use `paddlepaddle` 3.3.x **CPU** (no Windows GPU wheel for 3.x). Installed: paddle 3.3.1 + paddleocr 3.7.0 + paddlex 3.7.2.
- **DLL conflict**: importing `paddle` before `torch` in the same process crashes torch (`shm.dll` WinError 127) — paddle 3.3 pollutes the DLL search path. Always import torch/ultralytics first (recognize.py already does this).
- **CUDA torch and CUDA paddle cannot share a process** (measured in the ML env, both directions fail): `import paddle; import torch` → `torch/lib/shm.dll` or `cudnn_engines_precompiled64_9.dll` (WinError 127); `import torch; import paddle` → `nvidia/cublas/bin/cublas64_12.dll`. Registering DLL dirs (`os.add_dll_directory`, PATH) does not help. Serving is unaffected: inference uses the environment's real CUDA torch, and PaddleX's transformers engine never imports paddle (`paddle in sys.modules == False` after loading a safetensors rec model), which is why YOLO + transformers OCR share one process.
- PaddleOCR 3.x defaults to PP-OCRv6 models (downloaded to `~/.paddlex/official_models` on first run); model name mappings live in `code/anpr_engine.py` → `REC_MODEL_NAMES` (re-exported by recognize.py for CLI compatibility).
- **Recognition is the standalone module, not the pipeline**: `anpr_engine` uses `paddleocr.TextRecognition` instead of `PaddleOCR`, so nothing re-detects the plate — the pipeline's internal `PP-OCRv6_medium_det` (which paddlex 3.7 forces on even with `text_detection_model_name=None`) is never downloaded or loaded. Measured on one real crop: det+rec 87 ms vs rec-only 20 ms on GPU, identical text (沪K·R9888, 0.9989); per-request OCR 148–403 ms → 27–31 ms, whole request 284–616 ms → 136–148 ms, model load 46 s → 21 s. On the `paddle` engine (CPU, `anpr-ocr` env) the same module reads the crop correctly at ~80 ms.
- Two gotchas of that module (both handled in `anpr_engine._rec_row()`): results use the **singular** keys `rec_text`/`rec_score` while the pipeline used `rec_texts`/`rec_scores`, and an empty decode reports `rec_score = nan`, which is not JSON-safe. On CPU the paddle engine also needs `enable_mkldnn=False` — oneDNN crashes inside `predictor.run()` — which is why `ANPR_MKLDNN` defaults to 0.
- PaddleX/PaddleOCR inference exports are PIR JSON format (`inference.json` + `inference.pdiparams`, no `inference.pdmodel`); the `transformers` engine can only load safetensors/HF-format models, so such a directory is usable with the `paddle` engine only.
- When loading a custom rec model dir, the `model_name` passed to `TextRecognition` (`--ocr-version` + `--rec-model-dir` / `ANPR_REC_MODEL_DIR` here) must match that dir's `inference.yml` → `Global.model_name` (e.g. `PP-OCRv6_small_rec`), else the resolver raises "Model name mismatch". Such a dir may live anywhere and only needs `inference.json` + `inference.pdiparams` (+ `inference.yml`) for the `paddle` engine.

## Key conventions

- CCPD label fields: area, tilt, bbox, vertices (4 corners), plate (Chinese char + letter + 5 alphanum), brightness, blurriness.
- Four vertices order: right-bottom → left-bottom → left-top → right-top.

## Work rules

- Do not run scripts, only write code.

## Commands

```powershell
conda activate ML        # working env on this machine (E:\CondaEnvs\ML); the
                         # "YOLO-PADDLE-ANPR" env is an empty shell (no python.exe)
cd code
python process_dataset.py           # generate labels + dataset config
python train.py                     # fresh training (100 epochs)
python train.py --time 3            # train for max 3 hours
python train.py --time 3 --resume   # continue from last checkpoint
python train.py --epochs 200        # override epochs
python train.py --name exp2         # separate experiment
python recognize.py --source ../CCPD2019/ccpd_base/xxx.jpg
python recognize.py --source ../CCPD2019/ccpd_base/xxx.jpg --show
python recognize.py --source img.jpg --ocr-version PP-OCRv6
python recognize.py --source img.jpg --ocr-version PP-OCRv6_small --rec-model-dir <paddlex-export-dir>
```

- Labels generated alongside images (`CCPD2019/ccpd_base/xxx.txt`).
- Image list files (with absolute paths) go to `dataset/train.txt`, `dataset/val.txt`, `dataset/test.txt`.
- Training output (weights, plots, logs) to `runs/ccpd_yolo/`.
- Time-based training uses ultralytics' built-in `time` param (hours). When `--time` is set, it overrides `--epochs`.
- Resume uses ultralytics' native `resume` — loads `runs/{name}/weights/last.pt` and restores epoch/optimizer/lr state.

### TensorBoard monitoring

```powershell
pip install tensorboard
tensorboard --logdir=runs
```

Ultralytics auto-logs all metrics (loss, mAP, lr, images) to TensorBoard when the package is installed.

### Windows / ultralytics quirks

- `workers=0` required to avoid dataloader hangs.
- `data.yaml` must use absolute paths for train/val/test; `path` field is unreliable on Windows.

### HTTP API + Web UI

Start the FastAPI server (serves the web UI at `/` and REST at `/api`):

```powershell
conda activate ML        # working env on this machine (E:\CondaEnvs\ML); the
                         # "YOLO-PADDLE-ANPR" env is an empty shell (no python.exe)
cd code
pip install -r requirements-web.txt
python api_server.py --host 0.0.0.0 --port 8000     # open http://127.0.0.1:8000
```

Endpoints:

- `GET /api/health` — engine/model status + load time.
- `POST /api/recognize` — multipart: `file` (image), `conf` (0.05–0.95), `annotate` (bool, default true → returns annotated JPEG as data URL), `crops` (bool → per-plate crop data URLs).
- `POST /api/recognize/base64` — JSON `{image, conf, annotate, crops}`.
- `GET /api/demo` — run on a bundled sample at the repo root (`image.png` / `image_result.jpg`) if present.
- Response plates: `bbox` (xyxy ints), `det_conf`, `raw_text`, `plate` (charset-filtered), `valid`, `rec_conf`, `ocr_ms`; plus `timings_ms` (detect/ocr/total/request).

Env vars: `ANPR_HOST`, `ANPR_PORT`, `ANPR_PRELOAD` (`1`=preload at startup, fail-soft 503 on error; `0`=lazy on first request), `ANPR_WARMUP` (`1`=validate every engine/device combination with a real one-shot inference at startup), `ANPR_YOLO_WEIGHTS`, `ANPR_REC_MODEL_DIR` (auto-detect: `code/output/inference` → `output/best_accuracy/inference` → `output/latest/inference`, else the official pretrained model for `ANPR_OCR_VERSION`; usable with the `paddle` engine only), `ANPR_OCR_VERSION`, `ANPR_DEVICE` (`cpu`/`gpu`/`gpu:0`/`0`; `api_server.py` defaults to `gpu`, `recognize.py` to auto), `ANPR_ENGINE` (`api_server.py` and `recognize.py` both default to `transformers` = GPU via torch; set `paddle` for the PaddleX/PaddleOCR engine), `ANPR_REC_URL` (base URL of the recognition service for the split deployment, e.g. `http://127.0.0.1:8100`; when set, no local recogniser is loaded and a down service is reported instead of silently falling back), `ANPR_CONF`, `ANPR_MKLDNN`.

Notes:

- **Engines, not recognition backends**: `ANPR_ENGINE` selects the PaddleX inference engine — `transformers` (the default in both `api_server.py` and `recognize.py`: GPU via torch, safetensors/HF models only) or `paddle` (PaddleOCR/PaddleX, CPU on Windows). `paddleocr` is imported **lazily** in `anpr_engine._import_text_recognition()` so the transformers engine never touches paddle; the module-level ultralytics import still comes first (torch-before-paddle DLL rule).
- If `paddleocr`/`paddlex` cannot be imported in an environment (`OSError`, `RuntimeError: PDX has already been initialized`), only the `paddle` engine is lost, not serving as a whole: with CUDA torch already imported, `import paddle` dies on `nvidia/cublas/bin/cublas64_12.dll` (WinError 127), so every paddle attempt fails whatever device is requested. The `transformers` engine never imports paddle and keeps working — measured in the ML env: `/api/health` returns `loaded=true, engine=transformers, ocr_device=gpu:0` while the same interpreter cannot build a paddle engine at all. If no engine can be built, the server still starts, `/api/health` carries the failure in `engine.error` and `/api/recognize` answers 503.
- Post-processing is identical whichever engine runs: charset filtering + `normalize_plate()` shape validation, so `plate`/`valid`/`rec_conf` semantics are the same.
- **`engine=transformers` vs model format**: the transformers backend loads only safetensors/HF-format models. Official `PP-OCRv6_*` ship both formats (first run downloads `*_safetensors`), but a PaddleX/PaddleOCR inference export is PIR (`inference.json` + `.pdiparams`) and is therefore incompatible. When `ANPR_REC_MODEL_DIR` points at such a dir, the engine uses the built-in official rec model for `engine` inference, then falls back to `paddle` + that dir (CPU on Windows), and records the explanation in `model_summary()["notes"]` (server log + `/api/health`).
- Startup validates each (engine, device) combination with a warm-up `predict()` on a synthetic image and degrades in order transformers → paddle → cpu, so an incompatible backend or missing GPU never prevents the server from starting. `--no-warmup` / `ANPR_WARMUP=0` skips validation.
- `api_server.py --host` also accepts `host:port` in one argument (auto-split) — avoids `getaddrinfo failed` from `--host 127.0.0.1:8000`.
- Models load once per process (anpr_engine singleton, RLock-serialised inference). Do NOT run multiple uvicorn workers — each worker would load full duplicate models; `--workers` is capped at 1 by default.
- Engine returns plain JSON-ready dicts — no numpy types leak into responses.
- CLI (`recognize.py`) and API share the engine, so behaviour stays identical; CLI exposes the same knobs (`--engine`, `--device`, `--no-warmup`).

### Fine-tuning the plate recogniser (anpr-ocr env)

```powershell
E:\CondaEnvs\anpr-ocr\python.exe code\make_plate_rec_dataset.py --out rec_data_plate --train-samples 40000 --val-samples 4000
E:\CondaEnvs\anpr-ocr\python.exe code\train_ocr.py --smoke     # ~20 s: data/config/VRAM/eval plumbing
E:\CondaEnvs\anpr-ocr\python.exe code\train_ocr.py             # 30 epochs, ~5-7 h
E:\CondaEnvs\anpr-ocr\python.exe code\train_ocr.py --export-only
E:\CondaEnvs\anpr-ocr\python.exe code\verify_rec_plate.py --dataset rec_data_plate --split val --device gpu:0 --model-dir code\output_plate\inference --tag finetuned
```

`verify_rec_plate.py` scores a model with the same `TextRecognition` module serving uses, which only loads an **inference export** (`inference.json` + `inference.pdiparams`) — not a training checkpoint — so export first (`--export-only`); during training, read val `acc` from `<output>/train.log`.

Measured facts:

- **Use the official dict** (`PaddleOCR/ppocr/utils/dict/ppocrv6_dict.txt`, 18708 chars → CTC head 18710 = the pretrained head): loading the pretrained weights then produces **zero shape warnings** (verified). A small plate-only dict instead re-initialises the head (four `not matched` warnings) and starts from random — the user's earlier demo-data run showed `norm_edit_dis 0.0001` after 10 iters, versus 0.58 with the official dict.
- **Baseline to beat** (official `PP-OCRv6_small_rec` through the rec-only module, 4000 val crops): **29.73%** whole-plate exact match, 50.96% character accuracy, province char 32.82%. Per subset: base 58.88%, weather 51.33%, db 14.00%, fn 10.33%, challenge 5.00%, rotate 1.50%, tilt 0.80%, blur 0.50%. Per-sample dump: `code/eval_reports/official_val.json`.
- **The decoded labels are correct**: re-OCR-ing the crops with the official model reproduces positions 1..6 far more often than position 0, i.e. the CCPD index decoding holds and the province character is the model's weak spot.
- **Batch size**: an upstream-configured run (bs 128) peaked at **7.24 GB** on the 8 GB card, too close to the limit — `plate_rec.yml` uses 64 (1.5 GB measured at bs 16).
- **Env**: `anpr-ocr` ships CUDA paddle 3.3.1 (CUDA 12.9) and no torch, so training runs on the GPU without the torch/paddle DLL conflict, and the whole `ppocr` training stack imports there (`imgaug` is not needed, unlike the ML env earlier). Every run writes `<output>/run_config.yml` for reproducibility. `train_data/` at the repo root is PaddleOCR's English demo set — unrelated to plates.
- **Result of the first run** (30 epochs, 40k crops, best val acc 96.07%): the exported PIR scores **96.03%** whole-plate exact match on the 4000 val crops (character accuracy 99.22%, province char 99.50%) — the export reproduces the training-side number, unlike the earlier safetensors conversion. End to end (YOLO + recognition, 144 CCPD images sampled evenly across the 8 subsets) the fine-tuned model reaches **96.53%** versus **18.75%** for the official model, at identical throughput (13.8 vs 13.7 img/s, ~72 ms/image). Dumps: `code/eval_reports/finetuned_val.json`, `e2e_finetuned-e2e.json`, `e2e_official-e2e.json`.
- **Serving shape**: the export is PIR, so it needs the `paddle` engine, and `import paddle` only works inside `anpr-ocr`. The project therefore runs **two processes**: `code/rec_service.py` in `anpr-ocr` (port 8100, GPU) plus `code/api_server.py` in the ML env (port 8000, GPU YOLO) started with `--rec-url http://127.0.0.1:8100`. `start_anpr.bat` brings both up (it waits for the recogniser to report `loaded=true` before starting the API); `stop_anpr.bat` stops them by port and closes their windows. With `ANPR_REC_URL` set the API loads **no** local recogniser: a down service is reported (`/api/health` → `loaded=false` + reason, `/api/recognize` → 503) instead of silently serving the official model, `/api/health` re-probes the service on every call, and a mid-run outage fails only the affected plate (`ocr_ok=false` + reason) while recovery is automatic.
