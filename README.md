# YOLO-Paddle-ANPR · 车牌检测 + 识别

车牌检测与识别系统：YOLO 定位车牌，PaddleOCR `PP-OCRv6_small_rec` 识别字符。两者均已在 CCPD2019 上微调，支持 Web 界面、HTTP API 和单图命令行。

仓库包含部署所需的检测权重和识别模型。数据集、训练中间产物及评测明细不纳入版本控制。

| 文件 | 用途 |
| --- | --- |
| `models/detection/best.pt` | 微调后的 YOLO 检测权重 |
| `models/recognition/inference.json` | 识别模型 PIR 结构 |
| `models/recognition/inference.pdiparams` | 微调后的识别权重 |
| `models/recognition/inference.yml` | PaddleX 识别配置及字表 |

识别模型为 Paddle PIR 格式，需使用 `paddle` 引擎；Windows 双服务部署可同时使用 GPU YOLO 和该模型。以下单进程快速开始默认使用官方 `transformers` 识别模型，使用本仓库的微调识别模型请按[双服务部署](#61-上线拆成双服务推荐做法)启动。

```
图像/视频帧 ─▶ YOLO 检测车牌框 ─▶ 裁剪(+5% 边距) ─▶ PaddleOCR 识别字符 ─▶ 字符规整/校验 ─▶ 车牌文本
                                    │
                                    └─▶ 服务端绘制标注图 (data URL) + 分段耗时
```

两种使用方式：**HTTP API + Web 界面**（推荐）、**命令行单图识别**。

---

## 1. 快速开始：HTTP API + Web 界面

```powershell
conda activate <你的环境，例如 ML>          # 需已装 paddleocr/paddlex/paddle
cd code
pip install -r requirements-web.txt          # fastapi / uvicorn / python-multipart
python api_server.py --host 127.0.0.1 --port 8000
```

打开 <http://127.0.0.1:8000>：

- 📁 选择图片 / 拖拽 / `Ctrl+V` 粘贴 / 📷 拍照（含「视频连续识别」约 1.6s 一帧）
- 结果卡片：车牌大字（省份字高亮）、检测/识别置信度条、位置框、可选车牌裁剪图、一键复制
- **耗时显示**：每次识别显示 `YOLO 检测 / OCR 识别 / 合计`，并作为胶囊标签留在历史记录里；
  每块车牌另有自己的 `OCR xx ms`
- 状态胶囊：模型加载状态 + 当前推理后端（如 `模型就绪 · transformers@gpu:0`），悬停可看权重路径与提示
- 选图后**自动识别**；未选图直接点「开始识别」会弹出文件选择框

启动时会自动加载模型并**逐个后端组合实跑一次预热推理做校验**；任一组合失败会自动降级（不会因此起不来），
全部失败时服务仍可启动，接口返回 503 与具体原因。

模型自动定位顺序：

| 模型 | 定位顺序 |
| --- | --- |
| YOLO | `ANPR_YOLO_WEIGHTS` → `models/detection/best.pt` → `runs/ccpd_yolo/weights/best.pt`（前者不存在时） |
| 识别 | `ANPR_REC_MODEL_DIR` → `code/output/inference` → `code/output/best_accuracy/inference` → `code/output/latest/inference` → 官方预训练 `PP-OCRv6_small_rec` |

> `--host 127.0.0.1:8000` 这种「host 带端口」的写法也兼容（会自动拆分并告警）。

---

## 2. 命令行单图识别

```powershell
cd code
python recognize.py --source ..\CCPD2019\ccpd_base\xxx.jpg --show
python recognize.py --source img.jpg --conf 0.45 --device cpu
python recognize.py --source img.jpg --ocr-version PP-OCRv6_small `
                    --rec-model-dir <识别模型目录>
python recognize.py --source img.jpg --engine transformers --device gpu   # GPU 后端（见 §5）
```

输出示例：

```
[Timing] Models loaded in 24.83s (yolo=best.pt@0, ocr=transformers@gpu:0, rec=PP-OCRv6_small_rec)
Detected 1 plate(s) in img.jpg
Plate 1 [OK ]: 京A80293 (rec conf 1.00, det conf 0.87)
  └─ OCR: 316.4ms

[Timing] Detect: 71.1ms | OCR total: 316.4ms | Total: 449.7ms
Saved annotated image to: img_result.jpg
```

`recognize.py` 全部参数：

| 参数 | 说明 |
| --- | --- |
| `--source` | 输入图片路径（必填） |
| `--weights` | YOLO 权重，默认按上表自动定位 |
| `--ocr-version` | `PP-OCRv3/v4/v5/v6/PP-OCRv6_small`（见 `REC_MODEL_NAMES`） |
| `--rec-model-dir` | 自定义识别模型目录（PaddleX/PaddleOCR 推理导出：`inference.json` + `inference.pdiparams`；需与 `--ocr-version` 匹配，如 `PP-OCRv6_small`）。不指定则用官方预训练模型 |
| `--conf` | 检测置信度阈值，默认 0.5 |
| `--device` | `cpu` / `gpu` / `gpu:0` / `0`，默认自动 |
| `--engine` | OCR 推理后端：`transformers`（默认，GPU，见 §5）或 `paddle`（PaddleX/PaddleOCR，Windows 上为 CPU）；可用 `ANPR_ENGINE` 覆盖。加载失败的组合会自动回退 |
| `--no-warmup` | 跳过启动预热校验 |
| `--out` | 标注图输出路径（默认 `<name>_result.jpg`） |
| `--show` | 显示结果窗口 |

---

## 3. HTTP API

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/api/health` | 引擎/模型状态、加载耗时、实际生效的后端与提示 `notes` |
| POST | `/api/recognize` | multipart 上传图片：`file`, `conf`, `annotate`, `crops` |
| POST | `/api/recognize/base64` | JSON：`{"image": "<base64 或 dataURL>", "conf":0.5, "annotate":true, "crops":false}` |
| GET | `/api/demo` | 对仓库根目录示例图（`image.png` / `image_result.jpg`）跑一次识别 |
| GET | `/docs` | FastAPI 交互式文档（OpenAPI: `/openapi.json`） |
| GET | `/` | Web 前端（`code/web/index.html`） |

```bash
curl -X POST http://127.0.0.1:8000/api/recognize \
     -F "file=@image.png" -F "conf=0.5" -F "annotate=true" -F "crops=true"
```

返回结构（`crop` / `annotated_image` 为 JPEG data URL，未请求时为 `null`）：

```json
{
  "success": true,
  "count": 1,
  "image_size": [1160, 720],
  "plates": [{
    "index": 0,
    "bbox": [452, 371, 744, 458],
    "det_conf": 0.871,
    "raw_text": "京A80293",
    "plate": "京A80293",
    "valid": true,
    "rec_conf": 0.9997,
    "ocr_ok": true,
    "ocr_error": null,
    "ocr_ms": 604.3,
    "crop": "data:image/jpeg;base64,..."
  }],
  "annotated_image": "data:image/jpeg;base64,...",
  "timings_ms": {
    "detect_ms": 66.9,
    "ocr_total_ms": 604.3,
    "ocr_per_plate_ms": [604.3],
    "total_ms": 812.5,
    "request_ms": 830.1
  },
  "model": { "loaded": true, "engine": "paddle", "ocr_device": "auto", "notes": [] }
}
```

`/api/health` 关键字段：`engine.loaded` / `engine.error` / `engine.engine` / `engine.yolo_device` /
`engine.ocr_device` / `engine.rec_model_dir` / `engine.notes`。

---

## 4. 环境变量

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `ANPR_HOST` / `ANPR_PORT` | `127.0.0.1` / `8000` | 监听地址 |
| `ANPR_YOLO_WEIGHTS` | `models/detection/best.pt` | YOLO 权重，可由环境变量覆盖 |
| `ANPR_REC_MODEL_DIR` | 自动探测 `code/output/*/inference` | 自定义识别模型目录（PaddleX/PaddleOCR 推理导出）；留空则用官方预训练模型 |
| `ANPR_OCR_VERSION` | `PP-OCRv6_small` | 识别模型版本（见 `REC_MODEL_NAMES`） |
| `ANPR_DEVICE` | `gpu`（服务）/ 自动（CLI） | 设备：`cpu` / `gpu` / `gpu:0` / `0` |
| `ANPR_ENGINE` | `transformers` | OCR 推理后端：`transformers`（GPU，需 safetensors 模型，见 §5）或 `paddle`（PaddleX/PaddleOCR，Windows 上为 CPU） |
| `ANPR_CONF` | `0.5` | 默认检测置信度 |
| `ANPR_PRELOAD` | `1` | 启动即加载模型；`0` 为首次请求时懒加载 |
| `ANPR_WARMUP` | `1` | 启动时用一张合成图**实跑一次**校验后端；`0` 跳过 |
| `ANPR_MKLDNN` | `0` | 是否开启 Paddle MKLDNN 加速 |

命令行等价开关：`--engine {paddle,transformers}`、`--device`、`--no-warmup`。

---

## 5. GPU 推理（transformers 后端）

Windows 上 `paddlepaddle` 3.x 没有 GPU wheel，若想让 OCR 走 GPU，可用 PaddleX 的
`transformers` 后端（由 PyTorch/CUDA 驱动，需安装 `transformers`）：

```powershell
python api_server.py --host 127.0.0.1 --port 8000 --engine transformers --device gpu
# 或：$env:ANPR_ENGINE="transformers"; $env:ANPR_DEVICE="gpu"; python api_server.py
```

**务必注意模型格式**：`transformers` 后端只能加载 **safetensors/HF 格式**模型。官方
`PP-OCRv6_*` 两种格式都提供（首次运行会下载 `*_safetensors`），所以默认的官方
`PP-OCRv6_small_rec` 可以直接走 GPU；而 PaddleX/PaddleOCR 的推理导出是 **PIR 格式**
（`inference.json` + `inference.pdiparams`），两者不兼容。当 `ANPR_REC_MODEL_DIR` 指向
这样的目录时：

- **想要 GPU**：引擎先用官方 `PP-OCRv6_small_rec` 跑 `transformers`，并在日志与 `/api/health`
  的 `notes` 中说明该目录无法被该后端加载；
- **想要该目录里的模型**：引擎随后自动降级到 `paddle` 引擎加载它（Windows 上为 CPU）。

引擎启动时会对 `(engine, device)` 逐个组合实跑一次预热推理做校验，顺序为
`transformers → paddle → cpu`，第一个成功的生效；失败原因逐条写入日志。
跳过校验用 `--no-warmup` / `ANPR_WARMUP=0`。

---

## 6. 识别模型微调（CCPD 车牌）

官方 `PP-OCRv6_small_rec` 在 CCPD 车牌裁剪上整牌全对率为 **29.73%**
（清晰子集 58.9%，blur / tilt / rotate 均不足 2%）。微调在 `anpr-ocr` 环境里做
（CUDA paddle、无 torch，因此没有 DLL 冲突）。

**已完成一轮微调（30 epoch / 40k 裁剪，训练侧 best val acc 96.07%）的实际收益**：

| 评测口径 | 官方模型 | 微调模型 |
| --- | --- | --- |
| 裁剪图整牌全对（4000 张 val） | 29.73% | **96.03%** |
| 裁剪图字符准确率 | 50.96% | **99.22%** |
| 端到端 YOLO+识别（144 张 CCPD，分层） | 18.75% | **96.53%** |
| 端到端吞吐 | 13.7 img/s | 13.8 img/s |

分层看提升集中在困难子集：blur 0.5%→93.25%、tilt 0.8%→88.0%、challenge 5.0%→93.8%、
rotate 1.5%→95.5%（端到端口径 blur/tilt/challenge/rotate 分别 0%/5.6%/0%/0% → 94.4%/83.3%/100%/100%）。

复现步骤：

```powershell
conda activate anpr-ocr          # 或直接用 E:\CondaEnvs\anpr-ocr\python.exe
cd code
python make_plate_rec_dataset.py --out ..\rec_data_plate --train-samples 40000 --val-samples 4000
python train_ocr.py --smoke      # 20 秒冒烟：验证数据/配置/显存/评估链路
python train_ocr.py              # 正式微调（默认 30 epoch，约 5–7 小时）
python train_ocr.py --export-only                                 # 导出 PIR -> output_plate/inference
python verify_rec_plate.py --dataset ..\rec_data_plate --split val --device gpu:0 `
       --model-dir output_plate\inference --tag finetuned          # 与基准线对比
```

> 注意：`verify_rec_plate.py` 走的是线上同一个 `TextRecognition` 模块，它只能加载**推理导出**
> （`inference.json` + `inference.pdiparams`），不能直接读 `best_accuracy.pdparams` 训练权重；
> 所以先 `--export-only` 再验证。训练过程中的 val acc 看 `output_plate/train.log`。

要点：

- **数据集**：标签来自 CCPD 文件名的 7 位索引（`0_0_22_10_26_29_24` → `皖AC9M60`），
  裁剪用与线上完全一致的 **+5% padding**；8 个子集分层抽样，困难子集（blur/tilt/weather/challenge）
  刻意加权；跨子集重复图片已去重，train/val 不串。
- **字表**：用官方 `ppocrv6_dict.txt`（18708 字 → CTC 头 18710），与预训练**维度完全对齐**，
  加载预训练时零 shape warning，头不需要重新学（换成 67 字小词表会让头随机初始化，收敛慢且更难部署）。
- **配置**：`code/configs/plate_rec.yml`（架构保持上游不变，只改数据路径/bs/lr）；
  `train_ocr.py` 每次会写 `output_plate/run_config.yml`，运行可复现。8 GB 显存用 `--batch-size 64`
  （上游默认 128 会爆）。
- **评估**：`verify_rec_plate.py` 用**与线上同一个** `TextRecognition` 模块跑，输出整牌全对率、
  逐位字符准确率、按子集分层表；`--dump` 保存每个样本的预测，便于前后对比。
  `eval_e2e.py` 则跑**完整链路**（YOLO 检测 + 识别）并与 CCPD 文件名里的真值比对，
  可带 `--rec-url` 走远端微调模型、不带则走本地官方模型，用于 A/B。
- 本地评测明细可保存在 `code/eval_reports/`，该目录不纳入版本控制。

### 6.1 上线：拆成双服务（推荐做法）

微调模型是 **PIR** 导出，只有 `paddle` 引擎能加载，而 `import paddle` 在 ML 环境会被 CUDA torch
挡住（DLL 冲突）。因此拆成两个进程：检测留在 ML 环境（GPU torch），识别放进 `anpr-ocr`（GPU paddle），
用一次 `start_anpr.bat` 拉起。启动脚本中的两个 Python 路径是当前机器的示例，换机器时先修改 `PY_REC` 和 `PY_API`；两个环境分别需要 PaddleOCR/PaddleX/Paddle 与 YOLO/torch，均需要 FastAPI、Uvicorn，API 还需要 `python-multipart`：

```
┌───────────────────────────────┐       HTTP          ┌────────────────────────────────┐
│ ML 环境 (E:\CondaEnvs\ML)      │  POST /api/rec      │ anpr-ocr 环境                   │
│ api_server.py         :8000   │ ──────────────────▶ │ rec_service.py       :8100     │
│ YOLO 检测 (CUDA torch, GPU)    │  base64 车牌裁剪图   │ 微调 PP-OCRv6_small_rec (PIR)   │
│ Web UI / REST                 │ ◀────────────────── │ paddle 引擎 (GPU)               │
└───────────────────────────────┘    {text, score}    └────────────────────────────────┘
```

```powershell
E:\YOLO-Paddle-ANPR\start_anpr.bat     # 先启动识别服务并等它就绪，再启动 API
E:\YOLO-Paddle-ANPR\stop_anpr.bat      # 按端口精确停掉两个服务（含窗口）

# 等价的手动启动
E:\CondaEnvs\anpr-ocr\python.exe code\rec_service.py --model-dir models\recognition --port 8100
E:\CondaEnvs\ML\python.exe       code\api_server.py  --port 8000 --rec-url http://127.0.0.1:8100
```

- **识别服务接口**：`GET /health`；`POST /api/rec`
  （`{"images": ["<base64 jpg>", ...]}` → `{"results": [{"text","score","ms"}], "elapsed_ms"}`）。
- **开关**：设置 `ANPR_REC_URL`（或 `--rec-url`）后主 API **不再加载本地识别模型**，每个裁剪图走远端；
  留空即原来的单进程行为（官方模型 / 本地 `--rec-model-dir`）。
- **不做静默回退**（按需求）：识别服务未就绪时 `/api/health` 报 `engine.loaded=false` 且 `error`
  写明服务地址与原因，`/api/recognize` 直接返回 **503**；运行中远端断开时该车牌 `ocr_ok=false`
  并带回具体错误，服务恢复后自动可用（`/api/health` 每次实时探测远端，`engine.rec_remote` 是它的健康摘要）。
- **开销与资源**：单张多约 5–10 ms（base64 + HTTP 往返），端到端吞吐与单进程持平（13.8 img/s）；
  显存 YOLO ~1–2GB + paddle 识别 ~1–2GB，8GB 卡足够，紧张时给识别服务加 `--device cpu`
  （官方模型实测 CPU ~80 ms/张）。

---

## 7. 代码结构

```
code/
  anpr_engine.py          # 核心引擎：模型单例、RLock 串行化、后端/设备校验与降级、
                          # 字符表过滤 + 车牌格式校验、分段耗时、JSON 就绪输出
  api_server.py           # FastAPI 服务（/api/* + 托管 web/ 静态页面）
  web/index.html          # 单文件前端（上传/拖拽/粘贴/摄像头/连续识别 + 分段耗时展示）
  recognize.py            # 命令行单图识别（复用引擎）
  train.py                # YOLO 检测模型训练
  process_dataset.py      # CCPD2019 → YOLO 标签 + dataset/data.yaml
  tools.py                # 文件名解析 Stringparsing + 标注可视化
  make_plate_rec_dataset.py  # CCPD2019 → 车牌识别数据集（train/val.txt + crop_img + dict）
  train_ocr.py            # 识别模型微调入口（写 run_config.yml，调用 PaddleOCR/tools/train.py）
  verify_rec_plate.py     # 识别评估：整牌全对率 / 逐位准确率 / 分层表
  configs/plate_rec.yml   # 微调配置（上游 PP-OCRv6_small_rec.yml + 本项目数据与超参）
PaddleOCR/                # 官方仓库克隆（tools/train.py、tools/export_model.py）
rec_data_plate/           # 车牌识别数据集（40k train / 4k val 裁剪图）
models/detection/         # 已训练 YOLO 检测权重
models/recognition/        # 已训练 PaddleOCR 推理模型
runs/                     # 本地训练输出，不纳入版本控制
```

---

## 8. 优化说明

相对最初的 `recognize.py` 单图脚本：

1. **共享引擎**：CLI 与 HTTP API 用同一套推理代码，模型每进程只加载一次。
2. **修复崩溃**：旧脚本末尾引用了未定义的 `yolo_time`（NameError）；已改为完整分段计时。
3. **OCR 结果解析修复**：`rec_boxes` 是 numpy 数组，`x or []` 会抛
   `truth value of an array is ambiguous`；现在统一安全转换并按左右顺序拼接多行文本。
4. **字符规整**：按识别模型字表过滤噪声字符，校验「省份字 + 字母 + 数字」的车牌格式，给出 `valid` 标记。
5. **小图放大**：宽度 <220px 的裁剪图自动放大后再送 OCR（远距离车牌更稳）。
6. **后端可配置 + 自愈**：`paddle` / `transformers` 后端、`cpu` / `gpu` 设备可配置；
   启动预热校验 + 依次降级，兼容性问题不再导致服务起不来。
7. **并发安全与失败隔离**：PaddleX 预测器非线程安全 → 引擎内加锁串行化；
   单块车牌 OCR 失败不影响整次请求，错误原因随结果返回并展示在界面上。
8. **去掉重复检测（识别模块化）**：改用 PaddleOCR 的独立识别模块 `TextRecognition`
   替代 `PaddleOCR` 流水线 —— YOLO 已经定位好车牌，流水线内部的 `PP-OCRv6_medium_det`
   只是把同一条文本行再检测一遍。实测同一张裁剪图：det+rec 87ms → rec-only 20ms，
   文本完全一致；接口单张 OCR 148–403ms → 27–31ms，整请求 284–616ms → 136–148ms，
   启动加载 46s → 21s（少加载一个检测模型）。

### 已知限制

- 识别走 PaddleOCR 的独立识别模块 `TextRecognition`（不是 `PaddleOCR` 流水线），
  每个 YOLO 裁剪图只跑一次识别、不再重复检测；该模块解码为空时返回的 `rec_score`
  是 `nan`，引擎已归一化为 `0.0`（JSON 不允许 NaN）。
- 首次运行会联网下载官方模型到 `~/.paddlex/official_models`（`transformers` 后端则下载
  `*_safetensors`）。`ANPR_REC_MODEL_DIR` 只替换识别模型，检测模型仍是官方默认。
- Windows 上 paddle 3.x 无 GPU wheel，`paddle` 引擎只能跑 CPU；并且若环境里的 torch 带 CUDA，
  `import paddle` 会直接失败（`cublas64_12.dll` / WinError 127），此时 `paddle` 的所有组合都不可用 →
  用默认的 `transformers` 引擎（GPU + 官方 safetensors 模型，见 §5）即可，`transformers` 全程不 import paddle。
- 浏览器调用摄像头需要 HTTPS 或 localhost（本机 `http://127.0.0.1:8000` 满足）。
- 不要在 `import paddle` 之后再 import torch（DLL 搜索路径冲突会崩 torch）：
  `anpr_engine.py` 已按 `ultralytics → paddleocr` 的顺序导入，其他模块请勿打乱。
- 不要用多 worker 启动服务：每个 worker 都会加载一份完整模型（`--workers` 默认 1）。

---

## 9. 常见问题（FAQ）

| 现象 | 原因 / 处理 |
| --- | --- |
| `paddleocr` / `paddlex` 导入失败（`OSError`、`PDX has already been initialized`） | 只影响 `paddle` 引擎：环境里的 torch 带 CUDA 时，`import paddle` 会因 `cublas64_12.dll`（WinError 127）失败，`paddle` 的所有组合都不可用；而 `transformers` 引擎全程不 import paddle，照常可用（服务与 CLI 默认就是它）。确实需要 `paddle` 引擎时，在干净进程里验证 `python -c "import paddleocr, paddlex"` 并补齐依赖，且不要在 `import paddle` 之后再 import torch（见「已知限制」的 DLL 冲突条目） |
| `RuntimeError: PDX has already been initialized` | 同一进程里 paddlex 被重复导入，或上一次初始化失败留下的半初始化状态（常与 DLL 冲突同源）：重启服务进程重试，并保持 `anpr_engine.py` 的导入顺序（`ultralytics → paddleocr`）不被改动 |
| `[Errno 11001] getaddrinfo failed` 后服务退出 | 把端口写进了 host（如 `--host 127.0.0.1:8000`）。现已自动拆分；规范写法是 `--host 127.0.0.1 --port 8000` |
| `ValueError: No valid model files were found for engine 'transformers'` | 该后端只吃 safetensors/HF 模型，而 `ANPR_REC_MODEL_DIR` 指向的是 PIR 导出（`inference.json` + `inference.pdiparams`）。引擎会改用官方识别模型（GPU）；要用那个目录就设 `--engine paddle`（Windows 上为 CPU） |
| 卡片提示「未能识别字符」/ 识别置信度 0% | 车牌过小或模糊，超出官方 `PP-OCRv6_small_rec` 的能力：先降低检测阈值确认裁剪图质量，或换 `--ocr-version PP-OCRv6`（更重的识别模型）对照 |
| 改了 `web/index.html` 但界面没变化 | 浏览器缓存：`Ctrl+F5` 强制刷新；服务端无需重启（静态文件） |
| 首次启动很慢 | 需要下载官方模型 + 加载权重 + 预热校验，属正常；可用 `ANPR_PRELOAD=0` 改为首次请求加载 |
| 换模型/换环境后仍用旧模型 | 检查 `/api/health` 的 `rec_model_dir` 与 `notes`，确认实际生效的权重与后端 |
