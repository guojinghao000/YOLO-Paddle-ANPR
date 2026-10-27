"""Fine-tune PP-OCRv6_small_rec on CCPD2019 license plates.

The upstream entry point (PaddleOCR/tools/train.py) is kept, but this wrapper
owns the configuration: it loads code/configs/plate_rec.yml, applies the CLI
overrides and writes a self-contained run config next to the checkpoints, so
every run is reproducible from one file.

Use the torch-free anpr-ocr env (CUDA paddle, no torch -> no DLL conflict):

    E:\\CondaEnvs\\anpr-ocr\\python.exe train_ocr.py                 # full run
    E:\\CondaEnvs\\anpr-ocr\\python.exe train_ocr.py --smoke         # 1 epoch, few crops
    E:\\CondaEnvs\\anpr-ocr\\python.exe train_ocr.py --resume        # continue from latest
    E:\\CondaEnvs\\anpr-ocr\\python.exe train_ocr.py --export-only   # export PIR for serving

Typical flow (dataset is built by make_plate_rec_dataset.py):

    python make_plate_rec_dataset.py --out ..\\rec_data_plate --train-samples 40000 --val-samples 4000
    python train_ocr.py --smoke          # proves the pipeline in ~2 min
    python train_ocr.py                  # the real run (30 epochs)

Serving a finished run (paddle engine, rec-only module):

    ANPR_ENGINE=paddle  ANPR_REC_MODEL_DIR=<output>/inference  ANPR_OCR_VERSION=PP-OCRv6_small
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
CLONE = PROJECT_ROOT / "PaddleOCR"
TRAIN_PY = CLONE / "tools" / "train.py"
EXPORT_PY = CLONE / "tools" / "export_model.py"
DEFAULT_CONFIG = BASE_DIR / "configs" / "plate_rec.yml"
DEFAULT_DATASET = PROJECT_ROOT / "rec_data_plate"
DEFAULT_OUTPUT = BASE_DIR / "output_plate"
DEFAULT_PRETRAINED = PROJECT_ROOT / "PP-OCRv6_small_rec_pretrained.pdparams"
SMOKE_OUTPUT = BASE_DIR / "output_plate_smoke"


def posix(path: Path) -> str:
    """PaddleOCR configs are happier with forward slashes on Windows."""
    return str(Path(path).resolve()).replace("\\", "/")


def env_report() -> None:
    print(f"python      : {sys.executable}")
    print(f"cwd target  : {CLONE}")
    try:
        import paddle
        print(f"paddle      : {paddle.__version__}  cuda={paddle.device.is_compiled_with_cuda()}")
        if paddle.device.is_compiled_with_cuda():
            try:
                print(f"gpu         : {paddle.device.cuda.get_device_name(0)} "
                      f"(count={paddle.device.cuda.device_count()})")
            except Exception:                       # noqa: BLE001
                pass
        else:
            print("WARNING: this paddle build has no CUDA - training will be very slow")
    except Exception as exc:                        # noqa: BLE001
        sys.exit(f"cannot import paddle: {type(exc).__name__}: {exc}")
    try:
        import torch                                # noqa: F401
        print("WARNING: torch is importable in this env; paddle+torch CUDA usually "
              "conflict on Windows. Prefer the anpr-ocr env.")
    except ImportError:
        pass


def count_lines(path: Path) -> int:
    with path.open("rb") as fh:
        return sum(1 for _ in fh)


def make_smoke_labels(dataset: Path, out_dir: Path, n_train: int, n_val: int):
    """First-N-lines copies of the real label files (same data_dir, so the
    relative crop paths inside stay valid)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    made = {}
    for split, n in (("train", n_train), ("val", n_val)):
        src = dataset / f"{split}.txt"
        if not src.exists():
            sys.exit(f"missing label file: {src}")
        lines = src.read_text(encoding="utf-8").splitlines()[:n]
        dst = out_dir / f"{split}_smoke.txt"
        dst.write_text("\n".join(lines) + "\n", encoding="utf-8")
        made[split] = dst
        print(f"smoke labels {split}: {len(lines)} lines -> {dst}")
    return made


def build_run_config(args) -> tuple[Path, dict]:
    base = Path(args.config)
    if not base.exists():
        sys.exit(f"config not found: {base}")
    cfg = yaml.safe_load(base.read_text(encoding="utf-8"))

    dataset = Path(args.dataset).resolve()
    output = Path(args.output).resolve()
    dict_path = dataset / "dict.txt"
    train_txt, val_txt = dataset / "train.txt", dataset / "val.txt"

    for p, what in ((dataset, "dataset dir"), (dict_path, "dataset dict"),
                    (train_txt, "train.txt"), (val_txt, "val.txt")):
        if not p.exists():
            sys.exit(f"{what} not found: {p}\n"
                     f"build it first: python make_plate_rec_dataset.py --out {dataset}")
    pretrained = Path(args.pretrained).resolve()
    if not pretrained.exists():
        sys.exit(f"pretrained weights not found: {pretrained}")

    label_train, label_val = train_txt, val_txt
    if args.smoke:
        made = make_smoke_labels(dataset, output, args.smoke_train, args.smoke_val)
        label_train, label_val = made["train"], made["val"]

    g = cfg["Global"]
    g["use_gpu"] = not args.cpu
    g["distributed"] = False
    g["epoch_num"] = args.epochs
    g["pretrained_model"] = posix(pretrained)
    g["character_dict_path"] = posix(dict_path)
    g["save_model_dir"] = posix(output)
    g["save_res_path"] = posix(output / "predicts.txt")
    g["print_batch_step"] = args.print_step
    g["eval_batch_step"] = [0, args.eval_step]
    g["save_epoch_step"] = args.save_epoch_step
    g["checkpoints"] = posix(output / "latest") if args.resume else None
    g["save_inference_dir"] = None            # exported explicitly via --export

    cfg["Optimizer"]["lr"]["learning_rate"] = args.lr
    cfg["Optimizer"]["lr"]["warmup_epoch"] = args.warmup_epochs

    for section, label in (("Train", label_train), ("Eval", label_val)):
        cfg[section]["dataset"]["data_dir"] = posix(dataset)
        cfg[section]["dataset"]["label_file_list"] = [posix(label)]
    cfg["Train"]["sampler"]["first_bs"] = args.batch_size
    cfg["Train"]["loader"]["batch_size_per_card"] = args.batch_size
    cfg["Train"]["loader"]["num_workers"] = args.workers
    cfg["Eval"]["loader"]["batch_size_per_card"] = args.batch_size
    cfg["Eval"]["loader"]["num_workers"] = max(0, args.workers // 2)

    for item in args.opt or []:
        if "=" not in item:
            sys.exit(f"--opt expects key=value, got {item!r}")
        key, _, value = item.partition("=")
        node = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        try:
            node[parts[-1]] = yaml.safe_load(value)
        except Exception:                            # noqa: BLE001
            node[parts[-1]] = value

    output.mkdir(parents=True, exist_ok=True)
    run_cfg = output / "run_config.yml"
    run_cfg.parent.mkdir(parents=True, exist_ok=True)
    run_cfg.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False),
                       encoding="utf-8")
    return run_cfg, cfg


def run(cmd: list[str], log_path: Path) -> int:
    print("\n$ " + " ".join(cmd))
    print(f"(log -> {log_path})")
    t0 = time.perf_counter()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", errors="replace") as fh:
        proc = subprocess.Popen(cmd, cwd=str(CLONE), stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True,
                                encoding="utf-8", errors="replace", bufsize=1)
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            fh.write(line)
            fh.flush()
        proc.wait()
    print(f"\nexit code {proc.returncode} in {time.perf_counter() - t0:.0f}s")
    return proc.returncode


def main():
    ap = argparse.ArgumentParser(description="Fine-tune PP-OCRv6_small_rec on CCPD plates")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--dataset", default=str(DEFAULT_DATASET))
    ap.add_argument("--output", default=None,
                    help=f"checkpoint dir (default {DEFAULT_OUTPUT}, smoke: {SMOKE_OUTPUT})")
    ap.add_argument("--pretrained", default=str(DEFAULT_PRETRAINED))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64,
                    help="batch size per card (8 GB card: 64 is safe, 128 is not)")
    ap.add_argument("--lr", type=float, default=5.0e-4)
    ap.add_argument("--warmup-epochs", type=float, default=2.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--eval-step", type=int, default=500,
                    help="evaluate every N iterations (500 ~= once per epoch at 40k crops)")
    ap.add_argument("--print-step", type=int, default=20)
    ap.add_argument("--save-epoch-step", type=int, default=5)
    ap.add_argument("--resume", action="store_true", help="continue from <output>/latest")
    ap.add_argument("--cpu", action="store_true", help="force use_gpu=False (slow)")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny 1-epoch run to validate the pipeline")
    ap.add_argument("--smoke-train", type=int, default=300)
    ap.add_argument("--smoke-val", type=int, default=60)
    ap.add_argument("--export", action="store_true", help="export PIR after training")
    ap.add_argument("--export-only", action="store_true",
                    help="only export the PIR model from the best checkpoint")
    ap.add_argument("--opt", action="append", metavar="KEY=VALUE",
                    help="extra config override, repeatable (e.g. --opt Global.epoch_num=5)")
    args = ap.parse_args()

    if args.smoke:
        args.output = args.output or str(SMOKE_OUTPUT)
        args.epochs = min(args.epochs, 1)
        args.batch_size = min(args.batch_size, 16)
        args.workers = 0
        args.eval_step = min(args.eval_step, 30)
        args.print_step = min(args.print_step, 10)
        args.save_epoch_step = 1
    args.output = args.output or str(DEFAULT_OUTPUT)

    if not TRAIN_PY.exists():
        sys.exit(f"PaddleOCR clone not found at {CLONE} (expected {TRAIN_PY})")

    print("=" * 78)
    env_report()
    print("=" * 78)

    run_cfg, cfg = build_run_config(args)
    n_train = count_lines(Path(cfg["Train"]["dataset"]["label_file_list"][0]))
    n_val = count_lines(Path(cfg["Eval"]["dataset"]["label_file_list"][0]))
    bs = cfg["Train"]["loader"]["batch_size_per_card"]
    iters = max(1, n_train // max(1, bs))
    print(f"dataset     : {cfg['Train']['dataset']['data_dir']}")
    print(f"             : {n_train} train / {n_val} val crops  (bs={bs} -> ~{iters} iters/epoch)")
    print(f"dict        : {cfg['Global']['character_dict_path']}")
    print(f"pretrained  : {cfg['Global']['pretrained_model']}")
    print(f"output      : {cfg['Global']['save_model_dir']}")
    print(f"epochs      : {cfg['Global']['epoch_num']}   lr={cfg['Optimizer']['lr']['learning_rate']}"
          f"  warmup={cfg['Optimizer']['lr']['warmup_epoch']}  gpu={cfg['Global']['use_gpu']}")
    print(f"run config  : {run_cfg}")

    if args.export_only:
        best = pick_checkpoint(Path(cfg["Global"]["save_model_dir"]))
        return do_export(run_cfg, best)

    code = run([sys.executable, str(TRAIN_PY), "-c", str(run_cfg)],
               Path(cfg["Global"]["save_model_dir"]) / "train_stdout.log")
    if code != 0:
        sys.exit(code)

    if args.export:
        do_export(run_cfg, pick_checkpoint(Path(cfg["Global"]["save_model_dir"])))
    return 0


def pick_checkpoint(output: Path) -> Path:
    """Prefer best_accuracy.pdparams, fall back to latest.pdparams."""
    for name in ("best_accuracy", "latest"):
        cand = output / name
        if Path(str(cand) + ".pdparams").exists():
            return cand
    sys.exit(f"no checkpoint (*.pdparams) in {output} - train first")


def do_export(run_cfg: Path, best: Path) -> int:
    out = Path(best).parent / "inference"
    cmd = [sys.executable, str(EXPORT_PY), "-c", str(run_cfg),
           "-o", f"Global.pretrained_model={posix(best)}",
           f"Global.save_inference_dir={posix(out)}"]
    code = run(cmd, Path(best).parent / "export_stdout.log")
    if code == 0:
        print(f"\nPIR model exported -> {out}")
        print("serve it with:")
        print(f"  set ANPR_ENGINE=paddle")
        print(f"  set ANPR_REC_MODEL_DIR={out}")
        print(f"  set ANPR_OCR_VERSION=PP-OCRv6_small")
        print("then run api_server.py inside the anpr-ocr env (paddle is importable there).")
    return code


if __name__ == "__main__":
    sys.exit(main())
