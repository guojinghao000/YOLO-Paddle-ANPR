import argparse
import logging
import sys
import time as _time
from pathlib import Path
from ultralytics import YOLO
from ultralytics.utils import LOGGER


def log_metrics(trainer):
    epoch = trainer.epoch + 1
    metrics = trainer.metrics
    if metrics:
        loss = metrics.get("train/box_loss", 0) + metrics.get("train/cls_loss", 0) + metrics.get("train/dfl_loss", 0)
        mAP50 = metrics.get("metrics/mAP_0.5", 0)
        mAP50_95 = metrics.get("metrics/mAP_0.5:0.95", 0)
        LOGGER.info(f"Epoch {epoch:3d} | loss: {loss:.4f} | mAP@0.5: {mAP50:.4f} | mAP@0.5:0.95: {mAP50_95:.4f}")


if __name__ == '__main__':
    # 参数读取
    parser = argparse.ArgumentParser(description="YOLO license plate training")

    # 轮数
    parser.add_argument("--epochs", type=int, default=50, help="total epochs to train")
    # 训练时间 为0时使用epochs训练
    parser.add_argument("--time", type=float, default=0, help="max training hours (0 = epochs-based)")
    # 继续训练，从最新的checkpoint继续训练
    parser.add_argument("--resume", action="store_true", help="resume from last checkpoint in project/name")
    # 训练数据导出位置
    parser.add_argument("--project", type=str, default=None, help="project directory (default: ../runs)")
    parser.add_argument("--name", type=str, default="ccpd_yolo", help="experiment name")
    args = parser.parse_args()

    # 根目录
    BASE_DIR = Path(__file__).resolve().parent
    # 数据集目录和实验目录
    DATASET_DIR = BASE_DIR.parent / "dataset"
    PROJECT_DIR = Path(args.project) if args.project else BASE_DIR.parent / "runs"
    EXPERIMENT_DIR = PROJECT_DIR / args.name
    DATA_YAML = DATASET_DIR / "data.yaml"

    log_file = BASE_DIR / "training.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger(__name__)

    # yolo日志输出到文件和控制台
    LOGGER.handlers = []
    LOGGER.addHandler(logging.FileHandler(log_file, encoding='utf-8'))
    LOGGER.addHandler(logging.StreamHandler(sys.stdout))


    # 如果采取继续训练模式，尝试从上次的checkpoint继续训练，否则从yolo26n.pt开始训练
    if args.resume:
        last_ckpt = EXPERIMENT_DIR / "weights" / "last.pt"
        if last_ckpt.exists():
            logger.info("Resuming from checkpoint: %s", last_ckpt)
            model = YOLO(str(last_ckpt))
        else:
            logger.warning("No checkpoint found at %s, starting fresh from yolo26n.pt", last_ckpt)
            model = YOLO(str(BASE_DIR / "yolo26n.pt"))
            args.resume = False
    else:
        model = YOLO(str(BASE_DIR / "yolo26n.pt"))

    train_params = {
        "data": str(DATA_YAML),
        "epochs": args.epochs,
        "imgsz": 640,
        "batch": -1,
        "workers": 0,
        "device": "0",# 使用GPU 0进行训练，如果没有GPU可以改为"cpu"
        "cache": False,# 是否缓存数据集到内存中，False表示不缓存，True表示缓存
        "optimizer": "SGD",# 优化器类型，支持"SGD"和"Adam"，这里使用SGD
        "lr0": 0.01,# 初始学习率，训练开始时的学习率
        "momentum": 0.937,# 动量因子，用于加速SGD优化器的收敛速度
        "weight_decay": 0.0005,# 权重衰减系数，用于防止过拟合
        "cos_lr": True,# 是否使用余弦退火学习率调度器，True表示使用，False表示不使用
        "warmup_epochs": 3,# 预热阶段的轮数，在训练开始时逐渐增加学习率，帮助模型更稳定地收敛
        "augment": True,# 是否使用数据增强，True表示使用，False表示不使用
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "degrees": 10.0,
        "translate": 0.1,
        "scale": 0.5,
        "shear": 2.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.5,
        "mosaic": 1.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "close_mosaic": 1,
        "save": True,
        "save_period": 5,
        "val": True,
        "plots": True,
        "project": str(PROJECT_DIR),
        "name": args.name,
        "exist_ok": True,
        "pretrained": True,
        "verbose": True,
    }

    if args.time > 0:
        _train_start = _time.time()
        def _time_check(trainer):
            elapsed = _time.time() - _train_start
            if elapsed > args.time * 3600:
                LOGGER.info("Time limit %.1f h reached after epoch %d, saving checkpoint...", args.time, trainer.epoch + 1)
                trainer.stop = True
        model.add_callback("on_train_epoch_end", _time_check)

    if args.resume:
        train_params["resume"] = True

        def _fix_epochs(trainer):
            if trainer.args.epochs != args.epochs:
                trainer.args.epochs = args.epochs
                trainer.epochs = args.epochs
                trainer._setup_scheduler()
                trainer.scheduler.last_epoch = trainer.start_epoch - 1
        model.add_callback("on_train_start", _fix_epochs)

    logger.info("=" * 60)
    if args.time > 0:
        logger.info("YOLO training (%s, max %.1f h)", "resume" if args.resume else "fresh", args.time)
    else:
        logger.info("YOLO training (%s, %d epochs)", "resume" if args.resume else "fresh", args.epochs)
    logger.info("Model: %s", model.ckpt_path)
    logger.info("Data: %s", DATA_YAML)
    logger.info("Params: %s", train_params)
    logger.info("=" * 60)

    model.add_callback("on_train_epoch_end", log_metrics)

    results = model.train(**train_params)

    if hasattr(results, 'metrics'):
        logger.info("Training complete! Final metrics: %s", results.metrics)
    else:
        logger.info("Training complete!")

    logger.info("Log saved to: %s", log_file)
