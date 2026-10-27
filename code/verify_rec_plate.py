"""Evaluate a PaddleOCR recognition module on a plate dataset (and validate the
dataset labels themselves).

Runs in the torch-free ``anpr-ocr`` env through the standalone
``paddleocr.TextRecognition`` module — the same module the ANPR engine uses.

Examples
--------
# 1) validate freshly built CCPD labels against the official model
python verify_rec_plate.py --dataset ../rec_data_preview --split val --label-check

# 2) baseline vs fine-tuned on the real val split (same crops, per-subset table)
python verify_rec_plate.py --dataset ../rec_data_plate --split val --tag official
python verify_rec_plate.py --dataset ../rec_data_plate --split val --model-dir ../code/output_plate/best_accuracy/inference --tag finetuned

Metrics: exact match (整牌全对), per-character accuracy, per-position accuracy
(position 0 = province, 1 = letter, 2..6 = letters/digits) and mean score.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np

PLATE_ALLOWED = set("京津冀晋蒙辽吉黑沪苏浙皖闽赣鲁豫鄂湘粤桂琼渝川贵云藏陕甘青宁新警学"
                    "ABCDEFGHJKLMNPQRSTUVWXYZ0123456789")


def normalize(text: str) -> str:
    s = unicodedata.normalize("NFKC", str(text)).upper()
    s = s.replace(" ", "").replace("·", "").replace("-", "").replace(".", "")
    return "".join(c for c in s if c in PLATE_ALLOWED)


def load_split(dataset: Path, split: str, limit: int | None):
    rows = []
    for line in (dataset / f"{split}.txt").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or "\t" not in line:
            continue
        rel, label = line.rsplit("\t", 1)
        rows.append((dataset / rel, normalize(label)))
    if limit:
        rows = rows[:limit]
    return rows


def position_report(preds, labels, width=7):
    """Per-position character accuracy (ragged strings handled)."""
    hit = [0] * width
    total = [0] * width
    for p, g in zip(preds, labels):
        for i in range(width):
            if i >= len(g):
                continue
            total[i] += 1
            if i < len(p) and p[i] == g[i]:
                hit[i] += 1
    return [(i, hit[i], total[i], (hit[i] / total[i] if total[i] else 0.0))
            for i in range(width)]


def subset_of(path: Path) -> str:
    """Crop files are named '<subset>_<6-digit idx>_<label>.jpg'."""
    stem = path.stem
    return stem.rsplit("_", 2)[0] if stem.count("_") >= 2 else "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="dataset dir (has train.txt/val.txt)")
    ap.add_argument("--split", default="val", choices=["train", "val"])
    ap.add_argument("--model-dir", default=None,
                    help="PIR/safetensors model dir; omit for the official model")
    ap.add_argument("--model-name", default="PP-OCRv6_small_rec")
    ap.add_argument("--engine", default="paddle", choices=["paddle", "transformers"])
    ap.add_argument("--device", default="gpu:0")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default="model")
    ap.add_argument("--label-check", action="store_true",
                    help="dataset-label validation mode: prints per-position agreement")
    ap.add_argument("--dump", default=None, help="write per-sample results to this JSON")
    args = ap.parse_args()

    from paddleocr import TextRecognition          # lazy: keeps this import cheap

    dataset = Path(args.dataset).resolve()
    rows = load_split(dataset, args.split, args.limit or None)
    if not rows:
        sys.exit(f"no samples in {dataset / (args.split + '.txt')}")
    print(f"dataset : {dataset}  split={args.split}  samples={len(rows)}")

    kwargs = dict(model_name=args.model_name, engine=args.engine, device=args.device,
                  enable_mkldnn=False)
    if args.model_dir:
        kwargs["model_dir"] = str(Path(args.model_dir).resolve())
    print(f"model   : {args.model_dir or 'official ' + args.model_name} "
          f"(engine={args.engine}, device={args.device})")

    t0 = time.perf_counter()
    rec = TextRecognition(**kwargs)
    print(f"loaded in {time.perf_counter() - t0:.1f}s")

    preds, labels, scores, subs, failures = [], [], [], [], []
    t0 = time.perf_counter()
    for path, label in rows:
        try:
            out = list(rec.predict(str(path)))
            res = out[0] if out else {}
            text = res.get("rec_text", "")
            score = float(res.get("rec_score", 0.0) or 0.0)
            if not np.isfinite(score):
                score = 0.0
        except Exception as exc:                    # noqa: BLE001
            text, score = "", 0.0
            failures.append(f"{path.name}: {type(exc).__name__}: {exc}")
        preds.append(normalize(text))
        labels.append(label)
        scores.append(score)
        subs.append(subset_of(path))
    elapsed = time.perf_counter() - t0
    per_img = elapsed / len(rows) * 1000

    exact = sum(1 for p, g in zip(preds, labels) if p == g)
    char_hit = sum(sum(1 for a, b in zip(p, g) if a == b) for p, g in zip(preds, labels))
    char_tot = sum(len(g) for g in labels)
    print()
    print(f"=== {args.tag} ===")
    print(f"exact match (整牌全对): {exact}/{len(rows)} = {exact / len(rows):.2%}")
    print(f"character accuracy   : {char_hit}/{char_tot} = {char_hit / max(1, char_tot):.2%}")
    print(f"mean score           : {float(np.mean(scores)):.4f}"
          f"   | inference {per_img:.1f} ms/img  (total {elapsed:.1f}s)")

    print("per-position accuracy (0=province, 1=letter, 2..6=alnum):")
    pos_lines = []
    for i, hit, tot, acc in position_report(preds, labels):
        pos_lines.append(f"  pos{i}: {acc:6.2%} ({hit}/{tot})")
        print(pos_lines[-1])

    by_sub = collections.defaultdict(lambda: [0, 0, 0.0])
    for p, g, s, sub in zip(preds, labels, scores, subs):
        rec_ = by_sub[sub]
        rec_[0] += 1
        rec_[1] += 1 if p == g else 0
        rec_[2] += s
    print(f"{'subset':<18} {'n':>6} {'exact':>8} {'mean score':>11}")
    for sub in sorted(by_sub):
        n, ok, sc = by_sub[sub]
        print(f"{sub:<18} {n:>6} {ok / n:>8.2%} {sc / n:>11.4f}")

    if failures:
        print(f"\nfailures: {len(failures)}")
        for f in failures[:5]:
            print("  ", f)
    if args.label_check:
        bad = [(p, g) for p, g in zip(preds, labels) if p != g][:10]
        print("\nfirst mismatches (pred vs dataset label):")
        for p, g in bad:
            print(f"   pred={p!r:<10} label={g!r}")
    if args.dump:
        dump_path = Path(args.dump)
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        dump_path.write_text(json.dumps(
            [{"file": str(p), "label": g, "pred": pr, "score": s, "subset": sub}
             for (p, g), pr, s, sub in zip(rows, preds, scores, subs)],
            ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"per-sample results -> {dump_path}")


if __name__ == "__main__":
    main()
