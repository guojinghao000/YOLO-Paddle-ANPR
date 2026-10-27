"""Build a PaddleOCR recognition dataset of CCPD2019 license plates.

CCPD embeds every annotation in the file name:

    00205459770115-90_85-352&516_448&547-444&547_368&549_364&517_440&515-0_0_22_10_26_29_24-128-7.jpg
    └─ area ─────┘└tilt┘└── bbox ────────┘└──────────── 4 vertices ─────────────┘└─ plate ─────┘└b┘└bl┘

The 5th field is the plate as **seven indices** (Chinese char, letter, then five
letters/digits) — not text.  Fields 6/7 are brightness / blurriness.

Output layout (PaddleOCR `rec` format):

    <out>/crop_img/*.jpg           plate crops
    <out>/train.txt                "crop_img/xxx.jpg<TAB>皖AC9M60"  (train split)
    <out>/val.txt                  same, val split
    <out>/dict.txt                 character dict (copy of the official PP-OCRv6 dict)

Crops use the SAME padding rule as serving (`anpr_engine.crop_plate`: bbox grown
by 5%), so training and inference see the same distribution.

Runs in the torch-free `anpr-ocr` env (needs only cv2 / numpy / PIL).
"""
from __future__ import annotations

import argparse
import collections
import random
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
CCPD_ROOT = PROJECT_ROOT / "CCPD2019"

# --------------------------------------------------------------- CCPD tables
# Standard CCPD index -> character tables (the 5th file-name field).
PROVINCES = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏",
             "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼",
             "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
ALPHABETS = ["A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N",
             "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "O"]
ADS = ["A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P",
       "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
       "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "O"]

# PaddleOCR's own PP-OCRv6 dict: 18708 chars -> CTC head 18710 (dict + space +
# blank), exactly the pretrained head size, and it covers every plate char.
OFFICIAL_DICT = PROJECT_ROOT / "PaddleOCR" / "ppocr" / "utils" / "dict" / "ppocrv6_dict.txt"

# Default train sampling per subset: CCPD's easy subsets dominate the corpus,
# so the hard ones (blur / tilt / weather / challenge) are over-weighted here.
DEFAULT_QUOTA = {
    "ccpd_base": 16_000,
    "ccpd_challenge": 5_000,
    "ccpd_tilt": 5_000,
    "ccpd_blur": 4_000,
    "ccpd_weather": 3_000,
    "ccpd_fn": 3_000,
    "ccpd_rotate": 2_000,
    "ccpd_db": 2_000,
}
# ccpd_np is deliberately absent: its file names carry no plate annotation.


def parse_ccpd_name(name: str):
    """Split a CCPD file name into its seven fields (no regex, cheap)."""
    stem = name[:-4] if name.lower().endswith(".jpg") else name
    parts = stem.split("-")
    if len(parts) != 7:
        return None
    area, tilt, bbox, vertices, plate, brightness, blurriness = parts
    if "_" not in bbox or plate.count("_") != 6:
        return None
    return dict(area=area, bbox=bbox, plate=plate, brightness=brightness,
                blurriness=blurriness, nfields=len(parts))


def decode_plate(plate_field: str) -> str | None:
    """'0_0_22_10_26_29_24' -> '皖AC9M60' (None when out of range)."""
    parts = plate_field.split("_")
    if len(parts) != 7:
        return None
    try:
        idx = [int(p) for p in parts]
    except ValueError:
        return None
    if not (0 <= idx[0] < len(PROVINCES)) or not (0 <= idx[1] < len(ALPHABETS)):
        return None
    if any(not (0 <= i < len(ADS)) for i in idx[2:]):
        return None
    return PROVINCES[idx[0]] + ALPHABETS[idx[1]] + "".join(ADS[i] for i in idx[2:])


def bbox_from_field(bbox_field: str):
    """'352&516_448&547' -> (352, 516, 448, 547), i.e. (x1, y1, x2, y2)."""
    try:
        left, right = bbox_field.split("_")
        x1, y1 = (int(v) for v in left.split("&"))
        x2, y2 = (int(v) for v in right.split("&"))
    except Exception:
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def crop_plate(img: np.ndarray, box, pad_ratio: float = 0.05) -> np.ndarray:
    """Mirror of ``anpr_engine.crop_plate`` (kept here so this script needs no
    torch import): grow the box by ``pad_ratio`` and clip to the image."""
    x1, y1, x2, y2 = (int(v) for v in box)
    h, w = img.shape[:2]
    pw = int((x2 - x1) * pad_ratio)
    ph = int((y2 - y1) * pad_ratio)
    return img[max(0, y1 - ph):min(h, y2 + ph), max(0, x1 - pw):min(w, x2 + pw)]


def imread_unicode(path: Path):
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


def imwrite_unicode(path: Path, img: np.ndarray) -> bool:
    ok, buf = cv2.imencode(path.suffix or ".jpg", img,
                           [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not ok:
        return False
    buf.tofile(str(path))
    return True


def _work(job):
    """Crop + save one sample.  Runs in a worker process."""
    src, dst, box, pad_ratio = job
    img = imread_unicode(src)
    if img is None:
        return "unreadable", None
    crop = crop_plate(img, box, pad_ratio)
    if crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 8:
        return "degenerate", None
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not imwrite_unicode(dst, crop):
        return "write-failed", None
    return "ok", crop.shape


def list_subset_files(subset_dir: Path) -> list[str]:
    return sorted(p.name for p in subset_dir.glob("*.jpg"))


def collect(args):
    """Pick samples per subset: val first, then train, keeping them disjoint."""
    total_quota = sum(DEFAULT_QUOTA.values())
    scale = args.train_samples / total_quota
    quota_train = {k: int(round(v * scale)) for k, v in DEFAULT_QUOTA.items()}
    quota_val = {k: max(1, int(round(v / total_quota * args.val_samples)))
                 for k, v in DEFAULT_QUOTA.items()}

    rng = random.Random(args.seed)
    seen_names: set[str] = set()
    train, val = [], []
    stats = {}
    for subset, q_train in quota_train.items():
        sdir = CCPD_ROOT / subset
        if not sdir.is_dir():
            print(f"  ! missing subset dir: {sdir}")
            continue
        names = list_subset_files(sdir)
        rng.shuffle(names)
        taken_train = taken_val = 0
        for name in names:
            info = parse_ccpd_name(name)
            if info is None:
                continue
            if args.dedupe:
                # the same photo appears in several subsets (fn/challenge/db);
                # keep a single copy so val never duplicates a train image
                key = f"{info['area']}-{info['bbox']}-{info['plate']}"
                if key in seen_names:
                    continue
            label = decode_plate(info["plate"])
            if label is None:
                continue
            box = bbox_from_field(info["bbox"])
            if box is None:
                continue
            if args.charset and any(c not in args.charset for c in label):
                continue
            seen_names.add(f"{info['area']}-{info['bbox']}-{info['plate']}")
            subset_out = "val" if taken_val < quota_val[subset] else "train"
            if subset_out == "val":
                if taken_val >= quota_val[subset]:
                    continue
                taken_val += 1
            else:
                if taken_train >= q_train:
                    continue
                taken_train += 1
            item = (sdir / name, label, box, subset)
            (val if subset_out == "val" else train).append(item)
            if taken_train >= q_train and taken_val >= quota_val[subset]:
                break
        stats[subset] = (taken_train, taken_val, len(names))
    return train, val, stats


def main():
    ap = argparse.ArgumentParser(
        description="CCPD2019 -> PaddleOCR plate recognition dataset")
    ap.add_argument("--out", default=str(PROJECT_ROOT / "rec_data_plate"))
    ap.add_argument("--train-samples", type=int, default=40_000)
    ap.add_argument("--val-samples", type=int, default=4_000)
    ap.add_argument("--pad-ratio", type=float, default=0.05,
                    help="must match anpr_engine.crop_plate (serving uses 0.05)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--clean", action="store_true", help="wipe the output dir first")
    ap.add_argument("--no-dedupe", dest="dedupe", action="store_false",
                    help="keep cross-subset duplicates (not recommended)")
    ap.add_argument("--dict", default=str(OFFICIAL_DICT),
                    help="character dict to copy in and to validate labels against")
    ap.add_argument("--preview", type=int, default=0,
                    help="also write a contact sheet of N samples with decoded labels")
    ap.add_argument("--skip-crop", action="store_true",
                    help="only write the label files (no image copying)")
    args = ap.parse_args()

    dict_path = Path(args.dict)
    if not dict_path.exists():
        sys.exit(f"dict not found: {dict_path}")
    chars = dict_path.read_text(encoding="utf-8").splitlines()
    args.charset = set(chars)
    print(f"dict: {dict_path}  ({len(chars)} chars)")

    out = Path(args.out)
    if args.clean and out.exists():
        print(f"cleaning {out} ...")
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    img_dir = out / "crop_img"

    print(f"sampling {args.train_samples} train / {args.val_samples} val ...")
    train, val, stats = collect(args)
    print(f"{'subset':<18} {'train':>7} {'val':>6} {'available':>10}")
    for subset, (n_tr, n_va, avail) in stats.items():
        print(f"{subset:<18} {n_tr:>7} {n_va:>6} {avail:>10}")
    print(f"{'TOTAL':<18} {len(train):>7} {len(val):>6}")
    if not train or not val:
        sys.exit("empty split - nothing written")

    # sanity: label shape + character coverage (visual check feeds --preview)
    bad_shape = [l for _, l, _, _ in train + val
                 if not (len(l) == 7 and l[0] in PROVINCES and l[1] in ALPHABETS)]
    print(f"labels not matching 'province+letter+5': {len(bad_shape)}"
          + (f"  e.g. {bad_shape[:5]}" if bad_shape else ""))
    prov = collections.Counter(l[0] for _, l, _, _ in train)
    print("top provinces:", " ".join(f"{c}:{n}" for c, n in prov.most_common(10)))
    lens = collections.Counter(len(l) for _, l, _, _ in train)
    print("label lengths:", dict(sorted(lens.items())))

    jobs = []
    for split_name, items in (("train", train), ("val", val)):
        for i, (src, label, box, subset) in enumerate(items):
            dst = img_dir / split_name / f"{subset}_{i:06d}_{label}.jpg"
            jobs.append((src, dst, box, args.pad_ratio, split_name, label, i))

    if args.skip_crop:
        print("--skip-crop: no images written")
        rows = {"train": [], "val": []}
        for _src, dst, _box, _pr, split_name, label, i in jobs:
            rows[split_name].append(f"{dst.relative_to(out).as_posix()}\t{label}")
    else:
        print(f"cropping {len(jobs)} plates -> {img_dir} (workers={args.workers}) ...")
        rows = {"train": [], "val": []}
        done = 0
        reasons = collections.Counter()
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_work, (src, dst, box, pr)):
                       (src, dst, box, pr, split_name, label, i)
                       for src, dst, box, pr, split_name, label, i in jobs}
            for fut in as_completed(futures):
                _src, dst, _box, _pr, split_name, label, i = futures[fut]
                try:
                    status, _shape = fut.result()
                except Exception as exc:            # noqa: BLE001
                    status = f"error:{type(exc).__name__}"
                reasons[status] += 1
                if status == "ok":
                    rel = dst.relative_to(out).as_posix()
                    rows[split_name].append(f"{rel}\t{label}")
                done += 1
                if done % 5000 == 0:
                    print(f"  {done}/{len(jobs)} ...")
        print("crop results:", dict(reasons))

    for split_name in ("train", "val"):
        p = out / f"{split_name}.txt"
        p.write_text("\n".join(rows[split_name]) + "\n", encoding="utf-8")
        print(f"wrote {p}  ({len(rows[split_name])} lines)")
    shutil.copyfile(dict_path, out / "dict.txt")
    print(f"copied dict -> {out / 'dict.txt'}")

    if args.preview:
        make_preview(out, rows["val"][:args.preview], args.preview)
    print("done.")


def make_preview(out: Path, rows: list[str], n: int):
    """Contact sheet with the decoded label drawn on each crop (visual check)."""
    from PIL import Image, ImageDraw, ImageFont
    font = None
    for path in ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf"):
        try:
            font = ImageFont.truetype(path, 28)
            break
        except Exception:                             # noqa: BLE001
            continue
    tiles = []
    for row in rows[:n]:
        rel, label = row.split("\t")
        img = Image.open(out / rel).convert("RGB")
        scale = max(1.0, 200.0 / max(1, img.width))
        if scale > 1.0:
            img = img.resize((int(img.width * scale), int(img.height * scale)))
        canvas = Image.new("RGB", (img.width, img.height + 40), (255, 255, 255))
        canvas.paste(img, (0, 40))
        ImageDraw.Draw(canvas).text((6, 4), label, fill=(0, 0, 0), font=font)
        tiles.append(canvas)
    if not tiles:
        return
    w = max(t.width for t in tiles)
    h = sum(t.height for t in tiles)
    sheet = Image.new("RGB", (w, h), (255, 255, 255))
    y = 0
    for t in tiles:
        sheet.paste(t, (0, y))
        y += t.height
    dst = out / "preview_labels.jpg"
    sheet.save(dst, quality=92)
    print(f"preview written -> {dst}")


if __name__ == "__main__":
    main()
