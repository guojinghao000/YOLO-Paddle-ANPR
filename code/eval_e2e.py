"""End-to-end plate accuracy on CCPD images: YOLO detection + recognition.

Runs the real serving pipeline (anpr_engine.ANPREngine) over CCPD2019 images and
compares the reported plate text with the ground truth embedded in the file
name, so the number reflects the whole chain rather than recognition alone.

Run in the ML env (needs ultralytics/torch):

    # A) fine-tuned model through the recognition service
    python eval_e2e.py --images 150 --rec-url http://127.0.0.1:8100 --tag finetuned-e2e

    # B) baseline: official model via the local transformers engine
    python eval_e2e.py --images 150 --tag official-e2e

Both write per-image JSON next to code/eval_reports/ for later diffing.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import time
from pathlib import Path

from make_plate_rec_dataset import CCPD_ROOT, decode_plate, parse_ccpd_name

BASE_DIR = Path(__file__).resolve().parent
SUBSETS = ("ccpd_base", "ccpd_blur", "ccpd_tilt", "ccpd_challenge", "ccpd_fn",
           "ccpd_weather", "ccpd_rotate", "ccpd_db")


def pick_images(n: int, seed: int, per_subset: bool) -> list[tuple[Path, str, str]]:
    """Return [(image_path, ground_truth_plate, subset), ...]."""
    rng = random.Random(seed)
    out: list[tuple[Path, str, str]] = []
    if per_subset:
        per = max(1, n // len(SUBSETS))
        for subset in SUBSETS:
            names = [p.name for p in (CCPD_ROOT / subset).glob("*.jpg")]
            rng.shuffle(names)
            taken = 0
            for name in names:
                info = parse_ccpd_name(name)
                if info is None:
                    continue
                plate = decode_plate(info["plate"])
                if plate is None:
                    continue
                out.append((CCPD_ROOT / subset / name, plate, subset))
                taken += 1
                if taken >= per:
                    break
    else:
        names: list[tuple[Path, str]] = []
        for subset in SUBSETS:
            for p in (CCPD_ROOT / subset).glob("*.jpg"):
                names.append((p, subset))
        rng.shuffle(names)
        for path, subset in names:
            info = parse_ccpd_name(path.name)
            if info is None:
                continue
            plate = decode_plate(info["plate"])
            if plate is None:
                continue
            out.append((path, plate, subset))
            if len(out) >= n:
                break
    return out[:n] if not per_subset else out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--per-subset", action="store_true",
                    help="sample evenly across subsets (better for hard-case view)")
    ap.add_argument("--rec-url", default=None,
                    help="use the recognition service (fine-tuned model); omit for the "
                         "local official model")
    ap.add_argument("--engine", default="transformers",
                    help="local engine when --rec-url is omitted")
    ap.add_argument("--device", default="gpu")
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--dump", default=None)
    args = ap.parse_args()

    from anpr_engine import ANPREngine

    samples = pick_images(args.images, args.seed, args.per_subset)
    print(f"samples: {len(samples)} CCPD images "
          f"({collections.Counter(s for _, _, s in samples)})")

    engine = ANPREngine(weights=None, device=args.device, conf=args.conf,
                        infer_engine=None if args.rec_url else args.engine,
                        rec_url=args.rec_url, preload=True, raise_errors=False)
    summary = engine.model_summary()
    print(f"engine : loaded={summary['loaded']} engine={summary['engine']}"
          f" rec_backend={summary['rec_backend']} rec_url={summary['rec_url']}"
          f" rec={summary['rec_model_dir']}")
    if not summary["loaded"]:
        raise SystemExit(f"engine failed to load: {summary['error']}")

    rows, t0 = [], time.perf_counter()
    for i, (path, truth, subset) in enumerate(samples, 1):
        try:
            res = engine.recognize_file(path, conf=args.conf)
            plates = res["plates"]
            best = max(plates, key=lambda p: p["det_conf"]) if plates else None
            pred = best["plate"] if best else ""
            raw = best["raw_text"] if best else ""
            conf = best["rec_conf"] if best else 0.0
            dt = res["timings_ms"]
        except Exception as exc:                      # noqa: BLE001
            pred, raw, conf, dt = "", "", 0.0, {}
            print(f"  ! {path.name}: {type(exc).__name__}: {exc}")
        rows.append({"file": str(path), "truth": truth, "pred": pred, "raw": raw,
                     "subset": subset, "rec_conf": conf, "timings": dt})
        if i % 25 == 0:
            print(f"  {i}/{len(samples)} ...")
    elapsed = time.perf_counter() - t0

    exact = sum(1 for r in rows if r["pred"] == r["truth"])
    ch_hit = sum(sum(1 for a, b in zip(r["pred"], r["truth"]) if a == b) for r in rows)
    ch_tot = sum(len(r["truth"]) for r in rows)
    print(f"\n=== {args.tag} ===")
    print(f"plates found   : {sum(1 for r in rows if r['pred'])}/{len(rows)}")
    print(f"exact match    : {exact}/{len(rows)} = {exact / len(rows):.2%}")
    print(f"character acc  : {ch_hit}/{ch_tot} = {ch_hit / max(1, ch_tot):.2%}")
    print(f"throughput     : {len(rows) / elapsed:.1f} img/s "
          f"({elapsed / len(rows) * 1000:.0f} ms/img incl. detect+rec)")
    by_sub = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        by_sub[r["subset"]][0] += 1
        by_sub[r["subset"]][1] += 1 if r["pred"] == r["truth"] else 0
    print(f"{'subset':<18} {'n':>5} {'exact':>8}")
    for sub in sorted(by_sub):
        n, ok = by_sub[sub]
        print(f"{sub:<18} {n:>5} {ok / n:>8.2%}")
    miss = [r for r in rows if r["pred"] != r["truth"]][:8]
    if miss:
        print("first misses (truth -> pred):")
        for r in miss:
            print(f"   {r['truth']:<9} -> {r['pred'] or '(none)':<9} [{r['subset']}]")

    dump = Path(args.dump) if args.dump else BASE_DIR / "eval_reports" / f"e2e_{args.tag}.json"
    dump.parent.mkdir(parents=True, exist_ok=True)
    dump.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"per-image results -> {dump}")


if __name__ == "__main__":
    main()
