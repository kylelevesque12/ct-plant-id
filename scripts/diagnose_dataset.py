"""Diagnostic 1: how is training data actually distributed across classes?

Answers the question that gates class pruning: how many classes are in
"Metasequoia territory" — present in the output space but with too few images to
have been learnable? Those are the classes that produce confidently wrong
answers (Metasequoia -> "Deutzia crenata 92%") that neither calibration nor the
OOD bank catches, because the input is a real plant inside the plant manifold.

Runs against the full manifest, so it must run where the manifest lives (the DO
volume), not on a laptop with 18 rows.

Run:  python scripts/diagnose_dataset.py --data /mnt/ct-plant-data/data
"""
import argparse
import csv
import json
import os
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Buckets chosen around the decision: below ~20 images a class is effectively
# unlearnable and unmeasurable; 20-100 is weak; 100+ is workable.
BUCKETS = [(1, 1), (2, 4), (5, 9), (10, 19), (20, 49), (50, 99),
           (100, 199), (200, 299), (300, 10 ** 9)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "data"))
    ap.add_argument("--out", default=os.path.join(ROOT, "reports", "dataset_diagnosis.json"))
    args = ap.parse_args()

    manifest = os.path.join(args.data, "manifest.csv")
    if not os.path.exists(manifest):
        raise SystemExit(f"no manifest at {manifest}")

    per_class, per_split = Counter(), Counter()
    with open(manifest, newline="") as f:
        for row in csv.DictReader(f):
            per_class[row["species"]] += 1
            per_split[row.get("split", "?")] += 1

    n_classes = len(per_class)
    n_images = sum(per_class.values())
    print(f"manifest: {n_images:,} images across {n_classes:,} classes")
    print(f"splits: {dict(per_split)}")

    counts = sorted(per_class.values())
    print(f"\nimages per class: min {counts[0]} | "
          f"p25 {counts[len(counts)//4]} | median {counts[len(counts)//2]} | "
          f"p75 {counts[3*len(counts)//4]} | max {counts[-1]}")

    print(f"\n{'images/class':>16} {'classes':>9} {'share':>8} {'cum images':>12}")
    rows = []
    for lo, hi in BUCKETS:
        members = [c for c, n in per_class.items() if lo <= n <= hi]
        imgs = sum(per_class[c] for c in members)
        label = f"{lo}" if lo == hi else (f"{lo}+" if hi > 10 ** 8 else f"{lo}-{hi}")
        print(f"{label:>16} {len(members):>9} {len(members)/n_classes:>7.1%} {imgs:>12,}")
        rows.append({"bucket": label, "classes": len(members),
                     "share": round(len(members) / n_classes, 4), "images": imgs})

    # The pruning decision, stated directly.
    for threshold in (10, 20, 50):
        weak = [c for c, n in per_class.items() if n < threshold]
        imgs = sum(per_class[c] for c in weak)
        print(f"\nclasses with < {threshold} images: {len(weak):,} "
              f"({len(weak)/n_classes:.1%} of classes, {imgs:,} images = "
              f"{imgs/n_images:.1%} of data)")
        print(f"  -> pruning them removes {len(weak):,} chances to be confidently "
              f"wrong, at the cost of {imgs/n_images:.1%} of training data")

    thinnest = sorted(per_class.items(), key=lambda kv: kv[1])[:15]
    print("\nthinnest classes (prime confidently-wrong candidates):")
    for sp, n in thinnest:
        print(f"   {n:>4}  {sp}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({"n_classes": n_classes, "n_images": n_images,
               "splits": dict(per_split), "buckets": rows,
               "per_class": dict(per_class)}, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
