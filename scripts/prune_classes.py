"""Drop classes that cannot be learned from the training manifest.

The evidence (reports/stratified_eval.json), by training-image count:

    < 20 images   top-1 39.7%   genus 67.9%   mean confidence 66.2%
    20-99         top-1 69.4%   genus 84.6%   mean confidence 79.5%
    100+          top-1 85.4%   genus 91.4%   mean confidence 88.2%

The sparse tier is both **inaccurate and overconfident** — 40% right while
reporting 66% confidence. That is the failure mode neither temperature scaling
nor the OOD bank can catch: the model is genuinely confident, and the input is
genuinely a plant. It produced *Metasequoia* -> "Deutzia crenata 92%".

A class with a handful of images cannot be learned, but its output neuron still
competes on every photo. Removing it removes a lottery ticket for a confident
wrong answer and costs almost no data (183 classes = 0.3% of images in the v1
set). Softmax is a competition, so the probability mass also redistributes to
classes that know something.

The 20-99 band is deliberately KEPT: 69.4% top-1 and 84.6% genus is real
capability, especially with the genus fallback. Pruning at <50 would throw it
away.

This does not delete image files — it rewrites the manifest, so a threshold can
be revisited without re-downloading.

Run:  python scripts/prune_classes.py --manifest /root/round2/manifest.csv --write
"""
import argparse
import csv
import os
import shutil
from collections import Counter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--min-images", type=int, default=20,
                    help="drop classes with fewer than this many images")
    ap.add_argument("--write", action="store_true", help="apply (default: dry run)")
    args = ap.parse_args()

    with open(args.manifest, newline="") as f:
        reader = csv.DictReader(f)
        cols, rows = reader.fieldnames, list(reader)

    per_species = Counter(r["species"] for r in rows)
    thin = {s for s, n in per_species.items() if n < args.min_images}
    kept = [r for r in rows if r["species"] not in thin]

    print(f"manifest: {len(rows):,} images across {len(per_species):,} classes")
    print(f"classes under {args.min_images} images: {len(thin):,} "
          f"({len(thin)/len(per_species):.1%} of classes)")
    print(f"images lost: {len(rows) - len(kept):,} "
          f"({(len(rows)-len(kept))/len(rows):.2%} of data)")
    print(f"after pruning: {len(kept):,} images across "
          f"{len(per_species) - len(thin):,} classes")

    # Show the extremes so the threshold is a judgement, not a black box.
    sample = sorted(((per_species[s], s) for s in thin))[:10]
    if sample:
        print("\nthinnest classes being dropped:")
        for n, s in sample:
            print(f"   {n:>4}  {s}")
        near = sorted(((per_species[s], s) for s in thin), reverse=True)[:5]
        print("closest to the threshold (the borderline calls):")
        for n, s in near:
            print(f"   {n:>4}  {s}")

    if not args.write:
        print("\ndry run — re-run with --write to apply")
        return

    shutil.copy(args.manifest, args.manifest + ".bak")
    with open(args.manifest, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(kept)
    dropped = os.path.join(os.path.dirname(args.manifest), "pruned_classes.txt")
    with open(dropped, "w") as f:
        for n, s in sorted(((per_species[s], s) for s in thin)):
            f.write(f"{n}\t{s}\n")
    print(f"\nwrote {len(kept):,} rows (.bak kept); dropped list -> {dropped}")
    print("image files are untouched — the threshold can be revisited without "
          "re-downloading.")


if __name__ == "__main__":
    main()
