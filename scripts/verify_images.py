"""Find and drop unreadable images from the manifest.

The Open Data download was interrupted by a full disk, which truncates files
mid-write; a few failed fetches also wrote empty files. Those raise
PIL.UnidentifiedImageError at training time.

Dropping them from the MANIFEST (rather than substituting at load time) keeps
val/test honest — a substituted image would quietly corrupt the metrics.

Fully decodes each image (not just headers) so truncation is caught, in
parallel since it's I/O bound. ~2-5 min for 280k images.

Run:  python3 scripts/verify_images.py --data ~/data
"""
import argparse
import csv
import os
from concurrent.futures import ThreadPoolExecutor

from PIL import Image


def resolve(data_dir, path):
    if os.path.isabs(path):
        return path
    if path.startswith("data" + os.sep):
        return os.path.join(data_dir, os.path.relpath(path, "data"))
    return os.path.join(data_dir, path)


def check(args_tuple):
    data_dir, row = args_tuple
    p = resolve(data_dir, row["path"])
    try:
        if os.path.getsize(p) == 0:
            return row, False
        with Image.open(p) as im:
            im.convert("RGB")        # force a full decode; catches truncation
        return row, True
    except Exception:
        return row, False


def main(args):
    manifest = os.path.join(args.data, "manifest.csv")
    rows = list(csv.DictReader(open(manifest)))
    print(f"checking {len(rows):,} images with {args.workers} workers…")

    good, bad = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for n, (row, ok) in enumerate(
                ex.map(check, ((args.data, r) for r in rows), chunksize=256), 1):
            (good if ok else bad).append(row)
            if n % 50000 == 0:
                print(f"  {n:,} checked, {len(bad):,} bad so far")

    backup = os.path.join(args.data, "manifest_unverified.csv")
    if not os.path.exists(backup):
        os.rename(manifest, backup)
    with open(manifest, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(good)

    from collections import Counter
    splits = Counter(r["split"] for r in good)
    print(f"\nbad/unreadable: {len(bad):,}  ({100*len(bad)/len(rows):.2f}%)")
    print(f"kept:           {len(good):,}")
    print(f"species:        {len({r['species'] for r in good}):,}")
    print(f"splits:         train {splits['train']:,}  val {splits['val']:,}  "
          f"test {splits['test']:,}")
    if bad:
        print(f"example bad file: {bad[0]['path']}")
    print(f"\noriginal manifest saved as {os.path.basename(backup)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.expanduser("~/data"))
    ap.add_argument("--workers", type=int, default=16)
    main(ap.parse_args())
