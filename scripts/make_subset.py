"""Build a smaller, transfer-friendly subset of the dataset.

The full set is ~67 GB; moving that to a rented GPU is the slow step. Capping
images per species cuts it a lot for little accuracy cost (the balanced
sampler already handles class imbalance, and the median species has 300).

Uses HARDLINKS, so the subset takes seconds to build and costs no extra disk —
it points at the same file data, it doesn't copy it. rsync still transfers only
the selected files' contents.

Split ratios are preserved (each species is capped proportionally across
train/val/test), and selection is deterministic (sorted by observation id), so
the subset is reproducible.

Run on the droplet:
    python3 scripts/make_subset.py --cap 150
    # then transfer /mnt/ct_plant_data/subset (not .../data)
"""
import argparse
import csv
import os
from collections import Counter, defaultdict

SHARE = {"train": 0.70, "val": 0.15, "test": 0.15}


def slug(name):
    return name.strip().lower().replace(" ", "_")


def main(args):
    src_manifest = os.path.join(args.src, "manifest.csv")
    rows = list(csv.DictReader(open(src_manifest)))
    print(f"source: {len(rows):,} images, "
          f"{len({r['species'] for r in rows}):,} species")

    # group by (species, split), deterministic order
    groups = defaultdict(list)
    for r in rows:
        groups[(r["species"], r["split"])].append(r)
    for k in groups:
        groups[k].sort(key=lambda r: r["observation_uuid"])

    keep = []
    for (species, split), items in groups.items():
        cap = max(1, int(args.cap * SHARE.get(split, 0.15)))
        keep.extend(items[:cap])

    os.makedirs(args.out, exist_ok=True)
    src_parent = os.path.dirname(os.path.abspath(args.src))
    linked, missing = 0, 0
    out_rows = []
    for r in keep:
        full = os.path.join(src_parent, r["path"])
        if not os.path.exists(full):
            missing += 1
            continue
        rel = os.path.join("images", slug(r["species"]), os.path.basename(full))
        dest = os.path.join(args.out, rel)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        if not os.path.exists(dest):
            os.link(full, dest)          # hardlink: instant, no extra disk
        linked += 1
        out_rows.append({**r, "path": rel})

    with open(os.path.join(args.out, "manifest.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)

    splits = Counter(r["split"] for r in out_rows)
    per_sp = Counter(r["species"] for r in out_rows)
    sizes = sorted(per_sp.values())
    print(f"\nsubset (cap {args.cap}/species):")
    print(f"  images:  {linked:,}   ({100*linked/len(rows):.0f}% of source)")
    print(f"  species: {len(per_sp):,}")
    print(f"  splits:  train {splits['train']:,}  val {splits['val']:,}  "
          f"test {splits['test']:,}")
    print(f"  per species: min {sizes[0]}  median {sizes[len(sizes)//2]}  "
          f"max {sizes[-1]}")
    if missing:
        print(f"  note: {missing:,} manifest rows had no file on disk (skipped)")
    print(f"\nwrote {args.out}  — transfer THIS directory, not the full data/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/mnt/ct_plant_data/data")
    ap.add_argument("--out", default="/mnt/ct_plant_data/subset")
    ap.add_argument("--cap", type=int, default=150,
                    help="max images per species (across all splits)")
    main(ap.parse_args())
