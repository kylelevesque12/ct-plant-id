"""Fold the cultivated ornamentals into the round-2 dataset.

Why this is needed: the round-2 build pulls from the iNaturalist Open Data
bucket filtered to the CT checklist, which is *research-grade wild flora*. The
150 garden species added in Workstream B are not on that checklist — they were
pulled separately with captive=true, because research-grade explicitly excludes
cultivated plants. So a round-2 model trained without this step would silently
lose every ornamental and reintroduce the hydrangea failure that started the
whole scope expansion.

The images already exist on the volume from the earlier pull, so this copies
rather than re-downloads.

Run on the build droplet:
  python scripts/merge_ornamentals_into_round2.py \
      --volume /mnt/ct-plant-data/data --round2 /root/round2 --write
"""
import argparse
import csv
import os
import shutil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--volume", default="/mnt/ct-plant-data/data",
                    help="volume data dir holding the v1 manifest + images")
    ap.add_argument("--round2", default="/root/round2")
    ap.add_argument("--scope", default=None,
                    help="ornamental_species.csv (default: <repo>/data/ornamental_species.csv)")
    ap.add_argument("--write", action="store_true", help="apply (default: dry run)")
    args = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    scope_path = args.scope or os.path.join(root, "data", "ornamental_species.csv")
    ornamental = {r["species"] for r in csv.DictReader(open(scope_path))}
    print(f"{len(ornamental)} ornamental species in scope")

    vol_manifest = os.path.join(args.volume, "manifest.csv")
    r2_manifest = os.path.join(args.round2, "manifest.csv")
    with open(r2_manifest, newline="") as f:
        reader = csv.DictReader(f)
        cols, r2_rows = reader.fieldnames, list(reader)
    have = {r["species"] for r in r2_rows}
    print(f"round-2: {len(r2_rows):,} images across {len(have):,} species")

    # Ornamental rows from the volume manifest that round-2 lacks.
    add, missing_files = [], 0
    for r in csv.DictReader(open(vol_manifest)):
        if r["species"] not in ornamental:
            continue
        src = r["path"]
        if not os.path.isabs(src):
            # v1 paths are repo-relative ("data/images/..."); resolve to the volume.
            src = os.path.join(args.volume, os.path.relpath(src, "data"))
        if not os.path.exists(src):
            missing_files += 1
            continue
        add.append((r, src))

    species_added = {r["species"] for r, _ in add}
    print(f"ornamental images found on volume: {len(add):,} "
          f"across {len(species_added):,} species")
    if missing_files:
        print(f"  ({missing_files:,} manifest rows had no file on disk, skipped)")
    already = species_added & have
    if already:
        print(f"  note: {len(already)} of these species are ALSO in round-2 "
              f"(wild + cultivated records) — keeping both")

    if not args.write:
        print("\ndry run — re-run with --write to copy and merge")
        return

    img_root = os.path.join(args.round2, "images")
    copied = 0
    new_rows = []
    for r, src in add:
        slug = os.path.basename(os.path.dirname(src))
        dest_dir = os.path.join(img_root, slug)
        os.makedirs(dest_dir, exist_ok=True)
        dest = os.path.join(dest_dir, os.path.basename(src))
        if not os.path.exists(dest):
            shutil.copy2(src, dest)
            copied += 1
        row = {c: r.get(c, "") for c in cols}
        row["path"] = dest              # round-2 manifest uses absolute paths
        row["split"] = r["split"]       # keep the observation-keyed split as-is
        new_rows.append(row)

    shutil.copy(r2_manifest, r2_manifest + ".preornamental.bak")
    with open(r2_manifest, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(r2_rows + new_rows)

    total_species = len(have | species_added)
    print(f"\ncopied {copied:,} image files")
    print(f"merged manifest: {len(r2_rows) + len(new_rows):,} images across "
          f"{total_species:,} species (.bak kept)")


if __name__ == "__main__":
    main()
