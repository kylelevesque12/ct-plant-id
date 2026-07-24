"""Workstream B, step 3: fold the ornamental rows into training's manifest.csv.

Concatenates data/ornamental_manifest.csv into data/manifest.csv after checking
the columns line up and nothing collides, then reports the new scope. Backs up
the wild manifest first. Idempotent-ish: it de-dupes by path, so re-running after
a bigger ornamental pull just adds the new rows.

Run on the GPU box after pull_ornamentals.py, before training:
  python scripts/merge_ornamentals.py --write
"""
import argparse
import csv
import os
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(ROOT, "data", "manifest.csv")
ORN = os.path.join(ROOT, "data", "ornamental_manifest.csv")
BACKUP = os.path.join(ROOT, "data", "manifest_wild_backup.csv")


def read(path):
    with open(path) as f:
        r = csv.DictReader(f)
        return r.fieldnames, list(r)


def summarize(rows, label):
    from collections import Counter
    sp = {r["species"] for r in rows}
    splits = Counter(r["split"] for r in rows)
    print(f"  {label}: {len(rows):,} imgs | {len(sp):,} species | "
          f"train {splits['train']:,} val {splits['val']:,} test {splits['test']:,}")
    return sp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true",
                    help="actually write (default is a dry-run preview)")
    args = ap.parse_args()

    wcols, wild = read(MANIFEST)
    ocols, orn = read(ORN)
    if wcols != ocols:
        raise SystemExit(f"STOP: column mismatch\n  manifest: {wcols}\n  ornamental: {ocols}")

    print("=== before ===")
    wild_sp = summarize(wild, "wild")
    orn_sp = summarize(orn, "ornamental")

    have_paths = {r["path"] for r in wild}
    fresh = [r for r in orn if r["path"] not in have_paths]
    new_species = orn_sp - wild_sp
    overlap = orn_sp & wild_sp
    print(f"\nornamental rows to add: {len(fresh):,} "
          f"({len(orn) - len(fresh):,} already present)")
    print(f"NEW species added to scope: {len(new_species)}"
          + (f"   (note: {len(overlap)} ornamental species already in wild scope)"
             if overlap else ""))

    merged = wild + fresh
    print("\n=== after (merged) ===")
    summarize(merged, "combined")

    if not args.write:
        print("\ndry-run only — re-run with --write to commit (backs up wild manifest).")
        return

    shutil.copy(MANIFEST, BACKUP)
    with open(MANIFEST, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=wcols)
        w.writeheader()
        w.writerows(merged)
    print(f"\nwrote {len(merged):,} rows -> {os.path.relpath(MANIFEST, ROOT)}  "
          f"(wild backed up to {os.path.relpath(BACKUP, ROOT)})")


if __name__ == "__main__":
    main()
