"""Rebuild the dataset manifest from images already on disk.

Use when a download was interrupted or its manifest was lost: everything the
manifest needs is already encoded in the file layout —

    data/images/<species_slug>/<observation_id>_0.<ext>

so species, observation id (and therefore the leakage-safe split) and path all
recover without re-downloading a single photo.

Caveat: per-photo LICENSE is not recoverable from disk (it lived in the Open
Data photos.csv), so it's written as "unknown". Fine for private training;
re-scan photos.csv if you ever need licenses for redistribution.

Run:  python scripts/build_manifest_from_disk.py
"""
import csv
import hashlib
import os
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKLIST = os.path.join(ROOT, "data", "ct_checklist.csv")
IMG_DIR = os.path.join(ROOT, "data", "images")
MANIFEST = os.path.join(ROOT, "data", "manifest.csv")


def split_for(obs_id, val=0.15, test=0.15):
    """Identical to the downloader's split, so an observation keeps the same
    assignment it would have had — no leakage between train and test."""
    h = int(hashlib.md5(str(obs_id).encode()).hexdigest(), 16) % 10000 / 10000
    return "test" if h < test else "val" if h < test + val else "train"


def slug(name):
    return name.strip().lower().replace(" ", "_")


def main():
    # slug -> (taxon_id, proper species name), so folder names map back exactly
    by_slug = {}
    with open(CHECKLIST) as f:
        for r in csv.DictReader(f):
            by_slug[slug(r["name"])] = (r["taxon_id"], r["name"])

    rows, unknown_dirs = [], set()
    for species_dir in sorted(os.listdir(IMG_DIR)):
        full = os.path.join(IMG_DIR, species_dir)
        if not os.path.isdir(full):
            continue
        meta = by_slug.get(species_dir)
        if meta is None:
            unknown_dirs.add(species_dir)
            # fall back to a de-slugged name so nothing is silently dropped
            meta = ("", species_dir.replace("_", " ").capitalize())
        taxon_id, species = meta
        for fname in os.listdir(full):
            path = os.path.join(full, fname)
            if not os.path.isfile(path):
                continue
            obs_id = fname.split("_")[0]          # <obs_id>_0.<ext>
            rows.append({
                "observation_uuid": obs_id,
                "taxon_id": taxon_id,
                "species": species,
                "path": os.path.relpath(path, ROOT),
                "license": "unknown",
                "split": split_for(obs_id),
            })

    with open(MANIFEST, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["observation_uuid", "taxon_id",
                           "species", "path", "license", "split"])
        w.writeheader()
        w.writerows(rows)

    splits = Counter(r["split"] for r in rows)
    per_species = Counter(r["species"] for r in rows)
    print(f"images:  {len(rows):,}")
    print(f"species: {len(per_species):,}")
    print(f"splits:  train {splits['train']:,}  val {splits['val']:,}  "
          f"test {splits['test']:,}")
    counts = sorted(per_species.values())
    print(f"images/species: min {counts[0]}  median {counts[len(counts)//2]}  "
          f"max {counts[-1]}")
    if unknown_dirs:
        print(f"note: {len(unknown_dirs)} folders not on the checklist "
              f"(kept, taxon_id blank), e.g. {sorted(unknown_dirs)[:3]}")
    print(f"wrote {os.path.relpath(MANIFEST, ROOT)}")


if __name__ == "__main__":
    main()
