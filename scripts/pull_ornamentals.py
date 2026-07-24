"""Workstream B, step 2: pull cultivated (captive) photos for the ornamental
scope, in the SAME manifest/split format as the wild set — ready to concatenate
into training's manifest.csv.

Cultivated ornamentals are exactly what research-grade EXCLUDES (and what a CT
user photographs in a yard), so we pull captive=true observations. NOT
place-restricted: a garden hydrangea is the same species everywhere, and CT-only
cultivated counts are too thin to train on — take each species globally, capped.

Observation-keyed split (the leakage guard) via ctplantid.splits, identical to
the wild pipeline. ~150 species x up to --per-species photos. Polite rate
limiting; resumable (skips files already on disk). Run on the GPU box, or
locally then rsync the new data/images/* + manifest up.

Run: python scripts/pull_ornamentals.py --per-species 250
Writes data/images/<species>/ + data/ornamental_manifest.csv
"""
import argparse
import csv
import os
import sys
import time

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from ctplantid import splits  # noqa: E402

API = "https://api.inaturalist.org/v1/observations"
HEADERS = {"User-Agent": "ct-plant-id/0.1 (ornamental scope pull)"}
SCOPE = os.path.join(ROOT, "data", "ornamental_species.csv")
IMG_DIR = os.path.join(ROOT, "data", "images")


def slug(name):
    return name.strip().lower().replace(" ", "_")


def medium_url(u):
    return u.replace("/square.", "/medium.")  # ~500px training size


def observations_for(taxon_id, want):
    """Up to `want` cultivated observations (each with photos) for a taxon."""
    out, page = [], 1
    while len(out) < want and page <= 40:
        r = requests.get(API, headers=HEADERS, timeout=60, params={
            "taxon_id": taxon_id, "captive": "true", "photos": "true",
            "order_by": "votes", "per_page": 50, "page": page})
        r.raise_for_status()
        res = r.json().get("results", [])
        if not res:
            break
        out.extend(res)
        page += 1
        time.sleep(1.0)
    return out[:want]


def pull_species(row, per_species, writer):
    taxon_id, name = int(row["taxon_id"]), row["species"]
    dest = os.path.join(IMG_DIR, slug(name))
    os.makedirs(dest, exist_ok=True)
    saved = 0
    # Over-fetch observations; take one (first) photo each for individual diversity.
    for obs in observations_for(taxon_id, per_species * 2):
        if saved >= per_species:
            break
        photos = obs.get("photos") or []
        if not photos or not photos[0].get("url"):
            continue
        photo = photos[0]
        uuid = obs.get("uuid") or str(obs["id"])
        path = os.path.join(dest, f"{obs['id']}_0.jpg")
        if not os.path.exists(path):
            try:
                img = requests.get(medium_url(photo["url"]), headers=HEADERS, timeout=60)
                img.raise_for_status()
                with open(path, "wb") as fh:
                    fh.write(img.content)
                time.sleep(0.4)
            except Exception as e:
                print(f"    skip ({type(e).__name__})")
                continue
        writer.writerow({
            "observation_uuid": uuid,
            "taxon_id": taxon_id,
            "species": name,
            "path": os.path.relpath(path, ROOT),        # data/images/<slug>/<id>_0.jpg
            "license": photo.get("license_code") or "",
            "split": splits.split_for_observation(uuid),
        })
        saved += 1
    return saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-species", type=int, default=250)
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "ornamental_manifest.csv"))
    args = ap.parse_args()

    species = list(csv.DictReader(open(SCOPE)))
    print(f"pulling <= {args.per_species} cultivated photos each for "
          f"{len(species)} ornamentals (captive=true, global)…")
    total = 0
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["observation_uuid", "taxon_id",
                                          "species", "path", "license", "split"])
        w.writeheader()
        for i, row in enumerate(species):
            n = pull_species(row, args.per_species, w)
            f.flush()
            total += n
            print(f"  [{i+1}/{len(species)}] {row['species']}: {n}  (total {total})")
    print(f"\n{total} photos -> {os.path.relpath(args.out, ROOT)}"
          f"  (concat data rows into manifest.csv to train)")


if __name__ == "__main__":
    main()
