"""Observation-keyed image downloader for CT plant species (GOALS.md phase 1).

For each species, pull research-grade iNaturalist observations in Connecticut
and download their photos, RECORDING the observation id for every photo so
train/val/test splits can be made per-observation (no leakage). Writes a
manifest that the training dataloader reads.

Storage (flat by species; split lives in the manifest, not the path, so
changing split fractions never moves files):
    data/images/<species_slug>/<observation_id>_<n>.jpg
    data/manifest.csv   columns: observation_id, taxon_id, species, path,
                                 license_code, split

Polite by default and resumable (skips files already on disk). Defaults are
SMALL — override to scale up once you've picked the training scope.

Run (tiny smoke test):
    .venv/bin/python scripts/download_images.py --species-limit 3 --per-species 6
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from ctplantid import splits  # noqa: E402

API = "https://api.inaturalist.org/v1"
CT_PLACE_ID = 49
HEADERS = {"User-Agent": "ct-plant-id/0.1 (personal learning project)"}
CHECKLIST = ROOT / "data" / "ct_checklist.csv"
IMG_DIR = ROOT / "data" / "images"
MANIFEST = ROOT / "data" / "manifest.csv"


def slug(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def medium_url(photo_url: str) -> str:
    # iNat photo urls default to the 'square' thumbnail; 'medium' (~500px)
    # is a good training size.
    return photo_url.replace("/square.", "/medium.")


def top_species(limit: int, min_obs: int) -> list[dict]:
    rows = list(csv.DictReader(open(CHECKLIST)))
    rows = [r for r in rows if int(r["obs_count"]) >= min_obs]
    rows.sort(key=lambda r: -int(r["obs_count"]))
    return rows[:limit]


def observations_for(taxon_id: int, want: int) -> list[dict]:
    """Return up to `want` observations (each with its photos) for a taxon."""
    out, page = [], 1
    while len(out) < want:
        r = requests.get(
            f"{API}/observations",
            params={"taxon_id": taxon_id, "place_id": CT_PLACE_ID,
                    "quality_grade": "research", "photos": "true",
                    "order_by": "votes", "per_page": 50, "page": page},
            headers=HEADERS, timeout=60,
        )
        r.raise_for_status()
        results = r.json().get("results", [])
        if not results:
            break
        out.extend(results)
        page += 1
        time.sleep(1.0)
    return out[:want]


def download_species(sp_row: dict, per_species: int, writer) -> int:
    taxon_id, name = int(sp_row["taxon_id"]), sp_row["name"]
    dest = IMG_DIR / slug(name)
    dest.mkdir(parents=True, exist_ok=True)
    saved = 0
    # Pull a few more observations than photos wanted, since we take one
    # (the first) photo per observation for diversity across individuals.
    for obs in observations_for(taxon_id, per_species * 2):
        if saved >= per_species:
            break
        photos = obs.get("photos") or []
        if not photos:
            continue
        photo = photos[0]
        url = photo.get("url")
        if not url:
            continue
        obs_id = obs["id"]
        path = dest / f"{obs_id}_0.jpg"
        if not path.exists():
            try:
                img = requests.get(medium_url(url), headers=HEADERS, timeout=60)
                img.raise_for_status()
                path.write_bytes(img.content)
                time.sleep(0.5)
            except Exception as e:
                print(f"    skip photo ({type(e).__name__})")
                continue
        writer.writerow({
            "observation_id": obs_id,
            "taxon_id": taxon_id,
            "species": name,
            "path": str(path.relative_to(ROOT)),
            "license_code": photo.get("license_code") or "",
            "split": splits.split_for_observation(obs_id),
        })
        saved += 1
    return saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--species-limit", type=int, default=3)
    ap.add_argument("--per-species", type=int, default=6)
    ap.add_argument("--min-obs", type=int, default=100)
    args = ap.parse_args()

    if not CHECKLIST.exists():
        sys.exit("run fetch_ct_checklist.py first")

    IMG_DIR.mkdir(parents=True, exist_ok=True)
    species = top_species(args.species_limit, args.min_obs)
    print(f"downloading up to {args.per_species} photos each for "
          f"{len(species)} species (min {args.min_obs} obs)…")

    with open(MANIFEST, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "observation_id", "taxon_id", "species", "path",
            "license_code", "split"])
        writer.writeheader()
        total = 0
        for row in species:
            n = download_species(row, args.per_species, writer)
            total += n
            print(f"  {row['name']}: {n} photos")

    # Report the split shape of what we pulled.
    records = list(csv.DictReader(open(MANIFEST)))
    counts = splits.split_counts(records)
    print(f"\nmanifest: {total} photos across {len(species)} species")
    print(f"split (observation-keyed): {counts}")
    print(f"written to {MANIFEST.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
