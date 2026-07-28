"""Workstream B, step 1: pick the garden/ornamental species to ADD to scope.

Data-driven, not hand-guessed: ask iNaturalist which CULTIVATED plants people
actually log in Connecticut (captive=true = planted/cultivated, the exact set
research-grade excludes and the exact set a CT user photographs in a yard),
ranked by observation count. Drop anything already in the model's 2,360 classes.
The result is the candidate add-list; we cap it and pull images next.

Note: ornamentals aren't geographically scoped (a garden hydrangea is the same
species everywhere), but filtering to CT-cultivated keeps the list to what's
actually grown here rather than every tropical houseplant.

Run: .venv/bin/python scripts/build_ornamental_scope.py
Writes data/ornamental_species.csv
"""
import csv
import os
import sys
import time

import requests
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
API = "https://api.inaturalist.org/v1/observations/species_counts"
CT_PLACE_ID = 49
HEADERS = {"User-Agent": "ct-plant-id/0.1 (ornamental scope build)"}
CKPT = os.path.join(ROOT, "runs", "stage2", "model.pt")
OUT = os.path.join(ROOT, "data", "ornamental_species.csv")
TARGET = 150  # default cap on how many ornamentals to add (--target overrides)


def existing_classes(manifest=None):
    """Species already in scope, so the ornamental list only adds new ones.

    Prefers an explicit manifest (the round-2 build is the current truth about
    what's covered); otherwise falls back to whichever trained checkpoint exists.
    """
    if manifest:
        with open(manifest, newline="") as f:
            return {r["species"] for r in csv.DictReader(f)}
    for path in (os.path.join(ROOT, "runs", "b_stage2", "model.pt"), CKPT):
        if os.path.exists(path):
            ck = torch.load(path, map_location="cpu", weights_only=False)
            return set(ck["classes"])
    raise SystemExit("no manifest or checkpoint found — pass --exclude-manifest")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=TARGET,
                    help="how many ornamental species to keep")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--place-id", default=str(CT_PLACE_ID),
                    help="iNat place to rank by. CT (49) is a small sample — by "
                         "rank ~150 counts fall to single digits, which is noise. "
                         "A larger place (US = 1) ranks on far more data and "
                         "surfaces genuine landscape plants before houseplants.")
    ap.add_argument("--min-obs", type=int, default=0,
                    help="skip species below this observation count")
    ap.add_argument("--exclude-manifest", default=None,
                    help="manifest whose species are already covered "
                         "(e.g. /root/round2/manifest.csv)")
    args = ap.parse_args()
    target = args.target

    have = existing_classes(args.exclude_manifest)
    print(f"{len(have)} species already covered; finding CT-cultivated ones missing…")

    places = [int(p) for p in str(args.place_id).split(",") if p.strip()]
    print(f"ranking by cultivated observations pooled over places {places}")

    # Pool counts across places. CT alone is too small a sample to rank on (by
    # rank ~150 the counts are single digits); pooling climate-similar states
    # gives a stable ordering without importing Florida/Arizona plants the way
    # a US-wide query does.
    pooled = {}
    for place in places:
        page = 1
        while page <= 20:
            r = requests.get(API, headers=HEADERS, timeout=60, params={
                "place_id": place, "captive": "true", "iconic_taxa": "Plantae",
                "per_page": 200, "page": page,
            })
            r.raise_for_status()
            results = r.json()["results"]
            if not results:
                break
            for it in results:
                tx = it.get("taxon") or {}
                if tx.get("rank") != "species":
                    continue
                name = tx.get("name")
                if not name or name in have:
                    continue
                e = pooled.setdefault(tx["id"], {
                    "taxon_id": tx["id"], "species": name,
                    "common_name": tx.get("preferred_common_name", ""),
                    "cultivated_obs": 0})
                e["cultivated_obs"] += it["count"]
            page += 1
            time.sleep(1)

    rows = [e for e in pooled.values() if e["cultivated_obs"] >= args.min_obs]

    rows.sort(key=lambda x: -x["cultivated_obs"])
    rows = rows[:target]

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["taxon_id", "species", "common_name",
                                          "cultivated_obs"])
        w.writeheader()
        w.writerows(rows)

    print(f"\nwrote {len(rows)} ornamental candidates -> {os.path.relpath(args.out, ROOT)}")
    print(f"\ntop 25 cultivated species not yet in scope (places {args.place_id}):")
    for r in rows[:25]:
        print(f"  {r['cultivated_obs']:>5}  {r['species']:<34} {r['common_name']}")


if __name__ == "__main__":
    main()
