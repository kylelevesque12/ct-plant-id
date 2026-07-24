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
TARGET = 150  # cap on how many ornamentals to add


def existing_classes():
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    return set(ck["classes"])


def main():
    have = existing_classes()
    print(f"model already covers {len(have)} species; finding CT-cultivated ones it lacks…")

    rows, page = [], 1
    while len(rows) < TARGET * 3 and page <= 10:
        r = requests.get(API, headers=HEADERS, timeout=60, params={
            "place_id": CT_PLACE_ID, "captive": "true", "iconic_taxa": "Plantae",
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
            rows.append({
                "taxon_id": tx["id"],
                "species": name,
                "common_name": tx.get("preferred_common_name", ""),
                "ct_cultivated_obs": it["count"],
            })
        page += 1
        time.sleep(1)

    rows.sort(key=lambda x: -x["ct_cultivated_obs"])
    rows = rows[:TARGET]

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["taxon_id", "species", "common_name",
                                          "ct_cultivated_obs"])
        w.writeheader()
        w.writerows(rows)

    print(f"\nwrote {len(rows)} ornamental candidates -> {os.path.relpath(OUT, ROOT)}")
    print(f"\ntop 25 CT-cultivated species not yet in scope:")
    for r in rows[:25]:
        print(f"  {r['ct_cultivated_obs']:>5}  {r['species']:<34} {r['common_name']}")


if __name__ == "__main__":
    main()
