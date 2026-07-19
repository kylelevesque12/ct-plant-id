"""Pull the Connecticut plant checklist from iNaturalist, with counts.

This is the reconnaissance step for GOALS.md phase 1: before downloading any
images, learn the SHAPE of the problem. It hits the iNaturalist API for every
research-grade plant species observed in Connecticut and records each species'
observation count — a good proxy for how many training images we can get,
and therefore the head/mid/tail distribution.

Run: .venv/bin/python scripts/fetch_ct_checklist.py
Writes: data/ct_checklist.csv  (reproducible; gitignored)
"""
import csv
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from ctplantid import species as sp  # noqa: E402

API = "https://api.inaturalist.org/v1"
PLANTAE = 47126
HEADERS = {"User-Agent": "ct-plant-id/0.1 (personal learning project)"}
OUT = Path(__file__).resolve().parent.parent / "data" / "ct_checklist.csv"


def connecticut_place_id() -> int:
    r = requests.get(f"{API}/places/autocomplete",
                     params={"q": "Connecticut"}, headers=HEADERS, timeout=30)
    r.raise_for_status()
    for place in r.json()["results"]:
        # The state itself, not a town or park named Connecticut.
        if place["name"] == "Connecticut" and place.get("admin_level") == 1:
            return place["id"]
    # Fall back to the first result if admin_level isn't populated.
    return r.json()["results"][0]["id"]


def fetch_species_counts(place_id: int) -> list[dict]:
    out, page, per_page = [], 1, 500
    while True:
        r = requests.get(
            f"{API}/observations/species_counts",
            params={"place_id": place_id, "taxon_id": PLANTAE,
                    "quality_grade": "research", "per_page": per_page, "page": page},
            headers=HEADERS, timeout=60,
        )
        r.raise_for_status()
        data = r.json()
        results = data.get("results", [])
        if not results:
            break
        for row in results:
            taxon = row["taxon"]
            if taxon.get("rank") != "species":  # fold out subspecies/varieties for v1
                continue
            out.append({"taxon_id": taxon["id"],
                        "name": taxon["name"],
                        "obs_count": row["count"]})
        got = page * per_page
        print(f"  page {page}: {len(results)} taxa (running species total {len(out)})")
        if got >= data.get("total_results", 0):
            break
        page += 1
        time.sleep(1.0)  # be polite to the free API
    return out


def main():
    print("resolving Connecticut place_id…")
    place_id = connecticut_place_id()
    print(f"  place_id = {place_id}")

    print("fetching research-grade plant species counts…")
    rows = fetch_species_counts(place_id)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["taxon_id", "name", "obs_count"])
        w.writeheader()
        w.writerows(rows)

    counts = {r["name"]: r["obs_count"] for r in rows}
    summary = sp.tier_summary(counts)
    total = len(rows)
    print("\n=== Connecticut plant checklist ===")
    print(f"species (research-grade): {total}")
    print("tier split (by obs count, proxy for image availability):")
    for tier in ("head", "mid", "tail"):
        n = summary.get(tier, 0)
        pct = 100 * n / total if total else 0
        print(f"  {tier:4s}: {n:5d}  ({pct:4.1f}%)")
    print(f"\nwritten to {OUT.relative_to(OUT.parents[1])}")


if __name__ == "__main__":
    main()
