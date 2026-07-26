"""Fill in the missing native/introduced status for every species in scope.

Why this exists
---------------
Field testing showed "Status unknown" on almost everything, including obvious
natives like red maple and poison ivy. The cause: `data/attributes.csv` only
ever contained the official CT invasive list (88), a hand-verified weed set
(14), and the Workstream B ornamentals (150) — **no native data at all**. So
2,258 of 2,510 classes (90%) had no status.

GOALS.md always planned to source status "from USDA PLANTS and the CT invasive
species list". The invasive half was done; this is the other half.

Source
------
iNaturalist publishes a curated `establishment_means` per taxon per place, and
Connecticut is place_id 49. Spot-checked: Acer rubrum -> native,
Toxicodendron radicans -> native, Alliaria petiolata -> introduced.

Safety rules
------------
- **Never downgrade an existing status.** A species already marked invasive,
  ornamental or introduced keeps that value; this only fills blanks.
- Only `native` and `introduced` are written. Anything iNat reports as
  endemic is recorded as native; unclear values are left unknown rather than
  guessed, matching the module's existing conservative policy.
- Writes a .bak before touching the file.

Run: .venv/bin/python scripts/backfill_native_status.py --write
"""
import argparse
import csv
import os
import shutil
import sys
import time

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

API = "https://api.inaturalist.org/v1"
HEADERS = {"User-Agent": "ct-plant-id/0.1 (status backfill)"}
CT_PLACE_ID = 49
ATTRS = os.path.join(ROOT, "data", "attributes.csv")
CKPT = os.path.join(ROOT, "runs", "b_stage2", "model.pt")

# iNat establishment_means -> our status vocabulary.
MEANS_MAP = {
    "native": "native",
    "endemic": "native",
    "introduced": "introduced",
    "naturalised": "introduced",
    "naturalized": "introduced",
}


def _get(url, params, tries=4):
    for attempt in range(tries):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=45)
            r.raise_for_status()
            return r.json()
        except Exception:
            if attempt == tries - 1:
                return None
            time.sleep(3 * (attempt + 1))


def status_for(species):
    """iNat establishment_means for this species in Connecticut, or None."""
    payload = _get(f"{API}/taxa", {"q": species, "rank": "species", "per_page": 5,
                                   "place_id": CT_PLACE_ID})
    if not payload:
        return None
    for res in payload.get("results", []):
        if res.get("name", "").lower() != species.lower():
            continue
        em = (res.get("establishment_means") or {}).get("establishment_means")
        return MEANS_MAP.get((em or "").lower())
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="apply (default: dry run)")
    ap.add_argument("--limit", type=int, default=0, help="only process N species (testing)")
    args = ap.parse_args()

    import torch
    classes = torch.load(CKPT, map_location="cpu", weights_only=False)["classes"]

    with open(ATTRS, newline="") as f:
        reader = csv.DictReader(f)
        cols, rows = reader.fieldnames, list(reader)
    by = {r["species"]: r for r in rows}

    # Only species with no status yet — never overwrite invasive/ornamental.
    todo = [c for c in classes
            if c not in by or (by[c].get("status") or "unknown") == "unknown"]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(classes)} classes | {len(by)} already have a row | "
          f"{len(todo)} need a status")

    found = {"native": 0, "introduced": 0}
    misses = 0
    for i, sp in enumerate(todo, 1):
        st = status_for(sp)
        if st:
            found[st] += 1
            if sp in by:
                by[sp]["status"] = st
                by[sp]["source"] = "iNaturalist establishment_means (CT)"
            else:
                row = {c: "" for c in cols}
                row.update(species=sp, status=st, is_weed="false",
                           source="iNaturalist establishment_means (CT)")
                by[sp] = row
        else:
            misses += 1
        if i % 25 == 0 or i == len(todo):
            print(f"  [{i}/{len(todo)}] native {found['native']} | "
                  f"introduced {found['introduced']} | no data {misses}")
        time.sleep(0.7)  # polite to the API

    print(f"\nresolved {found['native'] + found['introduced']}/{len(todo)} "
          f"(native {found['native']}, introduced {found['introduced']}); "
          f"{misses} left unknown")

    if not args.write:
        print("dry run — re-run with --write to apply")
        return

    shutil.copy(ATTRS, ATTRS + ".bak")
    out = sorted(by.values(), key=lambda r: r["species"])
    with open(ATTRS, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(out)
    print(f"wrote {len(out)} rows to {os.path.relpath(ATTRS, ROOT)} (.bak kept)")


if __name__ == "__main__":
    main()
