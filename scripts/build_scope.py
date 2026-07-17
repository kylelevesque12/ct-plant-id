"""Compute the v1 training scope from the CT checklist (GOALS.md phase 1).

v1 = species with >= MIN_OBS iNaturalist observations (the head+mid tiers
that hold ~97% of all observations). This is a waypoint toward comprehensive
coverage, not the final scope. Cheap and reproducible — reads the checklist,
writes the scope list. No image downloads.

Run: .venv/bin/python scripts/build_scope.py
Writes: data/v1_species.csv
"""
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from ctplantid import species as sp  # noqa: E402

CHECKLIST = ROOT / "data" / "ct_checklist.csv"
OUT = ROOT / "data" / "v1_species.csv"
MIN_OBS = 20  # head+mid cutoff; revisable — this is the scope decision


def main():
    rows = list(csv.DictReader(open(CHECKLIST)))
    scope = [r for r in rows if int(r["obs_count"]) >= MIN_OBS]
    scope.sort(key=lambda r: -int(r["obs_count"]))

    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["taxon_id", "name", "obs_count", "tier"])
        w.writeheader()
        for r in scope:
            w.writerow({**{k: r[k] for k in ("taxon_id", "name", "obs_count")},
                        "tier": sp.tier_for_count(int(r["obs_count"]))})

    total_obs = sum(int(r["obs_count"]) for r in rows)
    scope_obs = sum(int(r["obs_count"]) for r in scope)
    print(f"v1 scope: {len(scope)} species (>= {MIN_OBS} obs) of {len(rows)} total")
    print(f"  observation coverage: {100*scope_obs/total_obs:.1f}% of all obs")
    print(f"  tiers in scope: {sp.tier_summary({r['name']: int(r['obs_count']) for r in scope})}")
    print(f"written to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
