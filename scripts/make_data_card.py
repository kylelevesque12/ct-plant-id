"""Generate the dataset data card from the CT checklist recon.

Produces docs/data_card.md. Reproducible from data/ct_checklist.csv — the
numbers here describe the AVAILABLE data (observation counts as an image
proxy) and the committed v1 scope. Per-photo license and actual downloaded
image counts are filled in AFTER the bulk build (marked below), because
those require the images physically present.

Run: .venv/bin/python scripts/make_data_card.py
"""
import csv
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from ctplantid import species as sp  # noqa: E402

CHECKLIST = ROOT / "data" / "ct_checklist.csv"
OUT = ROOT / "docs" / "data_card.md"
MIN_OBS = 20


def main():
    rows = [(r["name"], int(r["obs_count"])) for r in csv.DictReader(open(CHECKLIST))]
    counts = {n: c for n, c in rows}
    tiers = sp.tier_summary(counts)
    total = len(rows)
    obs = [c for _, c in rows]

    scope = [(n, c) for n, c in rows if c >= MIN_OBS]
    deferred = total - len(scope)
    single = sum(1 for _, c in rows if c == 1)
    under5 = sum(1 for _, c in rows if c < 5)

    md = f"""# Data card — CT plant classifier dataset

Generated from `data/ct_checklist.csv` by `scripts/make_data_card.py`.
Source: iNaturalist research-grade plant observations in Connecticut
(place_id 49). Snapshot: 2026-07-16. Observation count is used as a proxy
for available image count until the bulk build runs.

## Full CT flora (comprehensive target)

- Species (research-grade on iNaturalist): **{total}**
- Total observations: **{sum(obs):,}**
- Tier distribution (by observation count):
  - head (>= {sp.HEAD_MIN}): {tiers.get('head', 0)}
  - mid ({sp.TAIL_MAX}-{sp.HEAD_MIN - 1}): {tiers.get('mid', 0)}
  - tail (< {sp.TAIL_MAX}): {tiers.get('tail', 0)}

## v1 training scope (committed waypoint, not the final scope)

- Cutoff: species with >= {MIN_OBS} observations
- Species in scope: **{len(scope)}**
- Observation coverage: **{100*sum(c for _,c in scope)/sum(obs):.1f}%** of all
  CT plant observations (rare species are rare in the field, so this covers
  the large majority of real encounters)
- Per-species obs in scope: min {min(c for _,c in scope)}, median
  {int(statistics.median([c for _,c in scope]))}, max {max(c for _,c in scope)}

## Deferred tail (backlog to close, NOT a scope cut)

- Species below the v1 cutoff: **{deferred}**
- Of which < 5 observations: {under5}; exactly 1 observation: {single}
- These cannot be learned as species classes from iNaturalist alone. Path to
  comprehensive: extra data sources (GBIF, herbarium/museum specimens,
  Pl@ntNet), genus/family fallback, and the active-learning flywheel.

## Provenance & licensing

- Bulk images: to be pulled from the **iNaturalist Open Data set on AWS**
  (`s3://inaturalist-open-data`: `photos.csv`, `observations.csv`,
  `taxa.csv`) or the GBIF media export — NOT the public API (which asks
  callers not to bulk-download). The `scripts/download_images.py` API path
  is for small/incremental pulls only.
- Per-photo license: Open Data carries a `license` per photo; the built
  dataset must record it per image. (The API smoke-test manifest left
  `license_code` blank — a known gap the Open Data build closes.)

## Filled in after the bulk build (TODO)

- Actual images downloaded per species (vs. the obs proxy above)
- Observation-keyed split sizes (train/val/test) on the real set
- Per-photo license breakdown
- GBIF/state-flora cross-check: species with occurrence records but no
  research-grade iNat photos
"""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(md)
    print(f"wrote {OUT.relative_to(ROOT)} ({len(scope)} species in v1 scope, "
          f"{deferred} deferred)")


if __name__ == "__main__":
    main()
