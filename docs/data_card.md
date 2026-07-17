# Data card — CT plant classifier dataset

Generated from `data/ct_checklist.csv` by `scripts/make_data_card.py`.
Source: iNaturalist research-grade plant observations in Connecticut
(place_id 49). Snapshot: 2026-07-16. Observation count is used as a proxy
for available image count until the bulk build runs.

## Full CT flora (comprehensive target)

- Species (research-grade on iNaturalist): **2542**
- Total observations: **272,768**
- Tier distribution (by observation count):
  - head (>= 100): 464
  - mid (20-99): 512
  - tail (< 20): 1566

## v1 training scope (committed waypoint, not the final scope)

- Cutoff: species with >= 20 observations
- Species in scope: **976**
- Observation coverage: **97.2%** of all
  CT plant observations (rare species are rare in the field, so this covers
  the large majority of real encounters)
- Per-species obs in scope: min 20, median
  92, max 5136

## Deferred tail (backlog to close, NOT a scope cut)

- Species below the v1 cutoff: **1566**
- Of which < 5 observations: 985; exactly 1 observation: 497
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
