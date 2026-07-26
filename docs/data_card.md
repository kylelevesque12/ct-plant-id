# Data card — CT plant classifier dataset (FINAL, as built)

The dataset actually built and used for training. Supersedes the earlier
projected version. Built 2026-07-17 from the iNaturalist Open Data set on AWS.

## What it is

| | |
|---|---|
| **Images** | **500,648** |
| **Species** | **2,360** |
| **Train / val / test** | 350,179 / 75,120 / 75,349 |
| **Size on disk** | ~67 GB |
| **Split method** | observation-keyed (hash of observation id) — all photos of one observation share a split, so near-identical photos never straddle train/test |
| **Images per species** | min 1, median 300 (the per-species cap), max 300 |

## Scope: what's included and why

Only **land plants** — the four phyla you can actually photograph with a phone:

| species | phylum | |
|---|---|---|
| 2,223 | Tracheophyta | vascular plants (trees, shrubs, herbs, ferns, grasses) |
| 101 | Bryophyta | mosses |
| 39 | Marchantiophyta | liverworts |
| 2 | Anthocerotophyta | hornworts |

**137 species were deliberately excluded** — microscopic green algae (desmids:
*Cosmarium*, *Staurastrum*, *Micrasterias*, *Closterium* …). iNaturalist's
"Plantae" kingdom includes them, but they require a microscope, so as classes
they could never be correctly predicted from a phone photo and only added the
risk of a blurry close-up being labeled a desmid. Removing them cost **3,032
images (0.6% of the data)** to drop 137 unusable classes — a good trade.
Reproduce with `scripts/filter_manifest.py`; the pre-filter manifest is kept
at `data/manifest_all.csv`.

## Coverage: what's missing

47 species from the 2,544-species CT checklist got no images. **Verified: none
are common** — the most-observed missing species has **7 observations** in all
of Connecticut; the rest have 2–4. Nearly all are microscopic algae (excluded
anyway by the filter above). Only three are vascular plants, all genuinely
rare: *Crataegus straminea* (4 obs), *Heptacodium miconioides* (2),
*Crocanthemum propinquum* (2).

So coverage is effectively complete for the app's purpose: **everything a phone
camera could encounter in Connecticut.**

## Provenance and honest limitations

- **Source:** iNaturalist Open Data on AWS (`s3://inaturalist-open-data`),
  research-grade observations, streamed and filtered to the CT checklist. The
  public API was used only for the checklist and small prototypes — not bulk.
- **License data is missing.** The download was interrupted by a full disk and
  the manifest was rebuilt from the files on disk
  (`scripts/build_manifest_from_disk.py`), which recovers species, observation
  id, and path — but not the per-photo license, recorded as `unknown`. Fine for
  training a private model; **re-scan `photos.csv.gz` if the image set is ever
  redistributed.**
- **The download stopped at ~75% of its 670k target** (disk limit at 77 GB; the
  full cap-300 pull would have needed ~200 GB). Because photos were fetched in
  arbitrary order, what landed is effectively a large random sample across all
  species — and the median species still hit the full 300-image cap.
- **The long tail is real:** some species have a single image. Those cannot be
  reliably learned *or* measured, which is why the model needs genus-level
  fallback and an honest "not sure" rather than a forced guess.

## Reproducing

```
scripts/fetch_ct_checklist.py        # CT species checklist from the iNat API
scripts/download_opendata.py         # bulk build from Open Data (--workers for parallel)
scripts/build_manifest_from_disk.py  # rebuild manifest if a download is interrupted
scripts/filter_manifest.py           # drop non-photographable phyla
```

## Status coverage (updated 2026-07-26)

Field testing showed "Status unknown" on almost everything. Cause:
`data/attributes.csv` only ever held the CT invasive list (88), a hand-verified
weed set (14) and the Workstream B ornamentals (150) — **no native data at
all**, so 2,258 of 2,510 classes (90%) had no status.

`scripts/backfill_native_status.py` filled part of the gap from iNaturalist's
curated `establishment_means` for Connecticut (place_id 49), never overwriting
an existing invasive/ornamental/introduced value:

| status | classes | share |
|--------|---------|-------|
| unknown | 1,205 | 48.0% |
| introduced | 757 | 30.2% |
| native | 310 | 12.4% |
| ornamental | 150 | 6.0% |
| invasive | 88 | 3.5% |
| **have a status** | **1,305** | **52.0%** (was 10.0%) |

**Known bias — do not read these shares as a description of CT's flora.**
iNaturalist's establishment data is well curated for prominent species and
sparse for the tail, and the tail here is disproportionately native. Species
still marked unknown include *Asclepias incarnata*, *Panax quinquefolius*,
*Eutrochium maculatum* and *Senna hebecarpa* — all unambiguous CT natives. The
introduced labels spot-check as accurate; the native count is simply
**incomplete**, which is why introduced appears to outnumber native 2.4:1.

Remaining work: backfill the native tail from **USDA PLANTS**, the other source
named in GOALS.md, which publishes native/introduced status per state
comprehensively rather than by community curation.
