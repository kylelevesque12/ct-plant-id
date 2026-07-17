# Data reconnaissance — CT plant flora on iNaturalist

Source: iNaturalist API, research-grade plant (`Plantae`) observations in
the Connecticut place (place_id 49). Reproduce with
`scripts/fetch_ct_checklist.py`. Snapshot date: 2026-07-16.

## Headline numbers

- **2,542** research-grade plant species recorded in CT (confirms the
  ~2,000–2,500 flora estimate — comprehensive coverage is genuinely this big).
- **272,768** total observations across them.
- Observation count is used as a proxy for how many training images are
  obtainable per species.

## The long tail is severe

| tier | definition (obs) | species | share of species | share of all obs |
|------|------------------|---------|------------------|------------------|
| head | ≥ 100            | 464     | 18.3%            | —                |
| mid  | 20–99            | 512     | 20.1%            | —                |
| tail | < 20             | 1,566   | 61.6%            | —                |

- **head + mid (976 species, ≥20 obs) hold 97.2% of all observations.**
- 985 species have < 5 observations; **497 species have exactly one**.

## What this means for the plan

1. **976 well-sampled species cover ~97% of real-world encounters.** Rare
   species are rare precisely because people seldom photograph them, so a
   model trained on the head+mid is *functionally* strong in the field long
   before it's taxonomically complete. This is the near-term target that
   makes the app genuinely useful fast — and it's the concrete meaning of
   "head-first" in GOALS.md.
2. **Comprehensive (all 2,542) stays the destination**, but the tail cannot
   be reached with iNaturalist images alone: ~500 species with a single
   photo are untrainable as species classes. Closing the gap needs (a) extra
   data sources (GBIF, herbarium/museum specimens, Pl@ntNet), (b)
   genus/family fallback for the sparse tail, and (c) the active-learning
   flywheel over time. The coverage-fraction metric in GOALS.md phase 2 is
   what keeps that gap visible.
3. **The important weeds are already in the head.** Top species include
   several major invasives — garlic mustard (*Alliaria petiolata*), mugwort
   (*Artemisia vulgaris*), multiflora rose (*Rosa multiflora*) — so the
   weed-flagging use case is well served by the near-term model.

## Next steps (still phase 1)

- Cross-check this iNat list against a botanical CT flora / GBIF to catch
  species with occurrence records but no research-grade iNat photos (they
  belong on the checklist even if deferred for data).
- Decide the head+mid image-count floor for the first training set.
- Build the actual image downloader (observation-keyed, for leakage-safe
  splits) — the next concrete build step.
