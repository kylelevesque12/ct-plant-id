# Round 2 plan — peak model and app, on expiring credit

Written 2026-07-26. Constraint: **~$190 of DigitalOcean credit expires July 31**
(GitHub Student program ending, confirmed no extension), **no DO GPU access**,
and out-of-pocket spend should stay minimal. Lambda A10 is the only GPU option
and costs $1.29/hr.

The governing idea: **DO credit buys the expensive, slow, CPU/bandwidth-bound
work — building a better dataset. Lambda buys a few hours of GPU. Everything
else is free.**

---

## 0. Where the model actually fails (measured, not assumed)

| failure | evidence | root cause |
|---|---|---|
| **Trees weak** — 49.7% species / 67.7% genus | `reports/tree_benchmark.md`, 648 photos | fine-grained + missing image types |
| **Within-genus confusion** — Quercus 35% species vs 69% genus; Pinus 50/89; Carya 38/75 | same | congeners look alike; features not discriminative enough |
| **Near-empty classes confidently wrong** — *Metasequoia* → "Deutzia 92%" | same | classes that exist but were never learnable |
| **Field agreement 26% species / 52% genus** | `reports/field_comparison_picturethis.md`, 54 pairs | consistent with the above |
| **Evergreen-shrub confusion** — boxwood/cherry laurel vs inkberry at 91–97% | same | small-leaved evergreens under-separated |
| **Status 48% unknown** | `docs/data_card.md` | native tail never sourced |
| **OOD false positives ~3.7%** | field comparison | threshold at design budget; watch |

**What works and must not regress:** calibrated confidence is *monotonically*
predictive in the wild (74→57→43→33% genus agreement by label), poison ivy 3/3,
and the ornamental expansion fixed the hydrangea.

### The single biggest data finding

The downloader takes `photos[0]` and **discards every other photo on the
observation**. Measured availability:

| taxon | photos/observation | we take |
|---|---|---|
| *Quercus rubra* | **3.40** | 1 |
| *Acer rubrum* | 1.30 | 1 |
| *Toxicodendron radicans* | 1.23 | 1 |

**Trees have ~3× more photos per observation than herbaceous plants** — because
someone photographing a tree shoots the leaf, *then the bark, then the whole
tree*. Those are precisely the image types the tree benchmark found missing, and
they have been thrown away for free.

Critically, this is **leakage-safe by construction**: the split is
observation-keyed, so extra photos of an existing observation land in the same
split automatically.

---

## 1. Diagnostics to run FIRST (they gate two decisions)

Two numbers can't be obtained locally — the full manifest lives on the DO volume:

1. **Images-per-class distribution.** How many classes are near-empty
   (<20 images)? This decides whether class pruning is a major win or marginal.
   The data card says "min 1, median 300", so both extremes exist, but not how
   many are in *Metasequoia* territory.
2. **Accuracy stratified by tier (head/mid/tail) and by plant type**
   (tree / shrub / herbaceous). `GOALS.md` has always demanded stratified
   reporting and it has never been computed. The tree benchmark shows the global
   80.1% hides a much weaker sub-population; we don't know how many others.

**Gated decisions:** how aggressively to prune classes, and whether the 300-image
cap is worth raising broadly or only for the confusable genera.

Cost: a few CPU-minutes on the droplet we're spinning up anyway.

---

## 2. Data sources — evaluated

| source | value | verdict |
|---|---|---|
| **iNat Open Data, all photos per observation** | +~60% images overall, ~3× for trees, adds bark/form, **recovers the lost per-photo licences** | **do it — highest value by far** |
| **Complete the interrupted pull** | the 2026-07 build stopped at ~75% of target on a full disk | **do it — known gap** |
| **iNat casual/captive ornamentals, 150 → ~400** | field testing was almost entirely garden plants | **do it — matches real usage** |
| **USDA PLANTS** | comprehensive per-state native/introduced status | **do it — free, fixes 48% unknown** |
| Pl@ntNet-300K | ~306k expert-verified field photos, ~1k species; good domain match, partial CT overlap (naturalised Europeans) | **investigate** — measure overlap before committing |
| Bark-specific datasets (e.g. BarkNet) | directly targets the documented bark gap | **investigate** — small and focused; verify licence and species overlap |
| GBIF | mostly re-aggregates iNat, plus herbarium *specimens* (pressed, wrong domain) | skip |
| PlantCLEF training set | 1.4M images but global scope; large download for uncertain CT gain | skip for now |

**The honest ranking:** the top four are all within the existing pipeline and
need no new integration. Pl@ntNet and bark data are speculative until overlap is
measured — don't spend the week on them.

---

## 3. The plan

### Phase 1 — DO credit, this week (target: ~$120–190)

Spin up a **large droplet** (16 vCPU / high-bandwidth / ~400 GB) plus a
**temporary block volume**. Note DO block storage bills hourly — 500 GB for five
days is roughly **$8**, so the earlier "storage is a permanent trap" worry only
applies if it's left running. **Set a calendar reminder to destroy both.**

Work, in order:

1. **Diagnostics** (§1) — minutes.
2. **Rebuild the manifest from Open Data metadata** (`photos.csv.gz` +
   `observations.csv.gz`), selecting **every photo** of every in-scope
   observation, carrying `license` per photo.
3. **Bulk-download from S3** (`download_opendata.py --workers`) — this is the
   sanctioned bulk path and it parallelises, which is what makes a big droplet
   worth paying for.
4. **Expand ornamentals** to ~400 species.
5. **Targeted cap raise** for the confusable genera (Quercus, Carya, Pinus,
   Prunus, Betula, Acer) — *gated on §1*.
6. **Verify + integrity-check**, then transfer to Lambda when training.

**Keep the manifest as the durable artefact.** Images are regenerable from S3 at
any time; the manifest (with licences) is small, and it is what actually
encodes the dataset. That is what survives on the volume after the droplet dies.

### Phase 2 — Lambda GPU (~$15 out of pocket)

1. **Prune or genus-merge near-empty classes** — free, and it fixes the
   confidently-wrong failure mode that neither calibration nor OOD catches.
2. **Retrain** on the enlarged dataset. Single 384px stage, warm-started as
   before: ~$8–11 even with ~60% more data. Training cost is *not* the
   constraint.
3. **Workstream C, if budget allows: BioCLIP backbone + prototype head.** This
   is now better justified than when it was deferred — fine-grained
   within-genus discrimination is exactly its strength, and a
   distance-to-prototype head makes classification and OOD one mechanism
   instead of two.
4. **Re-fit temperature and rebuild the OOD bank** — both are model-specific and
   invalid after any retrain.

### Phase 3 — free app work (no compute)

1. **USDA PLANTS status backfill** — takes 48% unknown down substantially.
2. **Attribution / credits page** — a real licensing obligation (CC BY / CC BY-NC
   require attribution), and Phase 1 recovers the per-photo licence data that
   makes it possible.
3. **Stratified accuracy reporting** in the methodology page — stop quoting one
   global number.
4. Deploy and re-run the field comparison to measure movement.

---

## 4. Budget

| item | cost | paid by |
|---|---|---|
| Large droplet, ~5 days | ~$115 | **credit** |
| Temporary 500 GB volume, ~5 days | ~$8 | **credit** |
| Second droplet for parallel pulls (optional) | ~$60 | **credit** |
| Lambda A10 retrain | ~$8–11 | out of pocket |
| Lambda A10 BioCLIP run (optional) | ~$8–11 | out of pocket |
| **Out of pocket total** | **~$10–22** | |

**Do not manufacture spend.** If the useful work only consumes $120 of credit,
that is a good outcome — a rushed pull of data the benchmarks say won't help is
worse than letting credit lapse.

---

## 5. What this is expected to buy

- **Trees**: the main target. Bark and whole-tree photos are the missing image
  type, and they arrive free with the multi-photo pull.
- **Within-genus confusion**: attacked from two sides — more images per
  confusable species (Phase 1) and better fine-grained features (Phase 2).
- **Confidently-wrong sparse classes**: removed by pruning.
- **Status coverage**: 48% unknown → much lower, free.
- **Licensing**: per-photo licences recovered, attribution becomes possible.

What it will **not** do is close the gap to PictureThis. That was never the
right goal — they have orders of magnitude more data and a mature commercial
pipeline. The differentiators worth defending are honest scope, calibrated
confidence, explicit abstention, the genus fallback, and the native/invasive/
hazard call.
