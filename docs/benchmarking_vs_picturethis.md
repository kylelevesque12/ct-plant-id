# Benchmarking Fieldnote against PictureThis — what's known, and how to measure it

Written 2026-07-25. Answers four questions: what PictureThis's published accuracy
actually is, why our number isn't comparable to it as-is, how to run a defensible
comparison, and what the licensing position is.

---

## 1. What's actually published

The best independent study is from the **Rutgers Urban Forestry Program**, which
tested six apps on **55 common street and native forest tree species in New
Jersey**, using 4+ photographs each of *bark* and *leaves*, taken by experienced
arborists:

| app | genus accuracy | species accuracy |
|-----|---------------|------------------|
| **PictureThis** | **97.3%** | **83.9%** |
| iNaturalist | 92.3% | 69.6% |
| (range across all six) | 71.8–97.3% | 40.9–83.9% |

Other findings worth noting:

- **Accuracy collapses on bark**: PictureThis drops to ~65% genus / ~52% species
  on bark-only images, versus its leaf numbers above. Image type dominates.
- Other studies put PictureThis at ~94–96% on foraging-focused species sets, and
  it performs best of the tested apps on toxic-plant identification.
- Independent toxic/edible-plant studies in the American Midwest found plant ID
  apps as a category **not reliable** for safety-critical decisions.
- PictureThis's own marketing claims ">98% accuracy" across "400,000+ species."
  That is a vendor claim, not an independent measurement, and it should not be
  compared against a held-out test number.

## 2. Why our 80.1% is not comparable to their 83.9%

Fieldnote's measured **80.1% top-1** comes from held-out iNaturalist test photos
across **2,510 species**, including a long tail of rarely-photographed ones.
PictureThis's 83.9% comes from **55 common tree species**, photographed
deliberately by arborists, at 4+ images each.

Those are not the same task:

| | Fieldnote | Rutgers/PictureThis |
|---|---|---|
| classes | 2,510 | 55 |
| species difficulty | full CT flora incl. long tail | common, well-known trees |
| photo quality | crowd-sourced, variable | expert-taken, deliberate |
| metric | top-1 on held-out split | correct-or-not per app answer |

A 55-way problem over common trees is dramatically easier than a 2,510-way
problem over an entire state flora. **Quoting "80.1% vs 83.9%" as though we're
2 points behind would be dishonest** — the tasks differ by more than the gap.

## 3. How to actually run the comparison

### Option A — replicate the Rutgers protocol (cheap, defensible, do this first)

Evaluate *our* model on the same species set the study used — common Northeast
street/forest trees, which overlap heavily with the CT checklist — and report
**both genus and species accuracy**, the same two metrics they report.

Why this is worth doing:
- It requires no access to PictureThis at all; we compare to a published number.
- It forces us to report **genus-level accuracy**, which we currently don't
  compute at all despite `GOALS.md` naming it a headline metric. Genus accuracy
  is also the fairer comparison, since it's robust to the class-count gap.
- Caveat to state plainly: different photographs (ours from iNaturalist, theirs
  arborist-taken), so it is a *protocol* replication, not a controlled trial.

### Option B — paired head-to-head on a fixed photo set (most rigorous)

Take N photos (held-out test images, or fresh photos taken specifically for
this), run each through **both** Fieldnote and PictureThis, record both answers
against the known truth.

- PictureThis has **no free bulk API**, so this is manual phone testing.
- The published studies did ~220 tests (55 species × 4 images); **100–200 paired
  tests is a realistic afternoon** and is the same order of magnitude.
- This is the only way to get a genuine same-photos, same-conditions number.
- Stratify deliberately: easy/common vs hard/tail species, and flower vs leaf vs
  bark — the studies show image type matters more than app choice.

### Establishing ground truth (the hard part of Option B)

Self-labelled backyard photos are not a benchmark — the comparison would inherit
our own identification errors. Four sources of defensible truth, best first:

1. **Held-out iNaturalist test photos, uploaded to PictureThis from the photo
   library.** Truth is already established (research-grade = 2+ independent
   experts agreed), the *same image file* goes through both systems so nothing
   varies but the model, and it scales to hundreds of images with no fieldwork.
   **State the caveat:** these photos may be in PictureThis's training set,
   which biases in *their* favour — a conservative bias for us.
2. **A labelled arboretum.** Species are on the placard and the photos are new
   to both models — the cleanest unseen-data test. Nearby: Yale Marsh Botanical
   Garden, Connecticut College Arboretum, Bartlett Arboretum (Stamford),
   Elizabeth Park (Hartford). Caveats: placards go stale, and specimens are
   often named cultivars where the model predicts a species.
3. **Community verification** — post the photos to iNaturalist and let experts
   identify them. Slow (days) but rigorous, and it contributes data back.
4. **Nursery tags** — convenient, but usually cultivar-level
   (*Acer palmatum* 'Bloodgood') against a species-level prediction.

Recommended: **(1) for sample size, plus ~30–50 photos from (2)** as an
unseen-data check. Agreement between the two is the strong result.

### Does our scope even cover the Rutgers species?

Yes — the model carries **148 tree species** across the relevant genera
(Quercus 15, Prunus 17, Acer 13, Cornus 13, Pinus 9, …), including the
cultivated street trees the study features (Ginkgo, Zelkova, *Pyrus calleryana*,
*Platanus occidentalis*, *Gleditsia triacanthos*) — several of which only
entered scope with the Workstream B ornamental expansion.

New Jersey and Connecticut share essentially the same street and forest tree
flora, so the species set transfers. Before running the eval, check coverage
per species and **report any of the 55 we don't carry** rather than quietly
scoring only the ones we do.

### Option C — automated proxy via a competitor API

Plant.id (Kindwise) sells an API. It isn't PictureThis, but it gives an
*automatable* commercial baseline over hundreds of images instead of a manual
sample. Useful if we want a large-N commercial comparison; costs money.

### Option D — academic benchmarks

PlantCLEF/LifeCLEF publish leaderboards. Note the 2025 task is **multi-label
species identification in vegetation quadrat images** — a different problem from
our single-plant classification, so it isn't directly comparable. PlantCLEF 2023
(global-scale single-plant) is the closer analogue if we want an academic
reference point.

### Recommended

**A first** (it costs one eval script and fixes a real metrics gap), then **B**
on 100–200 photos for a genuine head-to-head. Report both, with the task-
difficulty caveat stated up front.

## 4. What model does PictureThis use?

**Not publicly disclosed.** PictureThis is a commercial product from Glority;
no architecture, training-set description, or evaluation methodology is
published. Their "400,000+ species / >98% accuracy" figures are marketing
claims with no published methodology behind them.

What can be said honestly for comparison purposes:

- They have **vastly more data and species coverage**, including cultivated
  ornamentals worldwide — the exact gap that made our hydrangea fail before
  Workstream B.
- We have something they don't publish: **a stated scope, a reproducible
  held-out evaluation, calibrated confidence, and an explicit abstention
  mechanism.** That's the honest differentiator to lead with — not raw accuracy.

## 5. Licensing and copyright

### The position

Training images came from the **iNaturalist AWS Open Data** bucket, which
contains only CC0, CC BY, and CC BY-NC licensed photos (all-rights-reserved
content is excluded). The default license on iNaturalist is **CC BY-NC**.

Three things follow:

1. **Non-commercial is the binding constraint.** iNaturalist's terms explicitly
   prohibit using their data to train AI/ML models *for commercial purposes*,
   and CC BY-NC prohibits commercial use of the underlying photos. Fieldnote is
   currently free, unmonetized, and portfolio/educational — which is the
   non-commercial case. **Adding ads, subscriptions, or any paid tier would
   change that analysis and require re-sourcing the training data** (CC0/CC BY
   only, or a different dataset).
2. **Attribution is required** for CC BY and CC BY-NC content. We are not
   redistributing photos — only trained weights — and whether model weights are
   a derivative work of the training images is legally unsettled. But
   attribution is cheap, expected by the community, and the right thing to do
   regardless of where that question lands.
3. **We currently cannot produce per-photo attribution.** `data/manifest.csv`
   records `license` as `unknown` for every row, because the manifest was
   rebuilt from disk after the disk-full incident rather than from the source
   metadata (documented in `docs/data_card.md`). The information is recoverable
   by re-scanning `photos.csv.gz` from the Open Data bucket and joining on
   photo/observation id.

### Concrete actions

- [ ] **Add an attribution/credits page to the app** naming iNaturalist and its
      contributors as the training-data source, the CC licenses involved, and
      linking to iNaturalist. Cheap, and it covers the main obligation.
- [ ] **Recover the license breakdown** by re-scanning `photos.csv.gz`, so the
      data card can state the CC0 / CC BY / CC BY-NC split factually.
- [ ] **Keep the app free** unless and until the training data is re-sourced.
      Note this constraint in `GOALS.md` so a future "let's monetize it" idea
      hits the constraint rather than discovering it late.
- [ ] Keep the existing "confirm anything you plan to eat, remove, or plant"
      disclaimer — independent studies find plant ID apps unreliable for
      safety-critical decisions, and that is exactly the liability surface.

---

## Sources

- Rutgers study summary — Illinois Extension, *How accurate are photo-based
  plant identification apps?*
- *An Analysis of the Accuracy of Photo-Based Plant Identification Applications
  on Fifty-Five Tree Species* (the underlying study)
- *Plant identification applications do not reliably identify toxic and edible
  plants in the American Midwest* (PubMed)
- *Swipe Right: a Comparison of Accuracy of Plant Identification Apps for Toxic
  Plants* (PubMed)
- iNaturalist Terms of Use; iNaturalist Help — "Can I use the photos and sounds
  posted on iNaturalist?"
- PlantCLEF 2025 overview (LifeCLEF/ImageCLEF)
