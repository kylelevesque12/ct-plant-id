# Tree benchmark — Fieldnote vs the published PictureThis numbers

Run 2026-07-25 by `scripts/eval_tree_benchmark.py`; raw results in
`reports/tree_benchmark.json`. This is Option A from
`docs/benchmarking_vs_picturethis.md`: replicate the Rutgers protocol on our own
model so there's a defensible comparison to a published number.

**This is a negative result, and an important one.**

## Headline

648 held-out test photos across 54 of 55 cohort species (98.2% coverage; only
*Malus floribunda* is not in scope).

| | genus top-1 | species top-1 |
|---|---|---|
| **PictureThis** (published) | **97.3%** | **83.9%** |
| **iNaturalist** (published) | 92.3% | 69.6% |
| **Fieldnote** | **67.7%** | **49.7%** |

Fieldnote's top-5 numbers: species 73.9%, genus 84.6%.

PictureThis is substantially better than us on trees. That gap is far too large
to explain away with the methodology caveats below — it is real.

## The finding that matters most

**Our headline 80.1% top-1 does not hold on trees.** The overall figure is
measured across a species mix dominated by herbaceous plants with distinctive
flowers. On trees the same model scores **49.7%**. Accuracy is not uniform
across plant types, and the project should stop quoting a single number as if it
were.

## Why it fails — three separate causes

### 1. Within-genus confusion (the dominant effect)

| genus | species top-1 | genus top-1 | gap |
|-------|--------------|-------------|-----|
| Quercus | 35% | 69% | **35 pts** |
| Pinus | 50% | 89% | **39 pts** |
| Carya | 38% | 75% | **38 pts** |
| Prunus | 46% | 79% | 33 pts |
| Betula | 56% | 78% | 22 pts |

The model usually knows it's an oak and can't say which oak. Sample predictions
for *Quercus rubra* — `Quercus robur 40%, Quercus stellata 18%, Quercus
muehlenbergii 9%` — are coherent, sensible confusions, not noise. With 15
Quercus species in scope this is a genuine fine-grained problem.

### 2. Near-empty classes that are confidently wrong

*Metasequoia glyptostroboides* scores **8% at genus level**, and its predictions
are not near-misses:

```
Deutzia crenata 92%   |  Salvia officinalis 80%  |  Persicaria punctata 26%
```

A dawn redwood called a sage at 80% confidence. The class exists in the output
space but has almost no training data (1 CT observation), so it was never
learned — and neither the calibration nor the OOD bank catches it, because the
input *is* a real plant sitting inside the general plant manifold.

**This is a new failure mode**: not out-of-scope, not low-confidence — a
confidently wrong answer from a class that should never have been trainable.

### 3. CT observation count does not predict accuracy — but data scale is NOT ruled out

| CT observation count | mean species top-1 |
|---|---|
| < 200 | **49%** |
| ≥ 200 | **49%** |

Identical. *Quercus alba* has 655 CT observations and scores 33%; *Ulmus
parvifolia* has 1 and scores 83%.

**Important caveat on this result.** CT observation count is *not* training-image
count. The dataset was built with a **per-species cap of 300 images, and the
median species hit that cap** (`docs/data_card.md`). So *Quercus alba*'s 655
observations became at most 300 training images — the same as a species with 300
observations. The correlation above is therefore measuring a variable that was
largely flattened by the cap, and it is much weaker evidence than it first
appears.

The honest conclusion is narrower: **within the range this dataset actually
spans, availability in Connecticut doesn't predict per-species accuracy.**
Whether *raising the cap* would help — more images per species for confusable
genera — is untested and remains a live option.

## Methodology caveats (state these with any citation of the numbers)

- **Different photos.** Ours are crowd-sourced iNaturalist test-split images;
  theirs were arborist-taken sets of bark and leaves, 4+ per species.
- **Image type favours us, if anything.** iNat photos skew to leaves, flowers
  and whole plants — the easy types. The study found bark-only drops PictureThis
  to ~65% genus / ~52% species. So the gap is unlikely to be an artefact of us
  getting harder pictures.
- **The cohort is a reconstruction** of the study's 55 species (the exact list is
  paywalled), chosen as common Northeast street and native forest trees.
- **Class-count asymmetry.** We choose among 2,510 classes. PictureThis chooses
  among ~400k but is a mature commercial product with far more data per class.
- Species we don't carry are reported, not silently dropped; the
  coverage-penalised figures (48.8% species / 66.5% genus) assume the missing
  species would always be wrong.

## What to do about it

1. **Ship the genus fallback.** `GOALS.md` has always described it and it is not
   implemented. Genus top-5 is **84.6%** — when the model can't pick a species it
   very often knows the genus, and "an oak — likely red oak or black oak" is a
   genuinely useful answer where a confident wrong species is a harmful one.
   This is the highest-value fix and needs no retraining.
2. **Prune or merge near-empty classes.** A class with a handful of training
   images cannot be learned but can still be predicted confidently. Either drop
   below a minimum-image threshold, or fold such species into a genus-level
   class. This directly addresses cause 2.
3. **Report accuracy stratified by plant type** (tree / shrub / herbaceous /
   graminoid) rather than one global number. The tree cohort is evidence that
   the global figure hides a much weaker sub-population.
4. **Consider tree-specific data**: bark, whole-tree form and winter twigs are
   how trees are actually identified in the field, and are under-represented in
   a dataset dominated by flower close-ups.

## Honest summary for the portfolio

The right framing is not "we're close to PictureThis." It's:

> Benchmarked against published independent numbers, the model is markedly
> weaker on trees (49.7% species / 67.7% genus) than on the overall species mix
> (80.1%). Diagnosis identifies fine-grained within-genus confusion and
> confidently-wrong sparse classes as concrete causes, with per-species data
> scale still open because the training set was capped at 300 images/species.

That is a more credible thing to present than a favourable headline, and it
comes with a concrete, evidence-backed remediation plan.
