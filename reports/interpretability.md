# What the model is actually doing — interpretability results

Run 2026-07-28 against `runs/b_stage2/model.pt` (the currently deployed model),
before the round-2 retrain. Figures in `reports/figures/`; raw numbers in
`reports/tree_attention.json`.

These are deliberately not generic visualizations — each one interrogates a
number this project already measured.

---

## 1. Feature space: the genus/species gap, made visible

`figures/embedding_map.png` — UMAP of the penultimate (pre-logit) embeddings,
the same space the OOD detector measures Mahalanobis distance in.

**Left panel, coloured by genus.** *Hydrangea* sits in its own island. *Pinus*
forms a large distinct region. *Toxicodendron* is a tight isolated cluster —
which is reassuring, because that's poison ivy, the safety-critical case, and it
matched PictureThis 3/3 in field testing. *Acer* and *Quercus* occupy broad but
largely separate territories, with a genuinely mixed zone where they meet.

**Right panel, the same points, but only *Quercus*, coloured by species.** There
is **no species-level structure at all** — *alba*, *rubra*, *velutina*,
*palustris*, *coccinea* and the rest are completely interleaved across the same
two blobs.

That is a picture of the exact number the tree benchmark produced:
**Quercus 35% species accuracy, 69% genus accuracy.** The genus is a coherent
region of feature space; the species inside it are not separated at all. It also
explains why the genus fallback works — the information the model reliably has
*is* the genus.

---

## 2. Grad-CAM: Workstream B fixed vocabulary, not perception

`figures/gradcam_hydrangea_old_vs_new.png` — the three hydrangea photos through
the old (2,360-class) and new (2,510-class) checkpoints.

Both models attend to the **same** regions: leaf margins, serrations, and
venation — the botanically diagnostic features. The old model was not failing to
*see* the plant. It was looking at the right evidence and had **nowhere to put
it**, so it landed on *Mentha spicata* (45%), *Philadelphus coronarius* (23%)
and *Diervilla lonicera* (50%).

**So the ornamental expansion solved a vocabulary problem, not a perception
problem.** That is a sharper claim than "the model now looks at the right
thing," and the figure supports it.

*Caveat:* the new model was warm-started from the old, so they share most of the
backbone and similar attention is expected. The informative part is that
identical attention produced completely different answers.

### A methodological note worth keeping

The textbook Grad-CAM target — the last convolutional layer — was **useless on
this architecture**:

| layer | resolution | cells above 0.05 |
|---|---|---|
| `head` (post conv_head) | 12×12 | 71.5% |
| `blocks[-1]` (the standard choice) | 12×12 | **3.5%** |
| `blocks[4]` (used) | **24×24** | 53.3% |

At `blocks[-1]` the gradient-weighted sum is almost entirely negative, so the
ReLU zeroes it and the map is empty. This was caught by measuring activation
coverage rather than by looking at the picture — the empty map is easy to
mistake for "the model attends to one small spot."

---

## 3. Attention across photo positions: a partial refutation

`reports/tree_attention.json` — 383 photos from multi-photo observations of 25
tree species (heavily *Quercus*, *Pinus*, *Carya*). Photo *position* within an
observation is a proxy for shot type: position 0 is nearly always the foliage
"hero" shot; later positions skew to bark, whole tree and habitat.

| position | n | top-1 | genus | confidence | attention entropy |
|---|---|---|---|---|---|
| 0 | 150 | **40.7%** | 77.3% | 71.4% | 0.848 |
| 1 | 150 | 32.0% | 68.0% | 67.8% | 0.852 |
| 2 | 83 | **25.3%** | 62.7% | 66.8% | 0.844 |

**Confirmed:** accuracy falls sharply with position — 15 points from the hero
shot to the third photo, at both species and genus level. Later views (bark,
form) really are much harder for the current model. That supports the round-2
multi-photo pull.

**Refuted:** the hypothesised *mechanism*. The prediction was that accuracy
would fall **and attention entropy would rise together** — the model having
nothing to lock onto in a bark photo. Entropy does not move at all
(0.848 → 0.852 → 0.844).

**Revised interpretation:** the model is not lost or unfocused on bark shots. It
attends just as decisively and reads them wrong — confidently applying
foliage-shaped reasoning to a photo of a trunk. Note also that entropy is *high
in absolute terms* (~0.85 of a maximum 1.0) at every position: on trees,
attention is broadly diffuse regardless of shot type.

### Caveats

- **Position is a proxy, not a label.** Some observers upload bark first. It's a
  real signal across hundreds of photos but noisy per image.
- **This cohort is the hardest available.** Species were selected by number of
  multi-photo observations, which favours large well-documented genera — 9
  *Quercus*, 7 *Pinus*, 4 *Carya*. The 40.7% at position 0 is therefore **not**
  comparable to the 66.5% overall tree figure in `stratified_eval.json`.

---

## 4. What this sets up

These are the **"before" measurements**. Re-running all three against the round-2
model gives a genuine before/after test of the rebuild:

1. **Does position-2 accuracy improve?** That is the direct test of whether
   training on bark and whole-tree views helps. Baseline: **25.3%**.
2. **Does the *Quercus* cloud develop species structure?** The confusable-genus
   cap raise (600 vs 300) is aimed exactly at this. Baseline: none visible.
3. **Does attention entropy fall on trees?** Baseline: ~0.85 at all positions.

Pre-registering the numbers before the retrain is the point — it makes the
comparison a test rather than a story told afterwards.
