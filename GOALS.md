# CT Plant ID — goal backlog

A phone app that photographs a plant, identifies the species (comprehensive
coverage of Connecticut's flora — thousands of species, not a hand-picked
subset), and says whether it's native, introduced, or a weed/invasive.
North star: a PictureThis-style tool, scoped to CT, done honestly —
including a real "not sure" answer and a candidate list rather than one
forced guess. Secondary goal: learn the perception stack (detection +
long-tail / out-of-distribution handling) that drives computer vision at
self-driving companies.

Run `/goal next` to dispatch the first unchecked Queue item (copy the
goal-loop harness from ../landscaping-planner/.claude/ first — see
"Verification" below for the twist the ML phases need).

## The end goal: comprehensive CT coverage

The destination is a functionally good app covering ALL of Connecticut's
flora — every vascular species recorded in the state (~2,000–2,500), not a
convenient subset. That is the definition of done for this project, and no
milestone below is the finish line until coverage is comprehensive.

The staging below (start with the well-sampled "head" species, ship a
working app, then expand toward the full checklist) is a **route, not a
retreat**. Head-first exists only so there's a functioning product early
and the hard tail work happens against a working system — every phase is
built to scale to the full species list, and "we shipped the head" is
explicitly NOT "done." The active-learning flywheel (later phase) is the
engine that closes the gap to comprehensive; the project stays open until it
has.

## The core challenge: the long tail

Comprehensive CT coverage means ~2,000–2,500 species, and their image
counts are wildly uneven: common species have thousands of iNaturalist
photos, many rare ones have single digits. This shapes every decision:

- **Top-1 accuracy is the wrong headline.** A model can't reliably pick one
  of 2,000 species from a phone photo, and it shouldn't pretend to. Judge it
  by **top-5 accuracy** (show candidates, like PictureThis does),
  **genus-level accuracy** (right genus is useful even if species is wrong),
  and per-class metrics **stratified by data availability** (head / mid /
  tail) — a single average hides the tail, which is where the work is.
- **Graceful degradation is the product.** Functional quality comes from
  returning a ranked candidate list + confidence and honestly abstaining
  ("not sure — here are the closest matches"), NOT from squeezing top-1 on
  species with 8 training images. Design for this, don't fight it.
- **Cover what has data; expand via the flywheel.** Include every species
  with enough images to learn; for the sparse tail, fall back to
  genus/family and let the active-learning loop (later phase) grow coverage
  over time. Document what's covered vs. deferred — never silently drop
  species and call it complete.
- **Lean on an iNaturalist-pretrained backbone.** Models pretrained on the
  iNat taxonomy already recognize thousands of plant species. The strong
  baseline is to start from one and restrict/fine-tune its output space to
  the CT checklist, rather than training thousands of classes from scratch.

## How these goals are verified

Two kinds of goals, checked differently:

- **Software goals** (data pipeline, attribute lookup, API, PWA) — verified
  the landscaping way: full test suite green AND an independent Codex review
  returning VERDICT: SHIP against the goal text.
- **ML goals** (classifier, detection, calibration/OOD) — a passing suite is
  necessary but NOT sufficient; the real gate is a **metric threshold on a
  held-out set**, from an eval script the goal produces, and for this
  project the metric is always reported **stratified head/mid/tail**, never
  a single average. Thresholds marked "(provisional)" are set for real after
  the first baseline — don't let a round number override an honest eval, and
  don't let a good head-class average paper over a broken tail.

Non-negotiables: no train/test leakage (an observation's photos never
straddle splits), every number reproducible from a script, and honest
reporting when a target is missed (state the number and a plan).

## Stack (planned)

Python, PyTorch + timm (iNaturalist-pretrained backbones), FastAPI + a
vanilla-JS PWA (reusing landscaping patterns), cloud GPU for training,
coremltools/TFLite for the optional on-device phase. Data: iNaturalist
research-grade observations (primary, taxonomy-aligned) + Pl@ntNet-300K;
CT species checklist from GBIF occurrence data cross-checked against a state
flora source; native/introduced/invasive status from USDA PLANTS and the CT
invasive species list.

## Queue

- [ ] **Comprehensive CT species checklist + data pipeline.** Assemble the
  FULL list of vascular plant species recorded in Connecticut from GBIF
  occurrence data, cross-checked against a state flora reference — the whole
  ~2,000–2,500, this is the permanent target the pipeline serves. Store it
  with taxonomy (species/genus/family) so hierarchical fallback is possible
  later. Build a reproducible script that pulls research-grade iNaturalist
  images for every species with data, cleans them, and writes train/val/test
  splits keyed so no observation straddles splits. The pipeline must handle
  the full checklist; the first training run may start from the well-sampled
  head, but the data layer is built for all of it from day one. Produce a
  data card showing total class count, the head/mid/tail distribution of
  image counts per species, and an explicit list of species currently
  deferred for insufficient data (a backlog to close, not a scope cut).
  Verification: script rebuilds the dataset from scratch; tests assert
  observation-level split integrity, that the class list matches the full
  checklist, and that the head/mid/tail tiers are computed and recorded;
  data card committed. (Software goal.)

- [ ] **Baseline long-tail classifier (cloud training).** Fine-tune an
  iNaturalist-pretrained backbone (EfficientNetV2 / ConvNeXt / ViT via timm)
  with its output restricted to the CT checklist, using long-tail-aware
  training (class-balanced sampling or loss). This first run may cover the
  well-sampled head/mid tiers to get a working model fast — but that is a
  starting point on the way to full coverage, not the finish; the eval must
  always report against the FULL checklist so the remaining gap to
  comprehensive is visible and tracked, never hidden by scoping the metric
  to trained classes only. Track runs. Deliver the model plus a reproducible
  eval report of **top-1, top-5, and genus-level accuracy, each stratified
  head/mid/tail**, plus a confusion analysis of the worst-confused pairs,
  and the current species-coverage fraction of the full CT checklist. Metric
  gate (provisional, set for real after baseline): **top-5 ≥ 85% on head+mid
  tiers** and a **documented tail strategy** (genus fallback) rather than a
  tail top-1 target that can't be met. Verification: eval script reproduces
  the headline numbers and the coverage fraction; leakage check passes;
  tests cover data-loading and metric code. (ML goal — metric 1.)

- [ ] **Species → attributes lookup and weed logic.** Map every species in
  the checklist to native / introduced / invasive / commonly-weedy —
  sourced programmatically from USDA PLANTS and the CT invasive species list
  (thousands of rows, so database-sourced with citations, not hand-typed).
  The app's "weed or not" answer is DERIVED from this table plus the genus
  fallback, never predicted by the model. Verification: coverage of 100% of
  the class list (with an explicit "status unknown" category for gaps rather
  than silent blanks); provenance recorded per source; unit tests for the
  lookup, the weed-decision rule, and the genus-level fallback when species
  is uncertain. (Software goal.)

- [ ] **Server + PWA (the shippable product).** A FastAPI endpoint that takes
  a photo and returns a **ranked candidate list** (top-k species with
  confidences) + attributes, plus an explicit "not sure" / genus-level
  answer when confidence is low. A vanilla-JS PWA with phone camera capture,
  a candidate-list result card (PictureThis-style), and the abstain state.
  Installable to a home screen. Verification: API tests (valid image →
  documented candidate-list shape; non-image/oversize → 4xx); the full
  photo→candidates loop exercised on a test image; the PWA driven in a
  browser to confirm capture→submit→result; all green; Codex SHIP.
  (Software goal.)

- [ ] **Detection front-end (detect → crop → classify).** Add a localization
  stage so the app works on cluttered real photos: detect the plant in the
  frame (YOLO/DETR-style box, or a lighter saliency crop), crop, then
  classify. Mirrors a real perception pipeline (find, then identify). Metric
  gate: on a held-out set of cluttered real-world photos, detect→classify
  beats whole-image top-5 by a **measured margin (provisional ≥ 5 points)**;
  report before/after. Verification: pipeline-stage tests, the before/after
  eval script, all green. (ML goal — metric 2.)

- [ ] **Long-tail OOD / "not sure" handling and calibration.** With thousands
  of classes the model is MORE prone to confident errors, so this phase
  matters more than in a small-class app. Add confidence calibration
  (temperature scaling) and out-of-distribution detection so non-CT plants
  and non-plant photos return "not sure" instead of a wrong species — the
  same long-tail/unknown-object problem AV perception faces. Metric gates:
  calibration reduces expected calibration error (ECE) vs. the raw model;
  OOD detection reaches **AUROC ≥ 0.85 (provisional)** separating
  in-distribution CT plants from a held-out OOD set (non-plants + species
  deliberately excluded from training). Verification: app returns "not sure"
  on OOD test inputs; eval reports ECE before/after and OOD AUROC; tests
  cover the threshold logic. (ML goal — metric 3.)

- [ ] **Active-learning loop.** Capture low-confidence and user-flagged
  photos, provide a small review/label step, and fold labeled batches into
  the next training round — the flywheel that grows tail coverage over time
  and mirrors how AV teams mine their long tail. Verification: the loop
  demonstrated end-to-end on one batch (uncertain images surfaced by the
  selection rule, labeled, retrained, metric change reported, ideally on a
  tail class); tests cover the uncertain-image selection logic. (Mixed.)

- [ ] **(Stretch — only after the server app is solid.) On-device model.**
  Convert the classifier to Core ML (iOS) or TFLite, quantize to fit, run
  inference on the phone with no server. Larger class count makes size and
  quantization tighter, so this is genuinely non-trivial. Metric gate:
  converted model runs on the target under a **stated size budget** with a
  top-5 accuracy drop **within a stated tolerance** vs. the server model.
  Verification: on-device inference demonstrated on a real photo; a
  before/after top-5 + size report. (ML goal — metric 4.)

## Done

(completed goals move here with the date)
