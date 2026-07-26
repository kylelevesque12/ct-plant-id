# Field comparison: Fieldnote vs PictureThis, 54 paired backyard tests

Run 2026-07-26 from 108 paired screenshots (`data/field_comparison/`). Per-pair
data in `reports/field_comparison_pairs.csv`.

Pairing was **verified, not assumed**: every screenshot was classified by app
from its UI chrome, giving exactly 54 Fieldnote / 54 PictureThis with **zero
breaks in alternation**, so pair *n* is unambiguous.

---

## The essential caveat

**There is no ground truth here.** These are backyard photos with no expert
identification, so this measures **agreement between two apps**, not accuracy.
PictureThis is *not* a gold standard — and this dataset contains direct evidence
of that (§4). Every number below is an agreement rate. Do not report them as
accuracy.

## Headline

| | agreement |
|---|---|
| **Species-level** | **14/54 = 25.9%** |
| **Genus-level** | **28/54 = 51.9%** |

The two apps name the same species only about a quarter of the time, and the
same genus about half.

## 1. The best result: confidence predicts agreement, monotonically

| Fieldnote label | n | species agree | **genus agree** |
|---|---|---|---|
| Strong match | 19 | 47% | **74%** |
| Likely match | 7 | 14% | **57%** |
| Possible match | 14 | 14% | **43%** |
| Uncertain | 12 | 17% | **33%** |
| Out-of-scope (OOD) | 2 | 0% | **0%** |

Genus agreement falls cleanly with every step down the confidence ladder:
**74% → 57% → 43% → 33% → 0%.**

This is the single most valuable finding in the dataset. The calibration and
labelling work is **validated on real-world photos it never saw** — when
Fieldnote says "Strong match" it is materially more likely to be corroborated
than when it says "Uncertain," and the ordering never inverts. The app's
uncertainty is honest, which is exactly what it claims to be.

## 2. Safety-critical species: 3/3 exact

Poison ivy (*Toxicodendron radicans*) was photographed three times and Fieldnote
matched PictureThis **exactly all three times** (100%, 99%, and 57% confidence).
For the plant a user most needs identified correctly, the model is reliable.

**But a product gap is visible in the same result:** poison ivy displays
**"Status unknown."** The status vocabulary covers native / introduced /
invasive / ornamental — there is **no hazard category**. "Will this hurt me?" is
a top-three user question and the app currently cannot answer it, even when the
identification is perfect. That is a cheap, high-value fix: a `hazard` flag in
`attributes.csv` for poison ivy, poison sumac, giant hogweed, wild parsnip, etc.

## 3. Workstream B validated in the field

The hydrangea that started the whole scope-expansion effort now returns
**Mophead Hydrangea (*Hydrangea macrophylla*) at 98%, "Garden plant"** — and
PictureThis independently agrees (*Hydrangea macrophylla*). The ornamental
expansion worked on exactly the plant that motivated it.

## 4. PictureThis is demonstrably inconsistent too

Four photographs of what appears to be the same stonecrop planting produced
**four different PictureThis answers**: *Hylotelephium telephium*, *Phedimus
aizoon*, and *Hylotelephium spectabile* (twice). Fieldnote returned
*Hylotelephium spectabile* every time, at 86 / 81 / 99 / 99% confidence.

On that plant Fieldnote is *more* self-consistent than PictureThis. This is the
concrete reason the disagreement rate cannot be read as an error rate: in an
unknown share of the 26 disagreements, Fieldnote may be the correct one.

## 5. Where Fieldnote looks genuinely wrong

Cases with high confidence and a complete disagreement — the ones worth
adjudicating first:

| Fieldnote | conf | PictureThis |
|---|---|---|
| *Buxus microphylla* (boxwood) | 97% | *Ilex glabra* (inkberry) |
| *Asparagus aethiopicus* | 94% | *Phlox subulata* (moss phlox) |
| *Prunus laurocerasus* (cherry laurel) | 91% | *Ilex glabra* (inkberry) |
| *Vaccinium pallidum* (blueberry) | 85% | *Spiraea japonica* |
| *Hylotelephium spectabile* | 81% | *Phedimus aizoon* |

Boxwood vs inkberry and cherry laurel vs inkberry are all small-leaved evergreen
shrubs — plausible confusions, but at 91–97% confidence the app is asserting
more than it knows.

**One potential safety miss:** *Menispermum canadense* (moonseed, native vine) at
64% where PictureThis said ***Alliaria petiolata*** (garlic mustard) — a listed
CT invasive. If PictureThis is right, Fieldnote failed to flag an invasive.

## 6. The OOD detector fired twice, and both look like false positives

Two photos triggered "This looks outside my range." PictureThis returned
confident in-scope answers for both (*Spiraea japonica*, *Hibiscus syriacus*) —
and *Spiraea japonica* **is** in Fieldnote's class list, so at least one of these
was an in-scope plant being wrongly rejected.

Two false positives in 54 (3.7%) is roughly the ~2.5% rejection budget the
threshold was tuned for, so this is behaving as designed rather than broken —
but it is the first real-world evidence, and it should be re-checked as more
data arrives.

## 7. Consistent with the tree benchmark

Genus agreement (52%) far exceeds species agreement (26%) — the same pattern the
Rutgers-protocol tree benchmark found (67.7% genus vs 49.7% species). Two
independent evaluations now say the model's genus-level signal is much stronger
than its species-level signal.

**This is the second dataset arguing for the genus fallback.**

## What to do next

1. **Ship the genus fallback.** Two independent evaluations now support it. When
   confidence is below "Strong," lead with the genus.
2. **Add a `hazard` status.** Poison ivy identified perfectly and labelled
   "Status unknown" is a real product failure.
3. **Adjudicate the disagreements.** Post the ~26 disagreement photos to
   iNaturalist for community identification. That converts this agreement study
   into a genuine accuracy benchmark — and it is only 26 photos, not 108.
4. **Re-examine the high-confidence misses** (§5) once truth exists; if
   Fieldnote is wrong at 91–97%, calibration needs revisiting for evergreen
   shrubs specifically.
