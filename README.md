# CT Plant ID

A phone app that photographs a plant, identifies the species (aiming at
comprehensive coverage of Connecticut's flora — thousands of species), and
reports whether it's native, introduced, or a weed/invasive. A
PictureThis-style tool scoped to CT, built to also exercise the perception
stack (detection + long-tail / out-of-distribution handling) used in
computer vision at self-driving companies.

See `GOALS.md` for the phased backlog and how goals are verified.

## Approach in one paragraph

The model predicts **species**; a separate lookup table turns that into the
"native / weed" answer (a weed is contextual, not a visual class). Because
comprehensive CT coverage is a long-tailed problem (~2,000–2,500 species
with very uneven image counts), the model is judged on top-5 and
genus-level accuracy stratified by data availability, and the app returns a
ranked candidate list with an honest "not sure" rather than one forced
guess. Training starts from an **iNaturalist-pretrained backbone** and
fine-tunes it — see "Fine-tuning" below.

## Layout

```
src/ctplantid/     library code
  species.py       CT checklist handling, taxonomy, head/mid/tail tiers
  metrics.py       top-k / genus-level / stratified accuracy
  model.py         load a pretrained backbone, swap in a CT-sized head
  dataset.py       (stub) iNaturalist image pipeline
scripts/
  inspect_model.py load a pretrained model and print its architecture
tests/             offline unit tests (no GPU, no model download)
.claude/           goal-loop harness (see GOALS.md)
```

## Fine-tuning: you keep full visibility

"Fine-tuning" here means downloading an existing model's **architecture +
weights** and continuing to train it on CT data. Nothing is hidden:

- The architecture is a plain `torch.nn.Module` — `print(model)` shows every
  layer. `scripts/inspect_model.py` does exactly this.
- You can **evaluate the pretrained model as-is** (before any fine-tuning)
  to get a baseline number, then measure how much fine-tuning adds. That
  baseline is scientifically useful, not a throwaway.
- The weights, training recipe, and papers for these backbones are public
  (timm + Hugging Face model cards), so there's no black box — unlike
  calling a closed API.

Fine-tuning typically **replaces the final classification head** (the last
layer mapping features → classes) because the CT species set differs from
what the backbone was trained on, and keeps the **feature-extractor
backbone** (frozen or trained at a small learning rate).

## Setup

```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q          # unit tests (no torch needed)
.venv/bin/python scripts/inspect_model.py   # see a real model's architecture
```
