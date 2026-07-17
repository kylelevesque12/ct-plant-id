"""iNaturalist image pipeline (stub — GOALS.md phase 1).

Planned responsibilities:
  - resolve the comprehensive CT species checklist (GBIF + state flora)
  - pull research-grade iNaturalist images per species
  - write train/val/test splits keyed by OBSERVATION id, so a single
    observation's photos never straddle splits (the leakage guard)
  - emit a data card: total classes, head/mid/tail distribution, deferred
    species list

Left as a stub with an honest signature; the first goal implements it.
"""


def build_dataset(checklist_path: str, out_dir: str) -> dict:
    raise NotImplementedError(
        "Implemented by GOALS.md phase 1 (data pipeline).")
