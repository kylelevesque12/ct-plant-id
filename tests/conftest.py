"""Test fixtures for the FastAPI app.

The whole point here is to NEVER load the real 89MB checkpoint. `app/main.py`
does `from ctplantid.predict import PlantModel` and then instantiates it at
import time (`model = PlantModel(...)`). So the trick is:

  1. Put `src/` on sys.path (so `ctplantid` is importable, same as main does).
  2. Replace `ctplantid.predict.PlantModel` with a tiny fake BEFORE `app.main`
     is ever imported. Because main binds the name at its own import time, it
     picks up whatever `ctplantid.predict.PlantModel` is at that moment.
  3. Import `app.main` fresh (evicting any cached copy) so its module-level
     `PlantModel(...)` call constructs the fake, not the real model.

The fake's `predict` reads its top-k probabilities from a class attribute, so a
test can dial `FakePlantModel.probs` up or down between requests to exercise the
not_sure threshold without re-importing anything.
"""
import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class FakePlantModel:
    """Stand-in for ctplantid.predict.PlantModel — no torch, no checkpoint.

    `classes` is a small, known species list. `predict` returns the documented
    [{"species", "prob"}, ...] shape, descending by prob, driven by the
    class-level `probs` list so tests can control confidence deterministically.
    """

    # Five plausible CT species; index 0 is an invasive, so if the real
    # attributes.py is present the top candidate exercises the invasive path.
    classes = [
        "Rosa multiflora",
        "Acer rubrum",
        "Toxicodendron radicans",
        "Quercus alba",
        "Betula lenta",
    ]

    # Default: a confident top-1 (0.85 > threshold => not_sure False).
    probs = [0.85, 0.07, 0.04, 0.03, 0.01]
    # Default in-scope; a test can set this True to exercise the OOD path.
    out_of_scope = False

    def __init__(self, ckpt_path=None, device=None):
        # Accept the same call signature as the real model but do no work.
        self.ckpt_path = ckpt_path
        self.device = "cpu"

    def predict(self, pil_image, k=5):
        n = min(k, len(self.classes))
        return [
            {"species": self.classes[i], "prob": float(self.probs[i])}
            for i in range(n)
        ]

    def identify(self, pil_image, k=5):
        # Mirrors the real model's rich result the app consumes.
        return {"candidates": self.predict(pil_image, k),
                "out_of_scope": self.out_of_scope, "ood_score": None}


@pytest.fixture
def client():
    """A TestClient whose app was built against FakePlantModel.

    Patches the PlantModel symbol, then imports app.main fresh so the
    module-level model instance is the fake. Resets FakePlantModel.probs to the
    confident default for isolation between tests.
    """
    from fastapi.testclient import TestClient

    FakePlantModel.probs = [0.85, 0.07, 0.04, 0.03, 0.01]
    FakePlantModel.out_of_scope = False

    predict_mod = importlib.import_module("ctplantid.predict")
    real_plantmodel = predict_mod.PlantModel
    predict_mod.PlantModel = FakePlantModel

    # Evict any previously imported app.main so its top-level model load re-runs
    # with the fake in place.
    sys.modules.pop("app.main", None)
    main = importlib.import_module("app.main")

    try:
        with TestClient(main.app) as c:
            yield c
    finally:
        predict_mod.PlantModel = real_plantmodel
        sys.modules.pop("app.main", None)
