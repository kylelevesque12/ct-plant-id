"""Shared inference for the app: load a trained checkpoint, predict top-k.

The checkpoint (from train.py) carries everything needed — class list, backbone
name, and the exact data_config — so serving reproduces training's preprocessing
with no guesswork. This module is the single source of truth for how a photo
becomes predictions; the API and any tests import from here.

Calibration: the model is underconfident (label smoothing). If a
`temperature.json` sits next to the checkpoint (fit by scripts/fit_temperature.py),
we divide the logits by that temperature before softmax so the reported
probability actually means "P(this ID is correct)". Default temperature 1.0 = raw.
"""
import json
import os

import timm
import torch


class PlantModel:
    def __init__(self, ckpt_path, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available()
                                 else "mps" if torch.backends.mps.is_available()
                                 else "cpu")
        ck = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.classes = ck["classes"]
        self.model = timm.create_model(ck["backbone"], pretrained=False,
                                       num_classes=len(self.classes))
        self.model.load_state_dict(ck["state_dict"])
        self.model.to(self.device).eval()
        self.tf = timm.data.create_transform(**ck["data_config"], is_training=False)

        # Optional calibration temperature fit on held-out data.
        self.temperature = 1.0
        sidecar = os.path.join(os.path.dirname(os.path.abspath(ckpt_path)),
                               "temperature.json")
        if os.path.exists(sidecar):
            self.temperature = float(json.load(open(sidecar)).get("temperature", 1.0))

    @torch.no_grad()
    def logits(self, pil_image):
        """Raw (uncalibrated) logit vector for one image — used to fit T."""
        x = self.tf(pil_image.convert("RGB")).unsqueeze(0).to(self.device)
        return self.model(x)[0].cpu()

    @torch.no_grad()
    def predict(self, pil_image, k=5):
        """Return [{"species", "prob"}, ...] top-k. Probabilities are
        temperature-calibrated when a temperature.json is present."""
        x = self.tf(pil_image.convert("RGB")).unsqueeze(0).to(self.device)
        probs = (self.model(x)[0] / self.temperature).softmax(dim=0)
        top = probs.topk(min(k, len(self.classes)))
        return [{"species": self.classes[i], "prob": float(p)}
                for p, i in zip(top.values.cpu(), top.indices.cpu())]
