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

import numpy as np
import timm
import torch
from PIL import ImageOps


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

        # Genus grouping, for the genus fallback. Two independent evaluations
        # (reports/tree_benchmark.md, reports/field_comparison_picturethis.md)
        # found the model's genus signal far stronger than its species signal —
        # "an oak" is a useful answer where a confident wrong oak is a harmful
        # one. P(genus) is the sum of its species' probabilities, so we build an
        # index once and scatter-add at inference.
        self.genera = sorted({c.split()[0] for c in self.classes})
        genus_pos = {g: i for i, g in enumerate(self.genera)}
        self._genus_index = torch.tensor(
            [genus_pos[c.split()[0]] for c in self.classes], dtype=torch.long)

        # Optional calibration temperature fit on held-out data.
        self.temperature = 1.0
        ckpt_dir = os.path.dirname(os.path.abspath(ckpt_path))
        sidecar = os.path.join(ckpt_dir, "temperature.json")
        if os.path.exists(sidecar):
            self.temperature = float(json.load(open(sidecar)).get("temperature", 1.0))

        # Optional OOD bank (scripts/build_ood_bank.py): whitening W + whitened
        # class means nu + a distance threshold. Lets identify() flag out-of-scope
        # inputs (garden ornamentals, non-plants) by feature distance. Absent (no
        # bank yet) => never flags, so this is inert until the bank is built.
        self.ood = None
        ood_path = os.path.join(ckpt_dir, "ood_bank.npz")
        if os.path.exists(ood_path):
            z = np.load(ood_path, allow_pickle=True)
            nu = torch.tensor(z["nu"], dtype=torch.float32)
            self.ood = {"W": torch.tensor(z["whiten"], dtype=torch.float32),
                        "nu": nu, "nu_sq": (nu * nu).sum(1),
                        "threshold": float(z["threshold"])}

    def _tensor(self, pil_image):
        """PIL -> preprocessed input tensor. Applies EXIF orientation FIRST so a
        phone photo isn't fed sideways (PIL.open doesn't auto-rotate); idempotent
        when the caller already transposed. Single source of truth for preprocessing."""
        img = ImageOps.exif_transpose(pil_image).convert("RGB")
        return self.tf(img).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def logits(self, pil_image):
        """Raw (uncalibrated) logit vector for one image — used to fit T."""
        return self.model(self._tensor(pil_image))[0].cpu()

    @torch.no_grad()
    def features(self, pil_image):
        """Penultimate (pre-logit) embedding for OOD scoring — distance in this
        feature space separates 'far from any trained plant' from 'hard plant'."""
        x = self._tensor(pil_image)
        f = self.model.forward_head(self.model.forward_features(x), pre_logits=True)
        return f[0].cpu()

    def _ood_score(self, feats):
        """Min Mahalanobis distance to any class centroid, in whitened space
        (see scripts/build_ood_bank.py). Larger = further from every trained class."""
        g = feats @ self.ood["W"].T
        d2 = (g * g).sum() - 2 * (g @ self.ood["nu"].T) + self.ood["nu_sq"]
        return float(d2.min())

    @torch.no_grad()
    def identify(self, pil_image, k=5):
        """Full result for the app: top-k candidates + an out_of_scope flag, set
        when an OOD bank is loaded AND the image sits beyond the distance
        threshold from every class. One backbone forward feeds both the
        classifier and the OOD feature (which must match the bank's features)."""
        x = self._tensor(pil_image)
        fmap = self.model.forward_features(x)
        logits = self.model.forward_head(fmap, pre_logits=False)[0]
        probs = (logits / self.temperature).softmax(dim=0)
        top = probs.topk(min(k, len(self.classes)))
        cands = [{"species": self.classes[i], "prob": float(p)}
                 for p, i in zip(top.values.cpu(), top.indices.cpu())]
        genus, genus_prob = self._top_genus(probs)
        out_of_scope, ood_score = False, None
        if self.ood is not None:
            feats = self.model.forward_head(fmap, pre_logits=True)[0].float().cpu()
            ood_score = self._ood_score(feats)
            out_of_scope = ood_score > self.ood["threshold"]
        return {"candidates": cands, "genus": genus, "genus_prob": genus_prob,
                "out_of_scope": out_of_scope, "ood_score": ood_score}

    @torch.no_grad()
    def predict(self, pil_image, k=5):
        """Return [{"species", "prob"}, ...] top-k. Probabilities are
        temperature-calibrated when a temperature.json is present."""
        probs = (self.model(self._tensor(pil_image))[0] / self.temperature).softmax(dim=0)
        top = probs.topk(min(k, len(self.classes)))
        return [{"species": self.classes[i], "prob": float(p)}
                for p, i in zip(top.values.cpu(), top.indices.cpu())]

    def _top_genus(self, probs):
        """Most likely genus and its probability — the SUM over that genus's
        species, which is why it can be far more confident than any single
        species when the model is torn between congeners."""
        totals = torch.zeros(len(self.genera))
        totals.index_add_(0, self._genus_index, probs.cpu())
        idx = int(totals.argmax())
        return self.genera[idx], float(totals[idx])
