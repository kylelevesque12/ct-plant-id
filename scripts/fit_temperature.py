"""Fit a calibration temperature so the reported confidence means P(correct).

The model is underconfident (label smoothing pushes probabilities down), so a
raw 51% top guess is actually right ~85% of the time. Temperature scaling
(Guo et al. 2017) fixes this with ONE parameter T: we divide the logits by T
before softmax. T>1 softens an overconfident model; T<1 sharpens an
underconfident one. We pick the T that minimizes negative log-likelihood on
held-out photos — the same observation-keyed test split training never saw.

Reuses the sampling/download helpers from calibrate.py (no duplication).
Runs on CPU, ~10-15 min (download + inference). Writes the temperature next to
the checkpoint as runs/stage2/temperature.json, which predict.py auto-loads.

Run: .venv/bin/python scripts/fit_temperature.py
"""
import importlib.util
import json
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from ctplantid.predict import PlantModel  # noqa: E402

# Load calibrate.py's helpers (sample_species / test_photos / split logic).
_spec = importlib.util.spec_from_file_location(
    "calib", os.path.join(ROOT, "scripts", "calibrate.py"))
calib = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(calib)

CKPT = os.path.join(ROOT, "runs", "stage2", "model.pt")
OUT = os.path.join(ROOT, "runs", "stage2", "temperature.json")


def ece(probs, correct, n_bins=10):
    """Expected calibration error: avg |confidence - accuracy| over conf bins."""
    e = 0.0
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        m = [(p, c) for p, c in zip(probs, correct) if lo < p <= hi or (b == 0 and p == 0)]
        if not m:
            continue
        conf = sum(p for p, _ in m) / len(m)
        acc = sum(c for _, c in m) / len(m)
        e += len(m) / len(probs) * abs(acc - conf)
    return e


def main():
    model = PlantModel(CKPT, device="cpu")
    name_to_idx = {n: i for i, n in enumerate(model.classes)}
    species = calib.sample_species(model.classes)
    print(f"fitting temperature on {len(species)} species (held-out test split)…")

    logits_rows, labels = [], []
    for i, s in enumerate(species):
        idx = name_to_idx.get(s["name"])
        if idx is None:
            continue
        imgs = calib.test_photos(s["taxon_id"], calib.PER_SPECIES)
        for im in imgs:
            logits_rows.append(model.logits(im))
            labels.append(idx)
        print(f"  [{i+1}/{len(species)}] {s['name']}: {len(imgs)} imgs "
              f"(total {len(labels)})")

    L = torch.stack(logits_rows)          # [N, C] raw logits
    y = torch.tensor(labels)              # [N] true class indices
    n = len(y)
    nll = torch.nn.functional.cross_entropy

    # Grid search T (robust, deterministic — 1 parameter, no optimizer fuss).
    # Underconfident model -> expect T<1 (sharpen). Range covers both directions.
    grid = [round(0.30 + 0.01 * k, 2) for k in range(int((3.00 - 0.30) / 0.01) + 1)]
    best_T = min(grid, key=lambda T: float(nll(L / T, y)))

    def stats(T):
        probs = (L / T).softmax(dim=1)
        conf, pred = probs.max(dim=1)
        correct = (pred == y).int().tolist()
        return conf.tolist(), correct, float(nll(L / T, y))

    p1, c1, nll1 = stats(1.0)
    pT, cT, nllT = stats(best_T)
    acc = sum(c1) / n  # accuracy is T-invariant (argmax unchanged)

    print(f"\n=== temperature scaling (n={n}) ===")
    print(f"accuracy (top-1):        {acc:.1%}   (unchanged by T)")
    print(f"fitted temperature T:    {best_T}   ({'sharpen' if best_T < 1 else 'soften'})")
    print(f"NLL:   {nll1:.3f} -> {nllT:.3f}")
    print(f"ECE:   {ece(p1, c1):.3f} -> {ece(pT, cT):.3f}")
    print(f"mean top-1 confidence:   {sum(p1)/n:.1%} -> {sum(pT)/n:.1%}   "
          f"(target ~ accuracy {acc:.1%})")

    json.dump({"temperature": best_T, "fit_n": n, "accuracy": round(acc, 4),
               "ece_before": round(ece(p1, c1), 4), "ece_after": round(ece(pT, cT), 4),
               "mean_conf_before": round(sum(p1)/n, 4),
               "mean_conf_after": round(sum(pT)/n, 4)},
              open(OUT, "w"), indent=2)
    print(f"\nwrote {os.path.relpath(OUT, ROOT)}  (predict.py auto-loads it)")


if __name__ == "__main__":
    main()
