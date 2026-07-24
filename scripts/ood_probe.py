"""Workstream A first-cut: do bank-free OOD signals separate the hydrangea
(out of scope) from real in-scope CT plants?

Two signals that need NO training-feature bank (which isn't local):
  - energy = -logsumexp(logits): low = confident/in-dist, high = OOD.
  - TTA disagreement: run K augmented crops; if the top-1 flips across views
    (low modal agreement, high predictive entropy) the model is guessing.
    This literally turns the "three angles disagree" failure into a detector.

Compares OOD probes (data/ood_probe/*) against the few in-scope species we have
locally. A real threshold needs more in-scope species (iNat download) — this is
a direction check, not the final calibration.

Run: .venv/bin/python scripts/ood_probe.py
"""
import glob
import os
import sys

import torch
import torchvision.transforms as T

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from ctplantid.predict import PlantModel  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402

CKPT = os.path.join(ROOT, "runs", "stage2", "model.pt")
K = 12  # augmented views per image


def build_tta(ckpt):
    cfg = torch.load(ckpt, map_location="cpu", weights_only=False)["data_config"]
    size = cfg["input_size"][-1]
    return T.Compose([
        T.RandomResizedCrop(size, scale=(0.6, 1.0), ratio=(0.8, 1.25)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.15, 0.15, 0.15),
        T.ToTensor(),
        T.Normalize(cfg["mean"], cfg["std"]),
    ])


@torch.no_grad()
def scores(m, tta, path):
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    # eval pass
    logits = m.model(m._tensor(img))[0]
    energy = -torch.logsumexp(logits, dim=0).item()
    calib_top = (logits / m.temperature).softmax(0).max().item()
    # TTA pass: K augmented views
    batch = torch.stack([tta(img) for _ in range(K)]).to(m.device)
    p = m.model(batch).softmax(1)                 # [K, C]
    top1 = p.argmax(1)
    modal = torch.bincount(top1).max().item() / K  # fraction agreeing on the mode
    mean_p = p.mean(0)
    entropy = -(mean_p * (mean_p + 1e-9).log()).sum().item()
    return dict(energy=energy, calib_top=calib_top, tta_agree=modal, entropy=entropy)


def main():
    m = PlantModel(CKPT, device="cpu")
    tta = build_tta(CKPT)

    ood = sorted(glob.glob(os.path.join(ROOT, "data", "ood_probe", "*")))
    inscope = []
    for sp in sorted(os.listdir(os.path.join(ROOT, "data", "images"))):
        inscope += sorted(glob.glob(os.path.join(ROOT, "data", "images", sp, "*")))[:5]

    hdr = f"{'image':<34} {'energy':>8} {'calibTop':>8} {'ttaAgree':>8} {'entropy':>8}"
    print("\n### OOD probes (hydrangea — SHOULD score OOD) ###")
    print(hdr)
    o = []
    for f in ood:
        s = scores(m, tta, f)
        o.append(s)
        print(f"{os.path.basename(f):<34} {s['energy']:>8.2f} {s['calib_top']:>8.2f} "
              f"{s['tta_agree']:>8.2f} {s['entropy']:>8.2f}")

    print("\n### In-scope (real CT plants — SHOULD score in-dist) ###")
    print(hdr)
    ins = []
    for f in inscope:
        s = scores(m, tta, f)
        ins.append(s)
        label = os.path.basename(os.path.dirname(f))[:20] + "/" + os.path.basename(f)[:12]
        print(f"{label:<34} {s['energy']:>8.2f} {s['calib_top']:>8.2f} "
              f"{s['tta_agree']:>8.2f} {s['entropy']:>8.2f}")

    def mean(rows, k):
        return sum(r[k] for r in rows) / len(rows)
    print("\n### separation (OOD mean  vs  in-scope mean) ###")
    for k in ["energy", "calib_top", "tta_agree", "entropy"]:
        print(f"  {k:<10} OOD {mean(o,k):>7.2f}   in-scope {mean(ins,k):>7.2f}")


if __name__ == "__main__":
    main()
