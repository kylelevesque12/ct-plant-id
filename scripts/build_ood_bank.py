"""Workstream A's real OOD, built in Workstream B: the Mahalanobis feature-distance
'bank'. Runs on the GPU box AFTER training (it needs the full training embeddings
+ a GPU). The bank-free probe (scripts/ood_probe.py) proved feature-distance is
the ONLY signal that tells 'garden hydrangea' (far from the manifold) from 'hard
in-scope plant like skunk cabbage' (near it) — this is that signal.

Method (Lee et al. 2018):
  - per-class mean mu_c of the penultimate embeddings, over a sample of TRAIN imgs
  - one shared (tied) covariance Sigma across classes (stable with thin tail data)
  - OOD score(x) = min_c (f-mu_c)^T Sigma^-1 (f-mu_c)  = distance to the nearest
    class in *whitened* feature space. Far from every class => out of scope.

Threshold: set by the in-scope FALSE-REJECT budget — reject at most --reject-pct
of real plants (measured on a held-out VAL sample), the asymmetric operating point
from docs/model_overhaul.md. We do NOT need labeled OOD to set it; we then just
REPORT where any data/ood_probe/* (the hydrangea) land relative to the threshold.

Serving stays cheap via whitening: precompute W (whitener) and nu_c = W·mu_c, so
score(f) = min_c || W·f - nu_c ||^2 is one matmul, not a 2,500-class Mahalanobis
loop per request.

Writes <model_dir>/ood_bank.npz  {classes, whiten W, whitened means nu, threshold}.
predict.py will load it next to temperature.json (integration is the follow-up
once this bank exists and can be tested against it).

Run on the box:
  python scripts/build_ood_bank.py --model runs/b_stage2/model.pt --data /workspace/data
"""
import argparse
import csv
import glob
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from ctplantid.predict import PlantModel  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402


def resolve(path, data_dir):
    """Mirror train.py's path resolution so paths written 'data/...' work whether
    the data dir is ./data or /workspace/data."""
    if os.path.isabs(path):
        return path
    if path.startswith("data" + os.sep):
        return os.path.join(data_dir, os.path.relpath(path, "data"))
    return os.path.join(data_dir, path)


class Rows(Dataset):
    def __init__(self, rows, data_dir, tf, cls_idx):
        self.rows, self.data_dir, self.tf, self.cls_idx = rows, data_dir, tf, cls_idx

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        try:
            img = ImageOps.exif_transpose(
                Image.open(resolve(r["path"], self.data_dir))).convert("RGB")
            return self.tf(img), self.cls_idx[r["species"]]
        except Exception:
            # Corrupt/missing file: emit a correctly-shaped tensor (tf resizes a
            # blank image) with label -1 so it's filtered out after extraction.
            return self.tf(Image.new("RGB", (64, 64))), -1


@torch.no_grad()
def extract(model, dl, dev):
    feats, labels = [], []
    for x, y in dl:
        x = x.to(dev, non_blocking=True)
        f = model.model.forward_head(model.model.forward_features(x), pre_logits=True)
        feats.append(f.float().cpu())
        labels.append(y)
    return torch.cat(feats), torch.cat(labels)


def sample_by_class(rows, per_class):
    by = {}
    for r in rows:
        by.setdefault(r["species"], []).append(r)
    out = []
    for sp, rs in by.items():
        rs.sort(key=lambda r: r["path"])          # deterministic
        out.extend(rs[:per_class])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(ROOT, "runs", "b_stage2", "model.pt"))
    ap.add_argument("--data", default=os.path.join(ROOT, "data"))
    ap.add_argument("--per-class", type=int, default=40, help="train imgs/class for the bank")
    ap.add_argument("--val-per-class", type=int, default=8, help="held-out imgs/class for threshold")
    ap.add_argument("--reject-pct", type=float, default=2.5, help="max %% of real plants we'll flag")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    m = PlantModel(args.model, device="cuda" if torch.cuda.is_available() else "cpu")
    dev = m.device
    cls_idx = {c: i for i, c in enumerate(m.classes)}
    rows = list(csv.DictReader(open(os.path.join(args.data, "manifest.csv"))))
    rows = [r for r in rows if r["species"] in cls_idx]
    train = sample_by_class([r for r in rows if r["split"] == "train"], args.per_class)
    val = sample_by_class([r for r in rows if r["split"] == "val"], args.val_per_class)
    print(f"device {dev} | {len(m.classes)} classes | "
          f"train sample {len(train)} | val sample {len(val)}")

    def loader(rs):
        return DataLoader(Rows(rs, args.data, m.tf, cls_idx), batch_size=args.batch,
                          num_workers=args.workers, pin_memory=(dev == "cuda"))

    print("extracting train embeddings…")
    F, y = extract(m, loader(train), dev)
    keep = y >= 0
    F, y = F[keep], y[keep]
    N, D = F.shape
    C = len(m.classes)

    # per-class means (only classes that actually have samples)
    means = torch.zeros(C, D)
    counts = torch.zeros(C)
    means.index_add_(0, y, F)
    counts.index_add_(0, y, torch.ones(N))
    pop = counts > 0
    means[pop] /= counts[pop].unsqueeze(1)
    print(f"  {int(pop.sum())}/{C} classes populated ({N} embeddings, dim {D})")

    # tied covariance from within-class-centered features, ridge-regularized
    centered = F - means[y]
    cov = (centered.T @ centered) / max(N - int(pop.sum()), 1)
    cov += (1e-3 * torch.diag(cov).mean()) * torch.eye(D)
    precision = torch.linalg.inv(cov)
    L = torch.linalg.cholesky(precision)          # precision = L L^T
    W = L.T.contiguous()                          # (f-mu)^T P (f-mu) = || W (f-mu) ||^2
    nu = (means[pop] @ W.T)                        # whitened means, populated classes only
    nu_sq = (nu * nu).sum(1)

    def scores(feat):                             # min whitened distance to any centroid
        g = feat @ W.T
        d2 = (g * g).sum(1, keepdim=True) - 2 * g @ nu.T + nu_sq.unsqueeze(0)
        return d2.min(1).values

    print("extracting val embeddings for the threshold…")
    Fv, yv = extract(m, loader(val), dev)
    Fv = Fv[yv >= 0]
    val_scores = scores(Fv)
    thr = float(torch.quantile(val_scores, 1 - args.reject_pct / 100))
    rejected = float((val_scores > thr).float().mean())
    print(f"\nthreshold {thr:.1f}  (rejects {rejected:.1%} of held-out real plants; "
          f"budget {args.reject_pct:.1f}%)")

    # Report where the OOD probes (the hydrangea etc.) fall — sanity, not tuning.
    probes = sorted(glob.glob(os.path.join(args.data, "ood_probe", "*")))
    if probes:
        print("\nOOD probe scores (want these ABOVE the threshold):")
        for p in probes:
            f = m.features(Image.open(p)).unsqueeze(0).float()
            s = float(scores(f)[0])
            print(f"  {os.path.basename(p):<28} score {s:8.1f}  "
                  f"{'OOD ✓ (flagged)' if s > thr else 'in-scope ✗ (missed)'}")

    out = os.path.join(os.path.dirname(os.path.abspath(args.model)), "ood_bank.npz")
    np.savez(out, classes=np.array([c for c, k in zip(m.classes, pop.tolist()) if k]),
             whiten=W.numpy(), nu=nu.numpy(), threshold=thr,
             reject_pct=args.reject_pct, dim=D)
    print(f"\nwrote {os.path.relpath(out, ROOT)}  (predict.py loads it beside temperature.json)")


if __name__ == "__main__":
    main()
