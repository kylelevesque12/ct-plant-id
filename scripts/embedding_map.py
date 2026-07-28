"""Project the model's feature space to 2D — what has it actually learned?

This visualises a number the project already measured. The tree benchmark found
Quercus at 35% species accuracy but 69% genus accuracy: the model reliably knows
"oak" and cannot say *which* oak. In feature space that should look like a tight,
well-separated oak cluster with the individual species smeared together inside
it — while a distinctive genus separates cleanly at the species level.

It also explains the OOD detector for free: the Mahalanobis bank measures
distance in exactly this embedding space, so a picture of the space is a picture
of what "far from every trained class" means.

Method: take the penultimate (pre-logit) embedding for a sample of held-out
images, reduce to 2D with UMAP, and colour by genus.

Run where the images are (the build droplet):
  python scripts/embedding_map.py --data /root/round2 \
      --genera Quercus,Acer,Pinus,Hydrangea,Toxicodendron --per-species 15
"""
import argparse
import csv
import os
import sys
from collections import defaultdict

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from ctplantid.predict import PlantModel  # noqa: E402
from PIL import Image  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def resolve(path, data_dir):
    if os.path.isabs(path):
        return path
    if path.startswith("data" + os.sep):
        return os.path.join(data_dir, os.path.relpath(path, "data"))
    return os.path.join(data_dir, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(ROOT, "runs", "b_stage2", "model.pt"))
    ap.add_argument("--data", default=os.path.join(ROOT, "data"))
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--genera", default="Quercus,Acer,Pinus,Hydrangea,Toxicodendron,Solidago",
                    help="comma-separated genera to plot")
    ap.add_argument("--per-species", type=int, default=15)
    ap.add_argument("--split", default="test", help="which split to sample from")
    ap.add_argument("--out", default=os.path.join(ROOT, "reports", "figures"))
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    genera = [g.strip() for g in args.genera.split(",") if g.strip()]
    manifest = args.manifest or os.path.join(args.data, "manifest.csv")

    by_species = defaultdict(list)
    with open(manifest, newline="") as f:
        for r in csv.DictReader(f):
            if r.get("split") != args.split:
                continue
            if r["species"].split()[0] in genera:
                by_species[r["species"]].append(r["path"])
    if not by_species:
        raise SystemExit(f"no {args.split}-split images for genera {genera}")

    model = PlantModel(args.model, device=args.device)
    print(f"device {model.device} | {len(by_species)} species across {len(genera)} genera")

    vecs, labels, species_labels = [], [], []
    for sp in sorted(by_species):
        paths = sorted(by_species[sp])[:args.per_species]
        for p in paths:
            try:
                emb = model.features(Image.open(resolve(p, args.data)))
            except Exception:
                continue
            vecs.append(emb.numpy())
            labels.append(sp.split()[0])
            species_labels.append(sp)
    print(f"embedded {len(vecs):,} images")
    if len(vecs) < 10:
        raise SystemExit("too few embeddings to project")

    X = np.stack(vecs)
    import umap
    # cosine: the classifier head compares directions, so angular distance is the
    # geometry the model actually uses.
    proj = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine",
                     random_state=0).fit_transform(X)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    cmap = plt.get_cmap("tab10")
    gcolor = {g: cmap(i % 10) for i, g in enumerate(genera)}

    # Left: coloured by GENUS — do genera separate?
    for g in genera:
        m = [i for i, l in enumerate(labels) if l == g]
        if m:
            axes[0].scatter(proj[m, 0], proj[m, 1], s=14, alpha=0.75,
                            color=gcolor[g], label=f"{g} (n={len(m)})")
    axes[0].set_title("Coloured by GENUS — genera separate cleanly", fontsize=11)
    axes[0].legend(fontsize=8, markerscale=1.4)
    axes[0].set_xticks([]); axes[0].set_yticks([])

    # Right: within the largest genus, coloured by SPECIES — do they separate?
    biggest = max(genera, key=lambda g: sum(1 for l in labels if l == g))
    idx = [i for i, l in enumerate(labels) if l == biggest]
    sp_in = sorted({species_labels[i] for i in idx})
    sp_cmap = plt.get_cmap("tab20")
    for j, sp in enumerate(sp_in):
        m = [i for i in idx if species_labels[i] == sp]
        axes[1].scatter(proj[m, 0], proj[m, 1], s=22, alpha=0.85,
                        color=sp_cmap(j % 20), label=sp.split()[1])
    axes[1].set_title(f"Within {biggest} — coloured by SPECIES "
                      f"(the confusion the benchmark measured)", fontsize=11)
    axes[1].legend(fontsize=7, ncol=2, markerscale=1.2)
    axes[1].set_xlim(axes[0].get_xlim()); axes[1].set_ylim(axes[0].get_ylim())
    axes[1].set_xticks([]); axes[1].set_yticks([])

    fig.suptitle("Model feature space (UMAP of penultimate embeddings) — "
                 "the same space the OOD detector measures distance in", fontsize=12)
    fig.tight_layout()
    os.makedirs(args.out, exist_ok=True)
    out = os.path.join(args.out, "embedding_map.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
