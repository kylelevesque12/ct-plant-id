"""Does the model actually use bark and whole-tree views, or only leaves?

The question behind the round-2 rebuild. Trees score 66.5% top-1 versus shrubs
at 84.6% (reports/stratified_eval.json), and the hypothesis was that the v1
dataset held almost only leaf close-ups because the downloader kept `photos[0]`
and discarded the rest — while trees average ~3.4 photos per observation
precisely because people shoot leaf, THEN BARK, THEN whole tree.

How this tests it: the round-2 pull stores photos as `<observation>_<idx>.<ext>`,
where idx is position within the observation. Position 0 is nearly always the
"hero" shot (usually foliage); later positions skew toward bark, form and
habitat. So position is a usable proxy for shot type without hand-labelling
thousands of images.

For every position we measure three things:

  * **accuracy**   — is the top-1 still right?
  * **confidence** — calibrated probability of the top class
  * **attention entropy** — how spread out the Grad-CAM is, normalised to 0..1.
    Low means the model is locked onto a region; high means it is smeared across
    the frame with nothing to grab. If the model has no idea what to do with a
    bark photo, entropy should rise and accuracy should fall together.

Run on the box holding the images:
  python scripts/tree_attention.py --data /root/round2 --max-species 25
"""
import argparse
import csv
import os
import re
import sys
from collections import defaultdict

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from ctplantid.predict import PlantModel  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch.nn.functional as F  # noqa: E402

sys.path.insert(0, os.path.join(ROOT, "scripts"))
from gradcam import gradcam  # noqa: E402

TREE_GENERA = {
    "Quercus", "Acer", "Betula", "Fagus", "Fraxinus", "Carya", "Pinus", "Tsuga",
    "Picea", "Platanus", "Tilia", "Ulmus", "Populus", "Liriodendron", "Abies",
    "Liquidambar", "Nyssa", "Juglans", "Robinia", "Gleditsia", "Ginkgo", "Larix",
    "Zelkova", "Castanea", "Catalpa", "Ailanthus", "Sassafras", "Metasequoia",
    "Cryptomeria", "Chamaecyparis", "Thuja", "Juniperus", "Magnolia", "Cercis",
}

_IDX = re.compile(r"_(\d+)\.[A-Za-z]+$")


def photo_index(path):
    m = _IDX.search(os.path.basename(path))
    return int(m.group(1)) if m else 0


def observation_of(path):
    base = os.path.basename(path)
    return _IDX.sub("", base)


def resolve(path, data_dir):
    if os.path.isabs(path):
        return path
    if path.startswith("data" + os.sep):
        return os.path.join(data_dir, os.path.relpath(path, "data"))
    return os.path.join(data_dir, path)


def attention_entropy(cam):
    """Normalised entropy of the CAM treated as a distribution over locations.

    0 = all attention on one cell, 1 = perfectly uniform (model has nothing to
    lock onto). Comparable across images because it is divided by log(n_cells).
    """
    p = cam.astype(np.float64).ravel()
    p = np.clip(p, 0, None)
    if p.sum() <= 0:
        return 1.0
    p = p / p.sum()
    nz = p[p > 0]
    return float(-(nz * np.log(nz)).sum() / np.log(len(p)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(ROOT, "runs", "b_stage2", "model.pt"))
    ap.add_argument("--data", default=os.path.join(ROOT, "data"))
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--max-species", type=int, default=25)
    ap.add_argument("--max-obs-per-species", type=int, default=6)
    ap.add_argument("--layer", default="block4")
    ap.add_argument("--out", default=os.path.join(ROOT, "reports", "figures"))
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    manifest = args.manifest or os.path.join(args.data, "manifest.csv")
    by_species_obs = defaultdict(lambda: defaultdict(list))
    with open(manifest, newline="") as f:
        for r in csv.DictReader(f):
            if r.get("split") != args.split:
                continue
            if r["species"].split()[0] not in TREE_GENERA:
                continue
            by_species_obs[r["species"]][observation_of(r["path"])].append(r["path"])

    # Only observations with 2+ photos can tell us anything about shot type.
    species = sorted(by_species_obs, key=lambda s: -sum(
        1 for v in by_species_obs[s].values() if len(v) > 1))[:args.max_species]
    if not species:
        raise SystemExit("no multi-photo tree observations found in that split")

    model = PlantModel(args.model, device=args.device)
    print(f"device {model.device} | {len(species)} tree species")

    recs = []
    examples = []
    for sp in species:
        multi = [(o, sorted(p, key=photo_index))
                 for o, p in by_species_obs[sp].items() if len(p) > 1]
        for obs, paths in multi[:args.max_obs_per_species]:
            for p in paths:
                full = resolve(p, args.data)
                try:
                    pil = Image.open(full)
                    cam, idx, prob = gradcam(model, pil, layer=args.layer)
                except Exception:
                    continue
                recs.append({
                    "species": sp, "pos": photo_index(p),
                    "correct": int(model.classes[idx] == sp),
                    "genus_ok": int(model.classes[idx].split()[0] == sp.split()[0]),
                    "conf": prob, "entropy": attention_entropy(cam),
                })
                if len(examples) < 4 and photo_index(p) == 0 and len(paths) >= 3:
                    examples.append((sp, paths, obs))
        if len(recs) and len(recs) % 200 < 6:
            print(f"  {len(recs)} photos scored…")

    if not recs:
        raise SystemExit("nothing scored")

    print(f"\n=== attention and accuracy by photo position (n={len(recs):,}) ===")
    print(f"{'position':>9} {'n':>7} {'top-1':>8} {'genus':>8} {'conf':>8} {'attn entropy':>14}")
    rows = []
    for pos in sorted({r["pos"] for r in recs}):
        g = [r for r in recs if r["pos"] == pos]
        row = {
            "pos": pos, "n": len(g),
            "top1": sum(r["correct"] for r in g) / len(g),
            "genus": sum(r["genus_ok"] for r in g) / len(g),
            "conf": sum(r["conf"] for r in g) / len(g),
            "entropy": sum(r["entropy"] for r in g) / len(g),
        }
        rows.append(row)
        print(f"{pos:>9} {row['n']:>7} {row['top1']:>7.1%} {row['genus']:>7.1%} "
              f"{row['conf']:>7.1%} {row['entropy']:>13.3f}")

    print("\nposition 0 is nearly always the foliage 'hero' shot; higher positions")
    print("skew to bark, whole tree and habitat. If accuracy falls and attention")
    print("entropy rises together, the model has little to grab in those views.")

    # Figure: one multi-photo observation per row, Grad-CAM across positions.
    if examples:
        ncols = max(len(p) for _, p, _ in examples)
        fig, axes = plt.subplots(len(examples), ncols,
                                 figsize=(3.0 * ncols, 3.2 * len(examples)),
                                 squeeze=False)
        for r, (sp, paths, obs) in enumerate(examples):
            for c in range(ncols):
                ax = axes[r][c]
                ax.axis("off")
                if c >= len(paths):
                    continue
                pil = Image.open(resolve(paths[c], args.data))
                cam, idx, prob = gradcam(model, pil, layer=args.layer)
                img = ImageOps.exif_transpose(pil).convert("RGB")
                side = min(img.size)
                img = img.crop(((img.width - side) // 2, (img.height - side) // 2,
                                (img.width + side) // 2,
                                (img.height + side) // 2)).resize((384, 384))
                heat = F.interpolate(torch.tensor(cam)[None, None], size=(384, 384),
                                     mode="bilinear", align_corners=False)[0, 0]
                ax.imshow(np.asarray(img))
                ax.imshow(heat.numpy(), cmap="jet", alpha=0.45)
                ok = "OK" if model.classes[idx] == sp else "X"
                ax.set_title(f"pos {photo_index(paths[c])} · {ok} {prob:.0%}\n"
                             f"H={attention_entropy(cam):.2f}", fontsize=8)
            axes[r][0].set_ylabel(sp, fontsize=8)
        fig.suptitle("Grad-CAM across photo positions of the same tree "
                     "(pos 0 = foliage; later = bark / whole tree)", fontsize=11)
        fig.tight_layout()
        os.makedirs(args.out, exist_ok=True)
        out = os.path.join(args.out, "tree_attention.png")
        fig.savefig(out, dpi=130, bbox_inches="tight")
        print(f"\nwrote {out}")

    import json
    os.makedirs(os.path.join(ROOT, "reports"), exist_ok=True)
    json.dump({"by_position": rows, "n": len(recs), "species": species},
              open(os.path.join(ROOT, "reports", "tree_attention.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
