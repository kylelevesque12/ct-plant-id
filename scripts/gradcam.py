"""Grad-CAM: which parts of a photo drove the model's prediction?

Why this exists (not a tutorial reproduction — each figure answers a measured
question from this project):

  * Trees score 66.5% top-1 versus shrubs at 84.6% (reports/stratified_eval.json).
    If the model is fixating on leaves and ignoring bark and canopy, Grad-CAM
    shows it directly — and that is the justification for the round-2
    multi-photo pull.
  * The hydrangea that motivated the whole ornamental expansion can be run
    through the OLD and NEW checkpoints side by side: "here is what it looked at
    when it said Black Swallow-Wort, and here is what it looks at now."

How it works, briefly: the last convolutional feature map holds spatial
information (a grid of positions) and semantic information (a channel per
learned feature). Grad-CAM takes the gradient of a class score with respect to
that feature map, averages each channel's gradient over space to get an
"importance" weight for that channel, then sums the channels by those weights.
Positive regions are the ones that pushed the score up. ReLU keeps only
supporting evidence, not evidence against.

Run:
  python scripts/gradcam.py --images data/ood_probe/*.jpg --out reports/figures
  python scripts/gradcam.py --images x.jpg --model runs/b_stage2/model.pt \
      --compare runs/stage2/model.pt          # old vs new, side by side
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from ctplantid.predict import PlantModel  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _target_module(model: PlantModel, layer: str):
    """Resolve which layer to attribute at.

    'head'  — output of forward_features (after conv_head, 1280ch @ 12x12)
    'block' — output of the last MBConv block (256ch @ 12x12), the conventional
              "last spatial conv" choice for EfficientNet
    'blockN'— an earlier block, e.g. 'block4', which is spatially FINER and so
              gives a less blocky map at the cost of being less class-specific
    """
    blocks = model.model.blocks
    if layer == "head":
        return None                      # handled via forward_features
    if layer == "block":
        return blocks[-1]
    if layer.startswith("block"):
        return blocks[int(layer[5:])]
    raise SystemExit(f"unknown --layer {layer}")


def gradcam(model: PlantModel, pil_image, class_idx=None, layer="block"):
    """Return (cam [H,W] normalised 0..1, predicted class index, probability).

    Grad-CAM: take the gradient of the class score w.r.t. a convolutional
    feature map, average each channel's gradient over space to get that
    channel's importance, then sum the channels weighted by importance. ReLU
    keeps only evidence that pushed the score UP.
    """
    x = model._tensor(pil_image)
    model.model.zero_grad(set_to_none=True)
    target = _target_module(model, layer)

    store = {}
    if target is None:
        feats = model.model.forward_features(x)
        feats.retain_grad()
        logits = model.model.forward_head(feats)[0]
        store["acts"] = feats
    else:
        def hook(_m, _inp, out):
            out.retain_grad()
            store["acts"] = out
        h = target.register_forward_hook(hook)
        logits = model.model(x)[0]
        h.remove()

    with torch.no_grad():
        probs = (logits.detach() / model.temperature).softmax(0)
    if class_idx is None:
        class_idx = int(probs.argmax())
    logits[class_idx].backward()

    acts = store["acts"]
    grads = acts.grad[0]                            # [C, H, W]
    weights = grads.mean(dim=(1, 2), keepdim=True)  # channel importance
    cam = (weights * acts.detach()[0]).sum(0).relu()
    cam = cam - cam.min()
    if float(cam.max()) > 0:
        cam = cam / cam.max()
    return cam.cpu().numpy(), class_idx, float(probs[class_idx])


def overlay(ax, pil_image, cam, title):
    img = ImageOps.exif_transpose(pil_image).convert("RGB")
    side = min(img.size)
    img = img.crop(((img.width - side) // 2, (img.height - side) // 2,
                    (img.width + side) // 2, (img.height + side) // 2)).resize((384, 384))
    heat = torch.tensor(cam)[None, None]
    heat = F.interpolate(heat, size=(384, 384), mode="bilinear", align_corners=False)
    ax.imshow(np.asarray(img))
    ax.imshow(heat[0, 0].numpy(), cmap="jet", alpha=0.45)
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def label_for(model, idx, prob):
    name = model.classes[idx]
    return f"{name}\n{prob:.0%}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--model", default=os.path.join(ROOT, "runs", "b_stage2", "model.pt"))
    ap.add_argument("--compare", default=None,
                    help="second checkpoint to show side by side (e.g. the old model)")
    ap.add_argument("--out", default=os.path.join(ROOT, "reports", "figures"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--layer", default="block4",
                    help="head | block | blockN (earlier = finer, less class-specific)")
    args = ap.parse_args()

    paths = []
    for pattern in args.images:
        paths.extend(sorted(glob.glob(pattern)) or [pattern])
    paths = [p for p in paths if os.path.exists(p)]
    if not paths:
        raise SystemExit("no images matched")

    models = [("new", PlantModel(args.model, device=args.device))]
    if args.compare:
        models.append(("old", PlantModel(args.compare, device=args.device)))
    print(f"{len(paths)} images | {len(models)} model(s)")

    os.makedirs(args.out, exist_ok=True)
    ncols = 1 + len(models)          # original + one CAM per model
    fig, axes = plt.subplots(len(paths), ncols,
                             figsize=(3.1 * ncols, 3.3 * len(paths)), squeeze=False)

    for r, path in enumerate(paths):
        pil = Image.open(path)
        img = ImageOps.exif_transpose(pil).convert("RGB")
        side = min(img.size)
        sq = img.crop(((img.width - side) // 2, (img.height - side) // 2,
                       (img.width + side) // 2, (img.height + side) // 2)).resize((384, 384))
        axes[r][0].imshow(np.asarray(sq))
        axes[r][0].set_title(os.path.basename(path), fontsize=8)
        axes[r][0].axis("off")

        for c, (tag, m) in enumerate(models, start=1):
            cam, idx, prob = gradcam(m, pil, layer=args.layer)
            overlay(axes[r][c], pil, cam, f"[{tag}] {label_for(m, idx, prob)}")
            print(f"  {os.path.basename(path):<22} [{tag}] {m.classes[idx]} {prob:.0%}")

    fig.suptitle("Grad-CAM — regions driving the prediction", fontsize=11)
    fig.tight_layout()
    out = os.path.join(args.out, f"gradcam_{args.layer}.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nwrote {os.path.relpath(out, ROOT)}")


if __name__ == "__main__":
    main()
