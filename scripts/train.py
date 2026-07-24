"""Scalable trainer — reads a pre-downloaded dataset, fine-tunes a backbone.

Unlike the Kaggle scripts (which download their own small subset inline),
this consumes the dataset built by download_opendata.py on the cloud box:
a data/ dir with manifest.csv (observation_id, taxon_id, species, path,
license, split) and images on disk. Designed for the real scaling run AND
cheap proxy sweeps (--limit-species) to tune before spending the big compute.

Embeds the methodology:
  - configurable backbone (default an iNaturalist/21k-pretrained model)
  - correct backbone-matched normalization + RandAugment
  - DISCRIMINATIVE learning rates: low LR for the pretrained backbone, higher
    for the fresh head; optional --freeze-epochs to warm up the head first
  - label smoothing, weight decay, cosine schedule, class-balanced sampling
  - validation-driven early stopping + best-checkpoint selection; test untouched
  - every run logs its config + per-epoch metrics to <out>/run.json so
    proxy sweeps are comparable

Examples:
  # cheap proxy sweep (minutes): 150 species, short
  python scripts/train.py --data data --limit-species 150 --epochs 6 --lr 3e-4
  # the full run
  python scripts/train.py --data data --backbone convnext_base.fb_in22k_ft_in1k_384 \
      --img-size 384 --epochs 30
"""
import argparse
import csv
import json
import os
import random
import time
from collections import Counter, defaultdict

import timm
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import torchvision.transforms as T

DEFAULT_BACKBONE = "tf_efficientnetv2_s.in21k_ft_in1k"  # 21k-pretrained; swap freely


def genus_of(name):
    return name.strip().lower().split(" ")[0]


def load_manifest(data_dir, limit_species):
    rows = list(csv.DictReader(open(os.path.join(data_dir, "manifest.csv"))))
    if limit_species:
        # proxy runs: keep the N species with the most images (the head)
        counts = Counter(r["species"] for r in rows)
        keep = {sp for sp, _ in counts.most_common(limit_species)}
        rows = [r for r in rows if r["species"] in keep]
    classes = sorted({r["species"] for r in rows})
    idx = {c: i for i, c in enumerate(classes)}
    for r in rows:
        r["label"] = idx[r["species"]]
    return rows, classes


class PlantSet(Dataset):
    def __init__(self, rows, data_dir, tf):
        self.rows, self.data_dir, self.tf = rows, data_dir, tf

    def __len__(self):
        return len(self.rows)

    def _path(self, r):
        path = r["path"]
        if os.path.isabs(path):
            return path
        if path.startswith("data" + os.sep):
            return os.path.join(self.data_dir, os.path.relpath(path, "data"))
        return os.path.join(self.data_dir, path)

    def __getitem__(self, i):
        # A handful of images are truncated (the download died on a full disk).
        # verify_images.py removes them up front; this is a backstop so one bad
        # file can't kill a multi-hour run — substitute another sample instead.
        for _ in range(10):
            r = self.rows[i]
            try:
                img = Image.open(self._path(r)).convert("RGB")
                return self.tf(img), r["label"]
            except Exception:
                i = random.randrange(len(self.rows))
        raise RuntimeError("too many unreadable images — run verify_images.py")


def balanced_sampler(rows, n_classes):
    counts = [0] * n_classes
    for r in rows:
        counts[r["label"]] += 1
    w = [1.0 / max(counts[r["label"]], 1) for r in rows]
    return WeightedRandomSampler(w, num_samples=len(rows), replacement=True)


@torch.no_grad()
def evaluate(model, dl, dev, classes):
    model.eval()
    t1 = t5 = g1 = n = 0
    k = min(5, len(classes))
    for x, y in dl:
        top = model(x.to(dev)).topk(k, dim=1).indices.cpu()
        for i, yi in enumerate(y):
            n += 1
            t1 += int(top[i][0] == yi)
            t5 += int(yi in top[i])
            g1 += int(genus_of(classes[top[i][0]]) == genus_of(classes[yi]))
    if not n:
        return None
    return {"n": n, "top1": t1 / n, "top5": t5 / n, "genus": g1 / n}


def param_groups(model, backbone_lr, head_lr):
    """Discriminative LRs: head params at head_lr, everything else at backbone_lr."""
    head_ids = {id(p) for p in model.get_classifier().parameters()}
    backbone = [p for p in model.parameters() if id(p) not in head_ids and p.requires_grad]
    head = [p for p in model.parameters() if id(p) in head_ids]
    return [{"params": backbone, "lr": backbone_lr},
            {"params": head, "lr": head_lr}]


def run(args):
    dev = "cuda" if torch.cuda.is_available() else \
          "mps" if torch.backends.mps.is_available() else "cpu"
    use_amp = dev == "cuda"
    os.makedirs(args.out, exist_ok=True)

    rows, classes = load_manifest(args.data, args.limit_species)
    tr = [r for r in rows if r["split"] == "train"]
    va = [r for r in rows if r["split"] == "val"]
    ts = [r for r in rows if r["split"] == "test"]
    print(f"device {dev} | {len(rows)} images, {len(classes)} classes | "
          f"train {len(tr)} val {len(va)} test {len(ts)}")

    model = timm.create_model(args.backbone, pretrained=True,
                              num_classes=len(classes)).to(dev)
    # Stage 2 of progressive resizing: start from a lower-res run's weights
    # (weights only — fresh optimizer/schedule for the new resolution).
    if args.init_from:
        ck = torch.load(args.init_from, map_location=dev, weights_only=False)
        # Load only shape-matching params. Same-class runs load everything; when
        # the class set changed (Workstream B added ornamentals) the classifier
        # head shape differs, so those params are skipped and stay freshly
        # initialized — warm-starting the CT-tuned BACKBONE, fresh head.
        msd = model.state_dict()
        compat = {k: v for k, v in ck["state_dict"].items()
                  if k in msd and v.shape == msd[k].shape}
        model.load_state_dict(compat, strict=False)
        reinit = len(msd) - len(compat)
        print(f"initialized from {args.init_from} "
              f"(trained at {ck['data_config']['input_size'][1]}px); "
              f"loaded {len(compat)}/{len(msd)} params"
              + (f", {reinit} reinit for new class set" if reinit else ""))
    cfg = timm.data.resolve_model_data_config(model)
    cfg["input_size"] = (3, args.img_size, args.img_size)
    train_tf = timm.data.create_transform(**cfg, is_training=True,
                                          auto_augment="rand-m7-mstd0.5")
    # Robustness augs for the phone-vs-iNat gap (Workstream B): RandAugment already
    # covers color/rotate/shear but NOT defocus blur or perspective — the exact
    # variance that flipped the hydrangea across three angles. Prepend them at the
    # PIL level (before timm's crop/normalize). Disable with --no-robust-aug.
    if not args.no_robust_aug:
        train_tf = T.Compose([
            T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.30),
            T.RandomPerspective(distortion_scale=0.20, p=0.30),
            train_tf,
        ])
    eval_tf = timm.data.create_transform(**cfg, is_training=False)

    train_dl = DataLoader(PlantSet(tr, args.data, train_tf), batch_size=args.batch,
                          sampler=balanced_sampler(tr, len(classes)),
                          num_workers=args.workers, pin_memory=(dev == "cuda"))
    val_dl = DataLoader(PlantSet(va, args.data, eval_tf), batch_size=args.batch,
                        num_workers=args.workers) if va else None
    test_dl = DataLoader(PlantSet(ts, args.data, eval_tf), batch_size=args.batch,
                         num_workers=args.workers) if ts else None

    opt = torch.optim.AdamW(param_groups(model, args.lr, args.head_lr),
                            weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # optional: freeze the backbone for the first --freeze-epochs (warm up head)
    def set_backbone_frozen(frozen):
        head_ids = {id(p) for p in model.get_classifier().parameters()}
        for p in model.parameters():
            if id(p) not in head_ids:
                p.requires_grad = not frozen

    # --- benchmark: measure before committing to an expensive run ---
    if args.benchmark:
        model.train()
        warmup, seen, t0 = 3, 0, None
        for i, (x, y) in enumerate(train_dl):
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            with torch.autocast(device_type="cuda", enabled=use_amp):
                loss = loss_fn(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            if i + 1 == warmup:          # start timing after warmup batches
                t0, seen = time.time(), 0
            elif t0 is not None:
                seen += len(x)
            if i + 1 >= args.benchmark + warmup:
                break
        dt = time.time() - t0
        ips = seen / dt
        epoch_s = len(tr) / ips
        total_h = epoch_s * args.epochs / 3600
        print(f"\n=== benchmark ({args.backbone} @ {args.img_size}px) ===")
        print(f"measured: {ips:,.0f} images/sec")
        print(f"1 epoch over {len(tr):,} images: {epoch_s/60:.1f} min")
        print(f"{args.epochs} epochs: {total_h:.1f} hours"
              f"  ≈ ${total_h * args.gpu_cost:.2f} at ${args.gpu_cost:.2f}/hr")
        print("(no training saved — this was a measurement run)")
        return

    log = {"config": vars(args), "backbone": args.backbone,
           "n_classes": len(classes), "epochs": []}
    best_val, best_state, since = -1.0, None, 0
    start_ep = 0

    # --- resume: pick up from the last completed epoch ---
    last_path = os.path.join(args.out, "last.pt")
    if args.resume and os.path.exists(last_path):
        ck = torch.load(last_path, map_location=dev, weights_only=False)
        model.load_state_dict(ck["state_dict"])
        opt.load_state_dict(ck["optimizer"])
        start_ep, best_val = ck["epoch"] + 1, ck.get("best_val", -1.0)
        for _ in range(start_ep):
            sched.step()
        print(f"resumed from epoch {start_ep} (best val so far {best_val:.1%})")

    for ep in range(start_ep, args.epochs):
        set_backbone_frozen(ep < args.freeze_epochs)
        model.train()
        tot = 0.0
        for x, y in train_dl:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            with torch.autocast(device_type="cuda", enabled=use_amp):
                loss = loss_fn(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += loss.item() * len(x)
        sched.step()
        train_loss = tot / max(len(tr), 1)

        rec = {"epoch": ep + 1, "train_loss": round(train_loss, 4)}
        msg = f"epoch {ep + 1}/{args.epochs}  loss {train_loss:.3f}"
        if val_dl:
            v = evaluate(model, val_dl, dev, classes)
            rec["val"] = {k: round(v[k], 4) for k in ("top1", "top5", "genus")}
            msg += f"  val top-1 {v['top1']:.1%} top-5 {v['top5']:.1%}"
            if v["top1"] > best_val:
                best_val, since = v["top1"], 0
                best_state = {k: t.detach().cpu().clone()
                              for k, t in model.state_dict().items()}
            else:
                since += 1
        log["epochs"].append(rec)
        print(msg)
        json.dump(log, open(os.path.join(args.out, "run.json"), "w"), indent=2)
        # Checkpoint EVERY epoch so a crash costs one epoch, not the whole run.
        torch.save({"state_dict": model.state_dict(), "optimizer": opt.state_dict(),
                    "epoch": ep, "best_val": best_val, "classes": classes,
                    "data_config": cfg, "backbone": args.backbone}, last_path)
        if val_dl and since >= args.patience:
            print(f"early stop: no val improvement for {args.patience} epochs")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    final_dl = test_dl or val_dl
    if final_dl is not None:
        r = evaluate(model, final_dl, dev, classes)
        log["final_test"] = {k: round(r[k], 4) for k in ("top1", "top5", "genus")}
        print(f"\nfinal held-out  top-1 {r['top1']:.1%}  top-5 {r['top5']:.1%}  "
              f"genus {r['genus']:.1%}  (n={r['n']})")

    torch.save({"state_dict": model.state_dict(), "classes": classes,
                "data_config": cfg, "backbone": args.backbone},
               os.path.join(args.out, "model.pt"))
    json.dump(log, open(os.path.join(args.out, "run.json"), "w"), indent=2)
    print(f"saved {args.out}/model.pt and run.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data", help="dir with manifest.csv + images")
    ap.add_argument("--out", default="runs/run1")
    ap.add_argument("--backbone", default=DEFAULT_BACKBONE)
    ap.add_argument("--img-size", type=int, default=384)
    ap.add_argument("--limit-species", type=int, default=0, help="proxy sweep: top-N")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4, help="backbone LR")
    ap.add_argument("--head-lr", type=float, default=2e-3, help="classifier head LR")
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--label-smoothing", type=float, default=0.05,
                    help="0.1 caused underconfidence (we fixed it post-hoc via "
                         "temperature scaling); 0.05 is gentler. Re-fit temp after.")
    ap.add_argument("--no-robust-aug", action="store_true",
                    help="disable the blur/perspective robustness augmentations")
    ap.add_argument("--freeze-epochs", type=int, default=1, help="warm up head first")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--benchmark", type=int, default=0,
                    help="time N batches, project hours+cost, then exit")
    ap.add_argument("--gpu-cost", type=float, default=1.50,
                    help="$/hr, for the benchmark's cost projection")
    ap.add_argument("--resume", action="store_true",
                    help="continue from <out>/last.pt")
    ap.add_argument("--init-from", default="",
                    help="start from another run's weights (stage-2 fine-tune)")
    run(ap.parse_known_args()[0])
