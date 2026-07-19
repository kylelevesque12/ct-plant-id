"""Improved Kaggle trainer for CT plant species — supersedes kaggle_train.py.

Same paste-into-one-cell workflow (GPU + Internet on, then Run), but with the
things a real run needs:

  - CORRECT input normalization from the backbone's own data config (the v1
    script skipped this — timm models expect their pretrained mean/std)
  - strong augmentation (RandAugment) to fight the overfitting we saw
  - label smoothing + weight decay + cosine LR schedule
  - a real train/val/TEST split: val drives early stopping and checkpoint
    selection; test is untouched until the final number, so it stays honest
  - class-balanced sampling (matters once mid/tail species come in)
  - mixed precision on GPU (faster on Kaggle's T4)

Reports val top-1 each epoch and final held-out (test) top-1 / top-5 / genus.

Scale it by editing the defaults at the bottom (or CLI flags locally):
    python scripts/kaggle_train_v2.py --species 3 --per-species 4 --epochs 2
"""
import argparse
import hashlib
import io
import subprocess
import sys
import time

import requests

try:
    import timm
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "timm"], check=True)
    import timm

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

API = "https://api.inaturalist.org/v1"
CT_PLACE_ID = 49
PLANTAE = 47126
HEADERS = {"User-Agent": "ct-plant-id/0.1 (learning project)"}
BACKBONE = "tf_efficientnetv2_s"


# ---------- observation-keyed split (no leakage) ----------

def split_for(obs_id, val=0.15, test=0.15):
    h = int(hashlib.md5(str(obs_id).encode()).hexdigest(), 16) % 10000 / 10000
    return "test" if h < test else "val" if h < test + val else "train"


def genus_of(name):
    return name.strip().lower().split(" ")[0]


# ---------- pull a subset from iNaturalist ----------

def top_species(n, min_obs):
    out, page = [], 1
    while len(out) < n:
        r = requests.get(f"{API}/observations/species_counts",
                         params={"place_id": CT_PLACE_ID, "taxon_id": PLANTAE,
                                 "quality_grade": "research", "per_page": 200,
                                 "page": page}, headers=HEADERS, timeout=60)
        r.raise_for_status()
        results = r.json()["results"]
        if not results:
            break
        for row in results:
            t = row["taxon"]
            if t["rank"] == "species" and row["count"] >= min_obs:
                out.append({"taxon_id": t["id"], "name": t["name"]})
        page += 1
        time.sleep(1)
    return out[:n]


def observations(taxon_id, want):
    out, page = [], 1
    while len(out) < want:
        r = requests.get(f"{API}/observations",
                         params={"taxon_id": taxon_id, "place_id": CT_PLACE_ID,
                                 "quality_grade": "research", "photos": "true",
                                 "order_by": "votes", "per_page": 50, "page": page},
                         headers=HEADERS, timeout=60)
        r.raise_for_status()
        res = r.json()["results"]
        if not res:
            break
        out.extend(res)
        page += 1
        time.sleep(1)
    return out[:want]


def build_subset(n_species, per_species, min_obs):
    species = top_species(n_species, min_obs)
    classes = [s["name"] for s in species]
    records = []
    for label, s in enumerate(species):
        got = 0
        for obs in observations(s["taxon_id"], per_species * 2):
            if got >= per_species:
                break
            photos = obs.get("photos") or []
            if not photos or not photos[0].get("url"):
                continue
            url = photos[0]["url"].replace("/square.", "/medium.")
            try:
                b = requests.get(url, headers=HEADERS, timeout=60).content
                img = Image.open(io.BytesIO(b)).convert("RGB")
            except Exception:
                continue
            records.append({"img": img, "label": label,
                            "split": split_for(obs["id"])})
            got += 1
            time.sleep(0.3)
        print(f"  [{label + 1}/{len(species)}] {s['name']}: {got} photos")
    return records, classes


# ---------- dataset ----------

class PlantSet(Dataset):
    def __init__(self, records, tf):
        self.records, self.tf = records, tf

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        return self.tf(r["img"]), r["label"]


def balanced_sampler(records, n_classes):
    counts = [0] * n_classes
    for r in records:
        counts[r["label"]] += 1
    weights = [1.0 / max(counts[r["label"]], 1) for r in records]
    return WeightedRandomSampler(weights, num_samples=len(records), replacement=True)


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


# ---------- train ----------

def run(args):
    dev = ("cuda" if torch.cuda.is_available()
           else "mps" if torch.backends.mps.is_available() else "cpu")
    use_amp = dev == "cuda"
    print(f"device: {dev} (amp={use_amp})")

    print("downloading subset from iNaturalist…")
    records, classes = build_subset(args.species, args.per_species, args.min_obs)
    tr = [r for r in records if r["split"] == "train"]
    va = [r for r in records if r["split"] == "val"]
    ts = [r for r in records if r["split"] == "test"]
    print(f"{len(records)} images, {len(classes)} classes | "
          f"train {len(tr)} / val {len(va)} / test {len(ts)}")

    model = timm.create_model(BACKBONE, pretrained=True,
                              num_classes=len(classes)).to(dev)

    # Correct, backbone-matched transforms (this is the normalization fix).
    cfg = timm.data.resolve_model_data_config(model)
    cfg["input_size"] = (3, args.img_size, args.img_size)
    train_tf = timm.data.create_transform(**cfg, is_training=True,
                                          auto_augment="rand-m7-mstd0.5")
    eval_tf = timm.data.create_transform(**cfg, is_training=False)

    train_dl = DataLoader(PlantSet(tr, train_tf), batch_size=args.batch,
                          sampler=balanced_sampler(tr, len(classes)), num_workers=2)
    val_dl = (DataLoader(PlantSet(va, eval_tf), batch_size=args.batch, num_workers=2)
              if va else None)
    test_dl = (DataLoader(PlantSet(ts, eval_tf), batch_size=args.batch, num_workers=2)
               if ts else None)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val, best_state, since_improved = -1.0, None, 0
    for ep in range(args.epochs):
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

        msg = f"epoch {ep + 1}/{args.epochs}  train loss {tot / max(len(tr), 1):.3f}"
        if val_dl:
            v = evaluate(model, val_dl, dev, classes)
            msg += f"  val top-1 {v['top1']:.1%}"
            if v["top1"] > best_val:
                best_val, since_improved = v["top1"], 0
                best_state = {k: t.detach().cpu().clone()
                              for k, t in model.state_dict().items()}
            else:
                since_improved += 1
        print(msg)
        if val_dl and since_improved >= args.patience:
            print(f"early stop: no val improvement for {args.patience} epochs")
            break

    # Restore the best (by val) weights before the final, untouched test eval.
    if best_state is not None:
        model.load_state_dict(best_state)

    final_dl = test_dl or val_dl
    if final_dl is not None:
        r = evaluate(model, final_dl, dev, classes)
        label = "test" if test_dl else "val (no test set at this scale)"
        print(f"\nheld-out ({label})  top-1 {r['top1']:.1%}  "
              f"top-5 {r['top5']:.1%}  genus {r['genus']:.1%}  (n={r['n']})")

    torch.save({"state_dict": model.state_dict(), "classes": classes,
                "data_config": cfg}, "model.pt")
    print("saved checkpoint -> model.pt")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--species", type=int, default=100)
    ap.add_argument("--per-species", type=int, default=200)
    ap.add_argument("--min-obs", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--lr", type=float, default=3e-4)
    run(ap.parse_known_args()[0])
