"""Self-contained first-run trainer for Kaggle (or anywhere with a GPU).

Paste this whole file into ONE Kaggle notebook cell, turn on GPU + Internet
in the notebook settings, and Run. It downloads a small subset of CT plant
species from iNaturalist, fine-tunes a pretrained EfficientNetV2-S on them
with an OBSERVATION-KEYED split (no leakage), and prints top-1 / top-5
accuracy. Nothing to upload — it fetches its own data.

Deliberately self-contained (no dependency on the ct-plant-id repo) so it
"just runs". Defaults are small for a fast first result; scale them up once
this works. Locally you can shrink it with CLI flags for a quick check:
    python scripts/kaggle_train.py --species 2 --per-species 2 --epochs 1
"""
import argparse
import hashlib
import io
import subprocess
import sys
import time

import requests

# timm may not be preinstalled on Kaggle; make the script self-sufficient.
try:
    import timm
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "timm"], check=True)
    import timm

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

API = "https://api.inaturalist.org/v1"
CT_PLACE_ID = 49
PLANTAE = 47126
HEADERS = {"User-Agent": "ct-plant-id/0.1 (learning project)"}


# ---------- observation-keyed split (no leakage) ----------

def split_for(obs_id, val=0.15, test=0.15):
    h = int(hashlib.md5(str(obs_id).encode()).hexdigest(), 16) % 10000 / 10000
    return "test" if h < test else "val" if h < test + val else "train"


# ---------- pull a small subset from iNaturalist ----------

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
    """Return (records, classes). Each record: dict(img=PIL, label=int, split)."""
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


def make_tf(size, train):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(size, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    return transforms.Compose([
        transforms.Resize(int(size * 1.15)),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
    ])


# ---------- train + eval ----------

def run(args):
    dev = ("cuda" if torch.cuda.is_available()
           else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {dev}")

    print("downloading subset from iNaturalist…")
    records, classes = build_subset(args.species, args.per_species, args.min_obs)
    tr = [r for r in records if r["split"] == "train"]
    te = [r for r in records if r["split"] in ("val", "test")]
    print(f"{len(records)} images, {len(classes)} classes | "
          f"train {len(tr)} / held-out {len(te)}")

    train_dl = DataLoader(PlantSet(tr, make_tf(args.img_size, True)),
                          batch_size=args.batch, shuffle=True, num_workers=2)
    test_dl = DataLoader(PlantSet(te, make_tf(args.img_size, False)),
                         batch_size=args.batch, num_workers=2)

    model = timm.create_model("tf_efficientnetv2_s", pretrained=True,
                              num_classes=len(classes)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    for ep in range(args.epochs):
        model.train()
        tot = 0.0
        for x, y in train_dl:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()
            tot += loss.item() * len(x)
        print(f"epoch {ep + 1}/{args.epochs}  train loss {tot / max(len(tr), 1):.3f}")

    # eval: top-1 and top-5 on the held-out (unseen-observation) photos
    model.eval()
    t1 = t5 = n = 0
    with torch.no_grad():
        for x, y in test_dl:
            x = x.to(dev)
            top = model(x).topk(min(5, len(classes)), dim=1).indices.cpu()
            for i, yi in enumerate(y):
                n += 1
                t1 += int(top[i][0] == yi)
                t5 += int(yi in top[i])
    if n:
        print(f"\nheld-out top-1: {t1 / n:.1%}   top-5: {t5 / n:.1%}   (n={n})")
    # On Kaggle the working dir is /kaggle/working, so a plain filename lands
    # in the persisted output; locally it lands in the current directory.
    torch.save({"state_dict": model.state_dict(), "classes": classes}, "model.pt")
    print("saved checkpoint -> model.pt")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--species", type=int, default=30)
    ap.add_argument("--per-species", type=int, default=60)
    ap.add_argument("--min-obs", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--lr", type=float, default=3e-4)
    # parse_known_args (not parse_args) so it ignores the "-f kernel.json"
    # that Jupyter/Kaggle injects into sys.argv — otherwise a pasted cell
    # crashes with an argparse error before it even starts.
    run(ap.parse_known_args()[0])
