"""Per-class evaluator for a trained CT plant model — find where it fails.

Loads a checkpoint (model.pt from kaggle_train_v2.py), rebuilds a held-out
TEST set for the same species, and reports per-class accuracy plus the
worst-confused species pairs. Overall numbers should roughly match the
training run's; the value here is the breakdown.

LEAKAGE-SAFE: it re-downloads photos but keeps only observations whose
hash-split is "test" — the same partition the model was NOT trained on. So
this grades on genuinely unseen data, not memorized training photos.

Kaggle use: new notebook, Internet on (GPU optional), "Add Input" -> your
finished training notebook's output (which contains model.pt), then paste
this and Run. It finds model.pt automatically.
    python scripts/kaggle_eval.py --cap 25
"""
import argparse
import glob
import hashlib
import io
import subprocess
import sys
import time
from collections import Counter, defaultdict

import requests

try:
    import timm
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "timm"], check=True)
    import timm

import torch
from PIL import Image

API = "https://api.inaturalist.org/v1"
CT_PLACE_ID = 49
PLANTAE = 47126
HEADERS = {"User-Agent": "ct-plant-id/0.1 (learning project)"}
BACKBONE = "tf_efficientnetv2_s"


def split_for(obs_id, val=0.15, test=0.15):
    h = int(hashlib.md5(str(obs_id).encode()).hexdigest(), 16) % 10000 / 10000
    return "test" if h < test else "val" if h < test + val else "train"


def genus_of(name):
    return name.strip().lower().split(" ")[0]


def find_checkpoint():
    for p in ["model.pt", "/kaggle/working/model.pt"]:
        if glob.glob(p):
            return p
    hits = glob.glob("/kaggle/input/**/model.pt", recursive=True)
    if hits:
        return hits[0]
    sys.exit("model.pt not found — attach your training notebook's output as input")


def top_species(n, min_obs):
    """Re-resolve taxon_ids for the trained species (same call as training)."""
    out, page = [], 1
    while len(out) < n:
        r = requests.get(f"{API}/observations/species_counts",
                         params={"place_id": CT_PLACE_ID, "taxon_id": PLANTAE,
                                 "quality_grade": "research", "per_page": 200,
                                 "page": page}, headers=HEADERS, timeout=60)
        r.raise_for_status()
        res = r.json()["results"]
        if not res:
            break
        for row in res:
            t = row["taxon"]
            if t["rank"] == "species" and row["count"] >= min_obs:
                out.append({"taxon_id": t["id"], "name": t["name"]})
        page += 1
        time.sleep(1)
    return out[:n]


def test_photos(taxon_id, cap, max_scan=300):
    """Download up to `cap` photos from TEST-split observations of a species."""
    imgs, scanned, page = [], 0, 1
    while len(imgs) < cap and scanned < max_scan:
        r = requests.get(f"{API}/observations",
                         params={"taxon_id": taxon_id, "place_id": CT_PLACE_ID,
                                 "quality_grade": "research", "photos": "true",
                                 "order_by": "votes", "per_page": 50, "page": page},
                         headers=HEADERS, timeout=60)
        r.raise_for_status()
        res = r.json()["results"]
        if not res:
            break
        for obs in res:
            scanned += 1
            if split_for(obs["id"]) != "test":
                continue
            photos = obs.get("photos") or []
            if not photos or not photos[0].get("url"):
                continue
            url = photos[0]["url"].replace("/square.", "/medium.")
            try:
                b = requests.get(url, headers=HEADERS, timeout=60).content
                imgs.append(Image.open(io.BytesIO(b)).convert("RGB"))
            except Exception:
                continue
            time.sleep(0.3)
            if len(imgs) >= cap:
                break
        page += 1
        time.sleep(1)
    return imgs


def main(args):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(find_checkpoint(), map_location=dev, weights_only=False)
    classes = ckpt["classes"]
    name_to_idx = {n: i for i, n in enumerate(classes)}
    print(f"loaded checkpoint: {len(classes)} classes, device {dev}")

    model = timm.create_model(BACKBONE, pretrained=False, num_classes=len(classes))
    model.load_state_dict(ckpt["state_dict"])
    model.to(dev).eval()
    tf = timm.data.create_transform(**ckpt["data_config"], is_training=False)

    print("rebuilding the held-out TEST set (same species, test split only)…")
    species = top_species(len(classes), args.min_obs)
    samples = []  # (tensor, true_idx)
    for s in species:
        if s["name"] not in name_to_idx:
            continue
        idx = name_to_idx[s["name"]]
        imgs = test_photos(s["taxon_id"], args.cap)
        samples.extend((tf(im), idx) for im in imgs)
        print(f"  {s['name']}: {len(imgs)} test photos")

    # inference
    per_class = defaultdict(lambda: [0, 0])   # name -> [correct, total]
    confusions = Counter()                     # (true, pred_top1) -> count
    t1 = t5 = g1 = 0
    with torch.no_grad():
        for i in range(0, len(samples), args.batch):
            batch = samples[i:i + args.batch]
            x = torch.stack([t for t, _ in batch]).to(dev)
            top = model(x).topk(min(5, len(classes)), dim=1).indices.cpu()
            for j, (_, y) in enumerate(batch):
                true_name = classes[y]
                pred = top[j][0].item()
                per_class[true_name][1] += 1
                hit = int(pred == y)
                per_class[true_name][0] += hit
                t1 += hit
                t5 += int(y in top[j])
                g1 += int(genus_of(classes[pred]) == genus_of(true_name))
                if not hit:
                    confusions[(true_name, classes[pred])] += 1

    n = len(samples)
    print(f"\n=== overall (n={n}) ===")
    print(f"top-1 {t1/n:.1%}   top-5 {t5/n:.1%}   genus {g1/n:.1%}")

    acc = sorted(((c/tot, name, tot) for name, (c, tot) in per_class.items() if tot),
                 key=lambda x: x[0])
    print("\n=== 15 weakest species (top-1 accuracy) ===")
    for a, name, tot in acc[:15]:
        print(f"  {a:5.0%}  {name}  (n={tot})")
    print("\n=== 5 strongest ===")
    for a, name, tot in acc[-5:]:
        print(f"  {a:5.0%}  {name}  (n={tot})")

    print("\n=== 15 most common confusions (true -> predicted) ===")
    for (tru, pred), cnt in confusions.most_common(15):
        print(f"  {cnt:3d}x  {tru}  ->  {pred}")

    with open("per_class_accuracy.csv", "w") as f:
        f.write("species,accuracy,n\n")
        for a, name, tot in acc:
            f.write(f"{name},{a:.4f},{tot}\n")
    print("\nsaved per_class_accuracy.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=25, help="test photos per species")
    ap.add_argument("--min-obs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=32)
    main(ap.parse_known_args()[0])
