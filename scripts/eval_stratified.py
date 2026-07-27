"""Diagnostic 2: accuracy stratified by data tier and by plant type.

GOALS.md has demanded stratified reporting since the project started ("never a
single average") and it has never been computed. The tree benchmark then proved
the point the hard way: the global 80.1% top-1 hides a tree sub-population
sitting at 49.7%. This measures how much else the global number is hiding.

Two stratifications:
  * **tier**  — by how much training data the class actually has (head / mid /
    tail / sparse). Tests whether accuracy tracks data volume once the 300-image
    cap is accounted for, which the tree benchmark could not settle.
  * **growth form** — tree / shrub / herbaceous. Approximated from a curated
    woody-genus list; approximate on purpose, and labelled as such in the output.

Evaluates on held-out TEST-split photos only, using the same observation-keyed
hash as training, so nothing scored here was learned from.

Run on the box that has the manifest:
  python scripts/eval_stratified.py --data /mnt/ct-plant-data/data --per-class 8
"""
import argparse
import csv
import hashlib
import json
import os
import random
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from ctplantid.predict import PlantModel  # noqa: E402
from PIL import Image  # noqa: E402

# Tiers by TRAIN-split image count. Note the split is ~70/15/15, so a class at
# the 300-image cap holds only ~210 TRAIN images — which is why the boundaries
# below are scaled to the train split rather than to total images. (An earlier
# version used total-image boundaries and left the top tier permanently empty,
# because no class can exceed the cap in the train split.)
TIERS = [("sparse", 0, 19), ("tail", 20, 69), ("mid", 70, 149), ("capped", 150, 10 ** 9)]

# Woody genera common in CT. Approximate by design: this is a reporting axis,
# not a botanical claim, and anything unlisted is counted herbaceous.
TREE_GENERA = {
    "Quercus", "Acer", "Betula", "Fagus", "Fraxinus", "Carya", "Pinus", "Tsuga",
    "Picea", "Platanus", "Tilia", "Ulmus", "Populus", "Liriodendron", "Abies",
    "Liquidambar", "Nyssa", "Juglans", "Robinia", "Gleditsia", "Ginkgo", "Larix",
    "Zelkova", "Castanea", "Catalpa", "Ailanthus", "Sassafras", "Metasequoia",
    "Cryptomeria", "Chamaecyparis", "Thuja", "Juniperus", "Magnolia", "Pyrus",
    "Malus", "Prunus", "Cercis", "Ostrya", "Carpinus", "Cladrastis", "Koelreuteria",
}
SHRUB_GENERA = {
    "Rhododendron", "Kalmia", "Vaccinium", "Ilex", "Viburnum", "Cornus", "Spiraea",
    "Hydrangea", "Buxus", "Syringa", "Forsythia", "Berberis", "Euonymus", "Rosa",
    "Rubus", "Lonicera", "Ligustrum", "Fothergilla", "Deutzia", "Weigela",
    "Physocarpus", "Aronia", "Clethra", "Hamamelis", "Amelanchier", "Sambucus",
    "Elaeagnus", "Rhus", "Toxicodendron", "Myrica", "Alnus", "Salix", "Corylus",
}


def growth_form(species):
    genus = species.split()[0]
    if genus in TREE_GENERA:
        return "tree"
    if genus in SHRUB_GENERA:
        return "shrub"
    return "herbaceous"


def split_for(uuid, val=0.15, test=0.15):
    h = int(hashlib.md5(str(uuid).encode()).hexdigest(), 16) % 10000 / 10000
    return "test" if h < test else "val" if h < test + val else "train"


def tier_of(n):
    for name, lo, hi in TIERS:
        if lo <= n <= hi:
            return name
    return "head"


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
    ap.add_argument("--per-class", type=int, default=8, help="test photos per class")
    ap.add_argument("--max-classes", type=int, default=0, help="0 = all")
    ap.add_argument("--out", default=os.path.join(ROOT, "reports", "stratified_eval.json"))
    args = ap.parse_args()

    model = PlantModel(args.model, device=None)
    in_scope = set(model.classes)

    rows = list(csv.DictReader(open(os.path.join(args.data, "manifest.csv"))))
    train_counts = defaultdict(int)
    test_by_class = defaultdict(list)
    for r in rows:
        sp = r["species"]
        if sp not in in_scope:
            continue
        split = r.get("split") or split_for(r.get("observation_uuid", ""))
        if split == "train":
            train_counts[sp] += 1
        elif split == "test":
            test_by_class[sp].append(r["path"])

    classes = sorted(test_by_class)
    if args.max_classes:
        random.seed(0)
        classes = random.sample(classes, min(args.max_classes, len(classes)))
    print(f"evaluating {len(classes)} classes, <= {args.per_class} test photos each "
          f"(device {model.device})")

    recs = []
    for i, sp in enumerate(classes, 1):
        paths = sorted(test_by_class[sp])[:args.per_class]
        for p in paths:
            full = resolve(p, args.data)
            try:
                preds = model.identify(Image.open(full), k=5)
            except Exception:
                continue
            names = [c["species"] for c in preds["candidates"]]
            recs.append({
                "species": sp,
                "tier": tier_of(train_counts[sp]),
                "form": growth_form(sp),
                "top1": int(names[0] == sp),
                "top5": int(sp in names),
                "genus1": int(names[0].split()[0] == sp.split()[0]),
                "conf": preds["candidates"][0]["prob"],
            })
        if i % 100 == 0:
            print(f"  [{i}/{len(classes)}] {len(recs)} photos scored")

    if not recs:
        raise SystemExit("no photos scored — check --data path")

    def report(key, order):
        print(f"\n=== by {key} ===")
        print(f"{key:>12} {'n':>7} {'top-1':>8} {'top-5':>8} {'genus':>8} {'mean conf':>10}")
        out = {}
        for val in order:
            g = [r for r in recs if r[key] == val]
            if not g:
                continue
            stats = {
                "n": len(g),
                "top1": sum(r["top1"] for r in g) / len(g),
                "top5": sum(r["top5"] for r in g) / len(g),
                "genus1": sum(r["genus1"] for r in g) / len(g),
                "mean_conf": sum(r["conf"] for r in g) / len(g),
            }
            out[val] = stats
            print(f"{val:>12} {stats['n']:>7} {stats['top1']:>7.1%} {stats['top5']:>7.1%} "
                  f"{stats['genus1']:>7.1%} {stats['mean_conf']:>9.1%}")
        return out

    overall = {
        "n": len(recs),
        "top1": sum(r["top1"] for r in recs) / len(recs),
        "top5": sum(r["top5"] for r in recs) / len(recs),
        "genus1": sum(r["genus1"] for r in recs) / len(recs),
    }
    print(f"\noverall: n={overall['n']} top-1 {overall['top1']:.1%} "
          f"top-5 {overall['top5']:.1%} genus {overall['genus1']:.1%}")
    by_tier = report("tier", [t[0] for t in TIERS])
    by_form = report("form", ["tree", "shrub", "herbaceous"])

    print("\nNOTE: growth form is approximated from a curated genus list — a "
          "reporting axis, not a botanical claim.")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({"overall": overall, "by_tier": by_tier, "by_form": by_form},
              open(args.out, "w"), indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
