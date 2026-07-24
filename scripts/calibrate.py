"""Calibration analysis: what does the model's confidence actually mean?

Answers "is a 26% top guess good?" with data. Samples species across the
difficulty range, pulls their genuinely held-out TEST photos (same
observation-keyed split as training — we hash the observation uuid), records
(top-1 prob, was top-1/top-5 correct), and bins by confidence to show the
real accuracy at each confidence level. Also reports ECE and a data-driven
"not sure" threshold to replace the guessed 0.30.

Runs on CPU. ~10-15 min (download + inference). Writes reports/calibration.json.

Run: .venv/bin/python scripts/calibrate.py
"""
import csv
import hashlib
import io
import json
import os
import sys
import time
from collections import defaultdict

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from ctplantid.predict import PlantModel  # noqa: E402
from PIL import Image  # noqa: E402

API = "https://api.inaturalist.org/v1"
CT_PLACE_ID = 49
HEADERS = {"User-Agent": "ct-plant-id/0.1 (calibration study)"}
CKPT = os.path.join(ROOT, "runs", "stage2", "model.pt")
CHECKLIST = os.path.join(ROOT, "data", "ct_checklist.csv")
OUT = os.path.join(ROOT, "reports", "calibration.json")

N_SPECIES = 60          # spread across head/mid/tail
PER_SPECIES = 15        # test photos each
MAX_SCAN = 250          # obs to scan per species to find enough test ones


def split_for(uuid, val=0.15, test=0.15):
    """Same hash split as training — reconstructs the held-out test set."""
    h = int(hashlib.md5(str(uuid).encode()).hexdigest(), 16) % 10000 / 10000
    return "test" if h < test else "val" if h < test + val else "train"


def sample_species(classes):
    """Stratified sample across the obs-count distribution so the calibration
    curve covers the full confidence range (easy head + hard tail)."""
    in_model = set(classes)
    rows = []
    for r in csv.DictReader(open(CHECKLIST)):
        if r["name"] in in_model:
            rows.append((int(r["obs_count"]), r["taxon_id"], r["name"]))
    rows.sort(reverse=True)  # most-observed first
    n = len(rows)
    # even strides through the sorted list -> a spread of head/mid/tail
    step = max(1, n // N_SPECIES)
    return [{"taxon_id": t, "name": nm} for _, t, nm in rows[::step]][:N_SPECIES]


def test_photos(taxon_id, cap):
    imgs, scanned, page = [], 0, 1
    while len(imgs) < cap and scanned < MAX_SCAN:
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
            if split_for(obs.get("uuid", obs["id"])) != "test":
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
            time.sleep(0.25)
            if len(imgs) >= cap:
                break
        page += 1
        time.sleep(1)
    return imgs


def main():
    model = PlantModel(CKPT, device="cpu")
    species = sample_species(model.classes)
    print(f"calibrating on {len(species)} species (spread head->tail)…")

    records = []  # (top_prob, correct1, correct5)
    for i, s in enumerate(species):
        imgs = test_photos(s["taxon_id"], PER_SPECIES)
        for im in imgs:
            preds = model.predict(im, k=5)
            names = [p["species"] for p in preds]
            records.append((preds[0]["prob"],
                            int(names[0] == s["name"]),
                            int(s["name"] in names)))
        print(f"  [{i+1}/{len(species)}] {s['name']}: {len(imgs)} imgs "
              f"(total {len(records)})")

    n = len(records)
    overall1 = sum(c for _, c, _ in records) / n
    overall5 = sum(c for _, _, c in records) / n

    # bin by top-1 prob
    bins = defaultdict(list)
    for prob, c1, c5 in records:
        bins[min(int(prob * 10), 9)].append((prob, c1, c5))

    print(f"\n=== calibration (n={n}) ===")
    print(f"overall  top-1 {overall1:.1%}   top-5 {overall5:.1%}")
    print(f"\n{'confidence':>12} | {'n':>5} | {'top-1 acc':>9} | {'top-5 acc':>9} | {'mean conf':>9}")
    ece, table = 0.0, []
    for b in range(10):
        items = bins.get(b, [])
        if not items:
            continue
        acc1 = sum(c for _, c, _ in items) / len(items)
        acc5 = sum(c for _, _, c in items) / len(items)
        mconf = sum(p for p, _, _ in items) / len(items)
        ece += len(items) / n * abs(acc1 - mconf)
        lo, hi = b / 10, (b + 1) / 10
        print(f"  {lo:.1f}-{hi:.1f}    | {len(items):>5} | {acc1:>8.1%} | {acc5:>8.1%} | {mconf:>8.1%}")
        table.append({"lo": lo, "hi": hi, "n": len(items),
                      "top1": round(acc1, 4), "top5": round(acc5, 4),
                      "mean_conf": round(mconf, 4)})

    print(f"\nECE (expected calibration error): {ece:.3f}")
    # suggest a not-sure threshold: lowest prob bin where top-1 is still a
    # majority (>=50%). Below that, "not sure" is genuinely honest.
    honest = [row["lo"] for row in table if row["top1"] >= 0.5]
    suggested = min(honest) if honest else None
    print(f"suggested not-sure threshold (top-1 stays >=50%): "
          f"{suggested if suggested is not None else 'n/a'}  (current is 0.30)")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump({"n": n, "overall_top1": overall1, "overall_top5": overall5,
               "ece": ece, "bins": table, "suggested_threshold": suggested},
              open(OUT, "w"), indent=2)
    print(f"\nwrote {os.path.relpath(OUT, ROOT)}")


if __name__ == "__main__":
    main()
