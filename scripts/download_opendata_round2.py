"""Round-2 dataset build from the iNaturalist Open Data bucket.

What changes versus scripts/download_opendata.py, and why (evidence in
docs/round2_plan.md, reports/tree_benchmark.md, reports/stratified_eval.json):

1. **Multiple photos per observation.** The v1 builder kept exactly one photo per
   observation and discarded the rest. Measured availability: Quercus rubra
   averages 3.40 photos/observation, Acer rubrum 1.30, Toxicodendron radicans
   1.23 — trees carry ~3x more because people shoot leaf, THEN BARK, THEN whole
   tree. Those are precisely the image types the tree benchmark found missing
   (trees 66.5% top-1 vs shrubs 84.6%, herbaceous 81.0%).

   Note this needs no woody-specific rule: asking for up to N photos per
   observation naturally gives trees ~3 and herbaceous plants ~1.3, because
   that is what exists. The data targets itself.

   **Leakage-safe by construction**: the split is keyed on observation_uuid, so
   every extra photo of an observation lands in the same split as its siblings.

2. **A higher cap for confusable genera.** Stratified eval shows accuracy still
   climbing at the cap (sparse <20 imgs 39.7% -> tail 20-99 69.4% -> capped 100+
   85.4%), and nothing in the v1 set exceeds 300, so whether image #301+ helps is
   untested. The genera where within-genus confusion is documented (Quercus 35%
   species vs 69% genus, Carya 38/75, Pinus 50/89) are the cheap place to find
   out.

3. **Licences are recovered.** The v1 manifest lost per-photo licences when it
   was rebuilt from disk after the disk-full crash; photos.csv.gz carries them,
   so this build restores attribution data.

4. **A disk headroom check up front** — the v1 build died at 100% disk.

Pruning of unlearnable classes is deliberately NOT done here: pull everything
available first, then prune on final counts (a class may be thin only because
the v1 pull was interrupted at ~75%). This script reports the candidates.

Run on the build droplet:
  python scripts/download_opendata_round2.py --out /root/round2 --photos-per-obs 3
"""
import argparse
import csv
import gzip
import io
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

BASE = "https://inaturalist-open-data.s3.amazonaws.com"
PHOTO_URL = BASE + "/photos/{photo_id}/medium.{ext}"
CHECKLIST = os.path.join(ROOT, "data", "ct_checklist.csv")

# Genera where within-genus confusion is measured, so extra images have the best
# chance of buying discrimination rather than redundancy.
CONFUSABLE_GENERA = {"Quercus", "Carya", "Pinus", "Prunus", "Betula", "Acer",
                     "Salix", "Crataegus", "Viburnum", "Solidago", "Carex"}

# A class below this many images at the end can't be learned or measured; the
# script reports them for pruning rather than deleting anything itself.
PRUNE_BELOW = 20


def split_for(obs_uuid, val=0.15, test=0.15):
    """Observation-keyed split — identical to v1 so old and new photos of the
    same observation always agree."""
    import hashlib
    h = int(hashlib.md5(str(obs_uuid).encode()).hexdigest(), 16) % 10000 / 10000
    return "test" if h < test else "val" if h < test + val else "train"


def slug(name):
    return name.strip().lower().replace(" ", "_")


def stream_tsv(name, max_rows=0):
    """Stream a gzipped TSV from the bucket without holding it in RAM."""
    r = requests.get(f"{BASE}/{name}", stream=True, timeout=120)
    r.raise_for_status()
    with gzip.open(r.raw, mode="rt", newline="") as fh:
        for i, row in enumerate(csv.DictReader(fh, delimiter="\t")):
            if max_rows and i >= max_rows:
                break
            yield row


def load_ct_taxa(path=None):
    """taxon_id -> species name. Accepts either the CT checklist (column `name`)
    or the ornamental scope file (column `species`), so the same builder can pull
    wild flora and cultivated ornamentals."""
    taxa = {}
    for row in csv.DictReader(open(path or CHECKLIST)):
        name = row.get("name") or row.get("species")
        if name:
            taxa[str(row["taxon_id"])] = name
    return taxa


def cap_for(species, cap, cap_confusable):
    return cap_confusable if species.split()[0] in CONFUSABLE_GENERA else cap


def check_disk(out_dir, need_gb):
    os.makedirs(out_dir, exist_ok=True)
    free_gb = shutil.disk_usage(out_dir).free / 1e9
    print(f"disk free at {out_dir}: {free_gb:.0f} GB (want >= {need_gb} GB)")
    if free_gb < need_gb:
        raise SystemExit(
            f"STOP: only {free_gb:.0f} GB free. The v1 build died at 100% disk — "
            f"resize, point --out elsewhere, or lower --photos-per-obs.")


def select_observations(ct_taxa, cap, cap_confusable, max_scan, quality="research"):
    """Pass 1: observation_uuid -> taxon_id, capped per taxon with headroom for
    observations whose photos fail.

    `quality`: "research" for wild flora (the v1 rule), or "any" for cultivated
    ornamentals — iNaturalist marks planted specimens **casual**, which is
    exactly why research-grade excludes garden plants and why the hydrangea was
    missing from v1 scope.
    """
    selected, per_taxon = {}, {}
    for i, row in enumerate(stream_tsv("observations.csv.gz", max_scan)):
        if i and i % 5_000_000 == 0:
            print(f"  obs scanned {i:,}, selected {len(selected):,}")
        tid = row["taxon_id"]
        name = ct_taxa.get(tid)
        if name is None:
            continue
        if quality == "research" and row["quality_grade"] != "research":
            continue
        limit = cap_for(name, cap, cap_confusable) * 2
        if per_taxon.get(tid, 0) < limit:
            selected[row["observation_uuid"]] = tid
            per_taxon[tid] = per_taxon.get(tid, 0) + 1
    return selected


def spread_indices(n_available, want):
    """Pick `want` positions spread evenly across `n_available`, always keeping
    position 0.

    Taking photos 0,1,2 off an observation often yields three near-identical
    burst frames of the same leaf, while the bark shot sits at position 4. The
    photos table carries a `position` field (order within the observation), so
    spreading across the full range is a free way to sample different subjects:
    6 photos -> 0, 2, 5 rather than 0, 1, 2.
    """
    if n_available <= want:
        return list(range(n_available))
    if want == 1:
        return [0]
    step = (n_available - 1) / (want - 1)
    return sorted({int(round(i * step)) for i in range(want)})


def select_photos(selected_obs, ct_taxa, cap, cap_confusable, per_obs, max_scan):
    """Pass 2: for each selected observation, keep up to `per_obs` photos chosen
    to be SPREAD across the observation's photo positions, then apply the
    per-species cap.

    Two passes over the buffered rows are needed because diversity selection
    can't be decided while streaming — we have to see all of an observation's
    photos before choosing which to keep.
    """
    by_obs = {}
    for i, row in enumerate(stream_tsv("photos.csv.gz", max_scan)):
        if i and i % 5_000_000 == 0:
            print(f"  photos scanned {i:,}, buffered {len(by_obs):,} observations")
        obs = row["observation_uuid"]
        if obs not in selected_obs:
            continue
        try:
            pos = int(row.get("position") or 0)
        except ValueError:
            pos = 0
        by_obs.setdefault(obs, []).append(
            (pos, row["photo_id"], row["extension"], row["license"]))

    total_available = sum(len(v) for v in by_obs.values())
    print(f"  {total_available:,} photos available across {len(by_obs):,} "
          f"observations ({total_available/max(len(by_obs),1):.2f} per observation)")

    # Choose spread photos per observation, then fill each species up to its cap.
    # Observations are consumed in arbitrary order, so a species' quota is spread
    # across many individuals rather than exhausted on the first few.
    photos, per_taxon = [], {}
    for obs, entries in by_obs.items():
        tid = selected_obs[obs]
        name = ct_taxa[tid]
        limit = cap_for(name, cap, cap_confusable)
        if per_taxon.get(tid, 0) >= limit:
            continue
        entries.sort()  # by position
        chosen = spread_indices(len(entries), per_obs)
        for out_idx, e in enumerate(chosen):
            if per_taxon.get(tid, 0) >= limit:
                break
            pos, photo_id, ext, lic = entries[e]
            per_taxon[tid] = per_taxon.get(tid, 0) + 1
            photos.append({"observation_uuid": obs, "taxon_id": tid, "species": name,
                           "photo_id": photo_id, "ext": ext, "license": lic,
                           "idx": out_idx, "position": pos})
    return photos


def _fetch_one(p, img_dir):
    """Download one photo unless already present. Resumable and concurrency-safe."""
    dest = os.path.join(img_dir, slug(p["species"]))
    os.makedirs(dest, exist_ok=True)
    path = os.path.join(dest, f"{p['observation_uuid']}_{p['idx']}.{p['ext']}")
    if not os.path.exists(path):
        try:
            b = requests.get(PHOTO_URL.format(photo_id=p["photo_id"], ext=p["ext"]),
                             timeout=60).content
            with open(path, "wb") as fh:
                fh.write(b)
        except Exception:
            return None
    return {"observation_uuid": p["observation_uuid"], "taxon_id": p["taxon_id"],
            "species": p["species"], "path": path, "license": p["license"],
            "split": split_for(p["observation_uuid"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/root/round2", help="build directory (local disk)")
    ap.add_argument("--photos-per-obs", type=int, default=3,
                    help="max photos per observation (v1 used 1)")
    ap.add_argument("--cap", type=int, default=300, help="images per species")
    ap.add_argument("--cap-confusable", type=int, default=600,
                    help="images per species for CONFUSABLE_GENERA")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--need-gb", type=int, default=140, help="required free disk")
    ap.add_argument("--max-scan", type=int, default=0, help="row cap for smoke tests")
    ap.add_argument("--taxa-csv", default=None,
                    help="species list (default: CT checklist). Point at "
                         "data/ornamental_species.csv for the cultivated pull.")
    ap.add_argument("--quality-grade", choices=["research", "any"], default="research",
                    help="'any' includes casual/needs_id — required for cultivated "
                         "ornamentals, which iNaturalist marks casual")
    args = ap.parse_args()

    img_dir = os.path.join(args.out, "images")
    manifest = os.path.join(args.out, "manifest.csv")
    check_disk(args.out, args.need_gb)

    ct_taxa = load_ct_taxa(args.taxa_csv)
    print(f"{len(ct_taxa):,} taxa | cap {args.cap} "
          f"(confusable {args.cap_confusable}) | up to {args.photos_per_obs} photos/obs "
          f"| quality: {args.quality_grade}")

    print("\npass 1: streaming observations.csv.gz (~12 GB)…")
    obs = select_observations(ct_taxa, args.cap, args.cap_confusable, args.max_scan,
                              args.quality_grade)
    print(f"  selected {len(obs):,} observations")

    print("\npass 2: streaming photos.csv.gz (~18 GB)…")
    photos = select_photos(obs, ct_taxa, args.cap, args.cap_confusable,
                           args.photos_per_obs, args.max_scan)
    print(f"  selected {len(photos):,} photos "
          f"({len(photos)/max(len(obs),1):.2f} per observation)")

    print(f"\ndownloading with {args.workers} workers…")
    os.makedirs(img_dir, exist_ok=True)
    rows, done = [], 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for res in pool.map(lambda p: _fetch_one(p, img_dir), photos):
            done += 1
            if res:
                rows.append(res)
            if done % 20000 == 0:
                free = shutil.disk_usage(args.out).free / 1e9
                print(f"  {done:,}/{len(photos):,} fetched, {len(rows):,} ok, "
                      f"{free:.0f} GB free")
                if free < 5:
                    print("  STOPPING: under 5 GB free")
                    break

    with open(manifest, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["observation_uuid", "taxon_id", "species",
                                          "path", "license", "split"])
        w.writeheader()
        w.writerows(rows)

    from collections import Counter
    per_species = Counter(r["species"] for r in rows)
    thin = [s for s, n in per_species.items() if n < PRUNE_BELOW]
    licensed = sum(1 for r in rows if (r["license"] or "").strip())
    print(f"\n{len(rows):,} images across {len(per_species):,} species "
          f"-> {os.path.relpath(manifest, args.out)}")
    print(f"licences recovered for {licensed:,}/{len(rows):,} photos "
          f"({licensed/max(len(rows),1):.1%})")
    print(f"classes still under {PRUNE_BELOW} images (prune candidates): {len(thin):,}")
    print("  prune with: python scripts/prune_classes.py --manifest "
          f"{manifest} --min-images {PRUNE_BELOW}")


if __name__ == "__main__":
    main()
