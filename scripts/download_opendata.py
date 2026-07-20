"""Bulk dataset builder from the iNaturalist Open Data set on AWS (cloud box).

This is the TOS-compliant path for the FULL 2,542-species dataset. It streams
the public Open Data metadata (no AWS account needed) and filters to the CT
checklist, then downloads only the photos it needs.

The metadata files are large and TAB-separated (verified):
    taxa.csv.gz          ~38 MB
    observations.csv.gz  ~12 GB   (global; streamed, never fully in RAM)
    photos.csv.gz        ~18 GB   (global; streamed)
so this MUST run on the DigitalOcean droplet, not a laptop. It streams each
file once, holding only the selected CT observations in memory.

Pipeline (see docs/model_and_training.md):
  pass 1  observations.csv.gz -> keep research-grade obs of CT taxa (capped/taxon)
  pass 2  photos.csv.gz       -> one photo per selected obs (id, ext, license)
  pass 3  download those photos -> data/images/<species>/<obs>_0.<ext>,
                                    data/manifest.csv (obs-keyed split, license)

Run on the droplet:
  python scripts/download_opendata.py --cap 300
Test the plumbing cheaply (streams only the first N rows, downloads nothing new):
  python scripts/download_opendata.py --max-scan 200000 --cap 5
"""
import argparse
import csv
import gzip
import hashlib
import io
import os
from concurrent.futures import ThreadPoolExecutor

import requests

BASE = "https://inaturalist-open-data.s3.amazonaws.com"
PHOTO_URL = BASE + "/photos/{photo_id}/medium.{ext}"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKLIST = os.path.join(ROOT, "data", "ct_checklist.csv")
IMG_DIR = os.path.join(ROOT, "data", "images")
MANIFEST = os.path.join(ROOT, "data", "manifest.csv")


def split_for(obs_uuid, val=0.15, test=0.15):
    h = int(hashlib.md5(str(obs_uuid).encode()).hexdigest(), 16) % 10000 / 10000
    return "test" if h < test else "val" if h < test + val else "train"


def slug(name):
    return name.strip().lower().replace(" ", "_")


def stream_tsv(name, max_rows=0):
    """Yield rows (as dicts) from a gzipped tab-separated Open Data file,
    streaming — never downloads the whole file into memory. max_rows>0 stops
    early (for cheap plumbing tests)."""
    url = f"{BASE}/{name}"
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        gz = gzip.GzipFile(fileobj=r.raw)
        text = io.TextIOWrapper(gz, encoding="utf-8", errors="replace")
        reader = csv.DictReader(text, delimiter="\t")
        for i, row in enumerate(reader):
            if max_rows and i >= max_rows:
                break
            yield row


def load_ct_taxa():
    """taxon_id -> species name, from the CT checklist (iNat taxon ids, same
    namespace as Open Data)."""
    out = {}
    with open(CHECKLIST) as f:
        for r in csv.DictReader(f):
            out[r["taxon_id"]] = r["name"]
    return out


def select_observations(ct_taxa, cap, max_scan):
    """Pass 1: obs_uuid -> taxon_id for research-grade obs of CT taxa, capped
    per taxon (cap*2 headroom for obs whose photo fails)."""
    selected, per_taxon = {}, {}
    limit = cap * 2
    for i, row in enumerate(stream_tsv("observations.csv.gz", max_scan)):
        if i and i % 5_000_000 == 0:
            print(f"  obs scanned {i:,}, selected {len(selected):,}")
        tid = row["taxon_id"]
        if tid in ct_taxa and row["quality_grade"] == "research":
            if per_taxon.get(tid, 0) < limit:
                selected[row["observation_uuid"]] = tid
                per_taxon[tid] = per_taxon.get(tid, 0) + 1
    return selected


def select_photos(selected_obs, ct_taxa, cap, max_scan):
    """Pass 2: one photo per selected obs, capped per taxon."""
    photos, per_taxon, seen_obs = [], {}, set()
    for i, row in enumerate(stream_tsv("photos.csv.gz", max_scan)):
        if i and i % 5_000_000 == 0:
            print(f"  photos scanned {i:,}, kept {len(photos):,}")
        obs = row["observation_uuid"]
        tid = selected_obs.get(obs)
        if tid is None or obs in seen_obs:
            continue
        if per_taxon.get(tid, 0) >= cap:
            continue
        seen_obs.add(obs)
        per_taxon[tid] = per_taxon.get(tid, 0) + 1
        photos.append({"observation_uuid": obs, "taxon_id": tid,
                       "species": ct_taxa[tid], "photo_id": row["photo_id"],
                       "ext": row["extension"], "license": row["license"]})
    return photos


def _fetch_one(p):
    """Download one photo if not already on disk; return its manifest row, or
    None on failure. Safe to run concurrently. Resumable: a photo already on
    disk is kept (row still returned) and not re-fetched — so re-running after
    an interrupted download only pulls what's missing."""
    dest = os.path.join(IMG_DIR, slug(p["species"]))
    os.makedirs(dest, exist_ok=True)
    path = os.path.join(dest, f"{p['observation_uuid']}_0.{p['ext']}")
    if not os.path.exists(path):
        try:
            url = PHOTO_URL.format(photo_id=p["photo_id"], ext=p["ext"])
            b = requests.get(url, timeout=60).content
            with open(path, "wb") as img:
                img.write(b)
        except Exception:
            return None
    return {"observation_uuid": p["observation_uuid"], "taxon_id": p["taxon_id"],
            "species": p["species"], "path": os.path.relpath(path, ROOT),
            "license": p["license"], "split": split_for(p["observation_uuid"])}


def download_photos(photos, workers):
    """Parallel, resumable photo download. `workers` concurrent fetches; the
    manifest is written from the main thread (single writer, no lock needed)."""
    os.makedirs(IMG_DIR, exist_ok=True)
    saved, total = 0, len(photos)
    chunk = 4000  # bound the number of in-flight futures / memory
    with open(MANIFEST, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["observation_uuid", "taxon_id",
                           "species", "path", "license", "split"])
        w.writeheader()
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for i in range(0, total, chunk):
                for row in ex.map(_fetch_one, photos[i:i + chunk]):
                    if row:
                        w.writerow(row)
                        saved += 1
                print(f"  downloaded {saved:,}/{total:,}")
    return saved


def main(args):
    ct_taxa = load_ct_taxa()
    print(f"CT checklist: {len(ct_taxa)} taxa")
    print("pass 1: scanning observations.csv.gz (streaming ~12 GB)…")
    obs = select_observations(ct_taxa, args.cap, args.max_scan)
    print(f"  selected {len(obs):,} observations")
    print("pass 2: scanning photos.csv.gz (streaming ~18 GB)…")
    photos = select_photos(obs, ct_taxa, args.cap, args.max_scan)
    print(f"  {len(photos):,} photos to fetch across "
          f"{len(set(p['taxon_id'] for p in photos))} species")
    print(f"pass 3: downloading photos with {args.workers} workers…")
    n = download_photos(photos, args.workers)
    print(f"done: {n:,} photos -> {os.path.relpath(MANIFEST, ROOT)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=300, help="max photos per species")
    ap.add_argument("--max-scan", type=int, default=0,
                    help="stop scanning each metadata file after N rows (testing)")
    ap.add_argument("--workers", type=int, default=16,
                    help="concurrent photo downloads (16 is a good default)")
    main(ap.parse_args())
