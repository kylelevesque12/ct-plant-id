"""Drop species that can't be photographed with a phone.

The CT checklist comes from iNaturalist's "Plantae" kingdom, which includes
microscopic green algae (desmids like Cosmarium, Staurastrum). Those are real
CT organisms but a phone camera will never produce an image of one, so keeping
them as classes only adds dead output slots — and the risk that a blurry
close-up gets confidently labeled a desmid.

Keeps the LAND PLANT phyla (vascular plants, mosses, liverworts, hornworts)
and drops the algae. Resolves each species' phylum from the iNaturalist
taxonomy (taxa.csv.gz, ~38 MB, streamed).

Reversible: the full manifest is backed up to data/manifest_all.csv.

Run:  python3 scripts/filter_manifest.py
"""
import csv
import gzip
import io
import os
import shutil
from collections import Counter

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(ROOT, "data", "manifest.csv")
BACKUP = os.path.join(ROOT, "data", "manifest_all.csv")
TAXA_URL = "https://inaturalist-open-data.s3.amazonaws.com/taxa.csv.gz"

# Land plants — everything you can point a camera at in the field.
KEEP_PHYLA = {"Tracheophyta", "Bryophyta", "Marchantiophyta", "Anthocerotophyta"}


def stream_taxa():
    with requests.get(TAXA_URL, stream=True, timeout=120) as r:
        r.raise_for_status()
        gz = gzip.GzipFile(fileobj=r.raw)
        yield from csv.DictReader(
            io.TextIOWrapper(gz, encoding="utf-8", errors="replace"), delimiter="\t")


def main():
    rows = list(csv.DictReader(open(MANIFEST)))
    wanted = {r["taxon_id"] for r in rows if r["taxon_id"]}
    print(f"manifest: {len(rows):,} images, {len({r['species'] for r in rows}):,} species")

    print("resolving phyla from the iNaturalist taxonomy…")
    ancestry, phyla = {}, {}
    for t in stream_taxa():
        if t["rank"] == "phylum":
            phyla[t["taxon_id"]] = t["name"]
        if t["taxon_id"] in wanted:
            ancestry[t["taxon_id"]] = t["ancestry"] or ""

    def phylum_of(taxon_id):
        for p in ancestry.get(taxon_id, "").split("/"):
            if p in phyla:
                return phyla[p]
        return "unknown"

    kept, dropped = [], []
    dropped_species, dropped_by_phylum = set(), Counter()
    for r in rows:
        ph = phylum_of(r["taxon_id"]) if r["taxon_id"] else "unknown"
        # Unresolved taxa are KEPT — never silently discard data we can't classify.
        if ph in KEEP_PHYLA or ph == "unknown":
            kept.append(r)
        else:
            dropped.append(r)
            dropped_species.add(r["species"])
            dropped_by_phylum[ph] += 1

    if not os.path.exists(BACKUP):
        shutil.copy(MANIFEST, BACKUP)
    with open(MANIFEST, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(kept)

    kept_species = {r["species"] for r in kept}
    splits = Counter(r["split"] for r in kept)
    print("\n=== dropped (not photographable) ===")
    for ph, n in dropped_by_phylum.most_common():
        print(f"  {n:6,} images  {ph}")
    print(f"  {len(dropped_species):,} species, {len(dropped):,} images total")
    print("\n=== kept ===")
    print(f"  images:  {len(kept):,}")
    print(f"  species: {len(kept_species):,}")
    print(f"  splits:  train {splits['train']:,}  val {splits['val']:,}  "
          f"test {splits['test']:,}")
    print(f"\nfull manifest backed up to {os.path.relpath(BACKUP, ROOT)}")


if __name__ == "__main__":
    main()
