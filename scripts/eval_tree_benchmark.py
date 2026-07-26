"""Option A: replicate the Rutgers tree-ID protocol on our model.

The published independent benchmark (Rutgers Urban Forestry, summarised by
Illinois Extension) tested six apps on 55 common New Jersey street and native
forest tree species and reported accuracy at BOTH genus and species level:

    PictureThis   97.3% genus   83.9% species
    iNaturalist   92.3% genus   69.6% species

This script scores our model the same two ways on the same kind of cohort, so
there is a defensible comparison to a published number without needing any
access to PictureThis.

It also fills a real gap: GOALS.md names genus-level accuracy a headline metric
and nothing in the repo computed it.

Honest framing, printed with the results:
  * The 55 species here are a RECONSTRUCTION of that cohort (the exact list is
    behind a paywall), chosen as common Northeast street + native forest trees.
  * Photos are held-out iNaturalist test-split images, not arborist-taken bark
    and leaf sets. iNat photos skew toward leaves, flowers and whole plants —
    the EASY image types. The study found bark alone drops PictureThis to ~65%
    genus / ~52% species, so this comparison flatters both sides equally but is
    not like-for-like on image type.
  * Species we do not carry are reported, never silently skipped.

Run: .venv/bin/python scripts/eval_tree_benchmark.py
Writes reports/tree_benchmark.json
"""
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
HEADERS = {"User-Agent": "ct-plant-id/0.1 (tree benchmark)"}
CKPT = os.path.join(ROOT, "runs", "b_stage2", "model.pt")
OUT = os.path.join(ROOT, "reports", "tree_benchmark.json")

PER_SPECIES = 12      # test photos per species
MAX_SCAN = 300        # observations to scan per species to find held-out ones

# Published reference numbers (Rutgers via Illinois Extension).
REFERENCE = {
    "PictureThis": {"genus": 0.973, "species": 0.839},
    "iNaturalist": {"genus": 0.923, "species": 0.696},
}

# A reconstruction of the study's cohort: common Northeast street and native
# forest trees. Native forest species first, then commonly planted street trees.
TREE_COHORT = [
    # native forest
    "Acer rubrum", "Acer saccharum", "Quercus rubra", "Quercus alba",
    "Quercus palustris", "Quercus velutina", "Fagus grandifolia",
    "Betula lenta", "Betula alleghaniensis", "Betula papyrifera",
    "Carya ovata", "Carya glabra", "Liriodendron tulipifera",
    "Liquidambar styraciflua", "Nyssa sylvatica", "Platanus occidentalis",
    "Fraxinus americana", "Tilia americana", "Ulmus americana",
    "Prunus serotina", "Sassafras albidum", "Juglans nigra",
    "Populus deltoides", "Pinus strobus", "Tsuga canadensis",
    "Cornus florida", "Cercis canadensis", "Castanea dentata",
    "Robinia pseudoacacia", "Ostrya virginiana", "Carpinus caroliniana",
    "Amelanchier canadensis", "Acer saccharinum", "Quercus bicolor",
    "Pinus rigida",
    # commonly planted street / ornamental trees
    "Ginkgo biloba", "Zelkova serrata", "Pyrus calleryana",
    "Acer platanoides", "Acer palmatum", "Tilia cordata",
    "Quercus acutissima", "Prunus serrulata", "Gleditsia triacanthos",
    "Catalpa speciosa", "Picea abies", "Pinus nigra",
    "Ailanthus altissima", "Ulmus parvifolia", "Koelreuteria paniculata",
    "Cladrastis kentukea", "Magnolia grandiflora", "Cryptomeria japonica",
    "Metasequoia glyptostroboides", "Malus floribunda",
]


def split_for(uuid, val=0.15, test=0.15):
    """Same observation-keyed hash split as training — reconstructs the held-out
    test set so nothing scored here was learned from."""
    import hashlib
    h = int(hashlib.md5(str(uuid).encode()).hexdigest(), 16) % 10000 / 10000
    return "test" if h < test else "val" if h < test + val else "train"


def genus_of(name):
    return name.split()[0] if name else ""


def _get(url, params, tries=4):
    """GET with retry/backoff — a transient timeout shouldn't kill a long run."""
    for attempt in range(tries):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=60)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt == tries - 1:
                print(f"    request failed after {tries} tries ({type(e).__name__})")
                return None
            time.sleep(3 * (attempt + 1))


def taxon_id_for(name):
    payload = _get(f"{API}/taxa", {"q": name, "rank": "species", "per_page": 5})
    if not payload:
        return None
    for r in payload.get("results", []):
        if r.get("name", "").lower() == name.lower():
            return r["id"]
    return None


def test_photos(taxon_id, cap):
    """Held-out test-split photos for a taxon. Tries research-grade first, then
    cultivated (captive) observations — planted street trees are often logged
    as captive, which is exactly how the ornamental scope was trained."""
    imgs, page, scanned = [], 1, 0
    for quality in ({"quality_grade": "research"}, {"captive": "true"}):
        page, scanned = 1, 0
        while len(imgs) < cap and scanned < MAX_SCAN:
            params = {"taxon_id": taxon_id, "photos": "true", "order_by": "votes",
                      "per_page": 50, "page": page, **quality}
            payload = _get(f"{API}/observations", params)
            if not payload:
                break
            results = payload.get("results", [])
            if not results:
                break
            for obs in results:
                scanned += 1
                uuid = obs.get("uuid") or str(obs["id"])
                if split_for(uuid) != "test":
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
        if len(imgs) >= cap:
            break
    return imgs


def main():
    model = PlantModel(CKPT, device="cpu")
    in_scope = [s for s in TREE_COHORT if s in set(model.classes)]
    missing = [s for s in TREE_COHORT if s not in set(model.classes)]

    print(f"cohort: {len(TREE_COHORT)} species | in model scope: {len(in_scope)} "
          f"| NOT carried: {len(missing)}")
    if missing:
        print("  not in scope (counted as failures in the honest total):")
        for s in missing:
            print(f"    - {s}")

    records = []
    for i, species in enumerate(in_scope):
        tid = taxon_id_for(species)
        if tid is None:
            print(f"  [{i+1}/{len(in_scope)}] {species}: no taxon id, skipped")
            continue
        imgs = test_photos(tid, PER_SPECIES)
        for im in imgs:
            preds = model.identify(im, k=5)
            names = [p["species"] for p in preds["candidates"]]
            records.append({
                "true_species": species,
                "pred_species": names[0],
                "species_top1": int(names[0] == species),
                "species_top5": int(species in names),
                "genus_top1": int(genus_of(names[0]) == genus_of(species)),
                "genus_top5": int(any(genus_of(n) == genus_of(species) for n in names)),
                "out_of_scope": bool(preds["out_of_scope"]),
            })
        print(f"  [{i+1}/{len(in_scope)}] {species}: {len(imgs)} photos "
              f"(total {len(records)})")

    if not records:
        raise SystemExit("no photos collected — check network / API")

    n = len(records)
    def rate(key):
        return sum(r[key] for r in records) / n

    scored = {
        "species_top1": rate("species_top1"),
        "species_top5": rate("species_top5"),
        "genus_top1": rate("genus_top1"),
        "genus_top5": rate("genus_top5"),
    }

    # Honest total: species we don't carry can never be right, so also report
    # accuracy penalised for the missing cohort members.
    coverage = len(in_scope) / len(TREE_COHORT)
    penalised = {k: v * coverage for k, v in scored.items()}

    print(f"\n=== tree benchmark (n={n} photos, {len(in_scope)} species) ===")
    print(f"{'metric':<16} {'scored':>8} {'coverage-penalised':>20}")
    for k in scored:
        print(f"{k:<16} {scored[k]:>7.1%} {penalised[k]:>19.1%}")
    print(f"\nspecies coverage: {len(in_scope)}/{len(TREE_COHORT)} = {coverage:.1%}")

    print("\n=== published reference (Rutgers, 55 NJ trees, bark+leaf photos) ===")
    for app, ref in REFERENCE.items():
        print(f"  {app:<14} genus {ref['genus']:.1%}   species {ref['species']:.1%}")
    print(f"  {'Fieldnote':<14} genus {scored['genus_top1']:.1%}   "
          f"species {scored['species_top1']:.1%}   (top-1, this cohort)")

    print("\nCAVEATS — quote these with any comparison:")
    print("  * different photos (iNat crowd-sourced vs arborist-taken bark+leaf sets)")
    print("  * iNat photos skew to leaves/flowers/whole plant = the EASY image types;")
    print("    the study found bark alone drops PictureThis to ~65% genus / ~52% species")
    print("  * cohort is a reconstruction of the study's 55 species, not the exact list")
    print("  * our model chooses among 2,510 classes; a 55-way task is far easier")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump({
        "n_photos": n,
        "cohort_size": len(TREE_COHORT),
        "species_in_scope": len(in_scope),
        "species_missing": missing,
        "coverage": coverage,
        "scored": scored,
        "coverage_penalised": penalised,
        "reference": REFERENCE,
        "per_species": _per_species(records),
    }, open(OUT, "w"), indent=2)
    print(f"\nwrote {os.path.relpath(OUT, ROOT)}")


def _per_species(records):
    by = defaultdict(list)
    for r in records:
        by[r["true_species"]].append(r)
    out = {}
    for sp, rs in by.items():
        out[sp] = {
            "n": len(rs),
            "species_top1": sum(r["species_top1"] for r in rs) / len(rs),
            "genus_top1": sum(r["genus_top1"] for r in rs) / len(rs),
        }
    return out


if __name__ == "__main__":
    main()
