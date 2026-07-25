"""Give the ~150 newly-in-scope garden species their common names and status.

After the Workstream B scope expansion the model can name ornamentals, but the
attribute lookup had never heard of them — so the app showed a bare Latin
binomial and "Status unknown". Two fixes, both from data we already have:

  1. common names — build_ornamental_scope.py already pulled them from iNat and
     stored them in data/ornamental_species.csv.
  2. status "ornamental" — these species came from an explicitly CULTIVATED
     (captive=true) query and a curated garden list, so "planted ornamental" is
     a fact we know, not a native/introduced guess. It's also the more useful
     answer: it tells the user this is a garden plant, not a weed to pull.

SAFETY: never downgrade an existing status. Some garden plants are also CT
invasives (Japanese barberry, burning bush); if attributes.csv already flags a
species invasive/weed, that flag wins and we leave the row untouched.

Run: .venv/bin/python scripts/merge_ornamental_attributes.py --write
"""
import argparse
import csv
import os
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCOPE = os.path.join(ROOT, "data", "ornamental_species.csv")
NAMES = os.path.join(ROOT, "data", "common_names.csv")
ATTRS = os.path.join(ROOT, "data", "attributes.csv")


def read(path):
    with open(path) as f:
        r = csv.DictReader(f)
        return r.fieldnames, list(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="apply (default: dry run)")
    args = ap.parse_args()

    orn = list(csv.DictReader(open(SCOPE)))
    name_cols, names = read(NAMES)
    attr_cols, attrs = read(ATTRS)

    have_name = {r["species"] for r in names}
    existing = {r["species"]: r for r in attrs}

    new_names = [{"species": o["species"], "common_name": o["common_name"]}
                 for o in orn if o["species"] not in have_name and o["common_name"]]

    new_attrs, protected = [], []
    for o in orn:
        sp = o["species"]
        if sp in existing:
            protected.append((sp, existing[sp]["status"]))   # keep invasive etc.
            continue
        new_attrs.append({"species": sp, "status": "ornamental", "is_weed": "false",
                          "cipwg_category": "", "source": "iNat cultivated scope (WS-B)"})

    print(f"ornamental species in scope:      {len(orn)}")
    print(f"common names to add:              {len(new_names)}")
    print(f"status rows to add ('ornamental'):{len(new_attrs)}")
    print(f"already had a status (untouched): {len(protected)}")
    for sp, st in protected:
        print(f"    KEEPING {st:<10} {sp}")

    if not args.write:
        print("\ndry run — re-run with --write to apply")
        return

    shutil.copy(NAMES, NAMES + ".bak")
    shutil.copy(ATTRS, ATTRS + ".bak")
    with open(NAMES, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=name_cols)
        w.writeheader()
        w.writerows(sorted(names + new_names, key=lambda r: r["species"]))
    with open(ATTRS, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=attr_cols)
        w.writeheader()
        w.writerows(sorted(attrs + new_attrs, key=lambda r: r["species"]))
    print(f"\nwrote {len(names) + len(new_names)} common names, "
          f"{len(attrs) + len(new_attrs)} attribute rows (.bak backups kept)")


if __name__ == "__main__":
    main()
