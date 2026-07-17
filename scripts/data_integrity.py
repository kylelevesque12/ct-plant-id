"""Data-integrity gate for the CT Plant ID dataset.

Asserts invariants on the real data artifacts (not just the split *logic*,
which the unit tests already cover). Run by scripts/goal_check.sh as the
Stop-hook metric gate for data-phase goals. Passes cleanly when data files
are absent (fresh clone), so it never blocks work that hasn't downloaded yet.

Exit 0 = all present artifacts are sound; exit 1 = a real violation.
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHECKLIST = ROOT / "data" / "ct_checklist.csv"
MANIFEST = ROOT / "data" / "manifest.csv"


def fail(msg):
    print(f"FAIL: {msg}")
    sys.exit(1)


def main():
    checks = []

    if CHECKLIST.exists():
        rows = list(csv.DictReader(open(CHECKLIST)))
        if len(rows) < 1000:
            fail(f"checklist has only {len(rows)} species — expected the full "
                 f"CT flora (~2,500)")
        checklist_species = {r["name"] for r in rows}
        checks.append(f"checklist: {len(rows)} species")
    else:
        checklist_species = None
        checks.append("checklist: absent (skipped)")

    if MANIFEST.exists():
        rows = list(csv.DictReader(open(MANIFEST)))
        if not rows:
            fail("manifest is empty")

        # 1. Every image path actually exists.
        missing = [r["path"] for r in rows if not (ROOT / r["path"]).exists()]
        if missing:
            fail(f"{len(missing)} manifest paths missing on disk, e.g. {missing[0]}")

        # 2. No observation straddles splits — the leakage guard, on real data.
        by_obs = defaultdict(set)
        for r in rows:
            by_obs[r["observation_id"]].add(r["split"])
        straddlers = {o for o, s in by_obs.items() if len(s) > 1}
        if straddlers:
            fail(f"{len(straddlers)} observations span >1 split (leakage), "
                 f"e.g. {next(iter(straddlers))}")

        # 3. Splits are the expected labels.
        bad = {r["split"] for r in rows} - {"train", "val", "test"}
        if bad:
            fail(f"unexpected split labels: {bad}")

        # 4. Manifest species are on the checklist (if we have one).
        if checklist_species is not None:
            orphans = {r["species"] for r in rows} - checklist_species
            if orphans:
                fail(f"manifest has species not in checklist, e.g. "
                     f"{next(iter(orphans))}")

        checks.append(f"manifest: {len(rows)} photos, {len(by_obs)} observations, "
                      f"no leakage")
    else:
        checks.append("manifest: absent (skipped)")

    print("data integrity OK — " + "; ".join(checks))
    sys.exit(0)


if __name__ == "__main__":
    main()
