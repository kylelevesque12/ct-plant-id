"""Per-candidate attribute lookup: common name, CT status, weed flag.

`annotate(species)` is called once per candidate on every /api/identify request,
so all data is loaded ONCE at import time into in-memory dicts and every lookup
is an O(1) dict hit on the normalized Latin name — no network, no file I/O, no
work at request time.

Contract (consumed by app/main.py):

    annotate("Alliaria petiolata")
    # -> {"common_name": "garlic mustard" | None,
    #     "status": "native" | "introduced" | "invasive" | "ornamental" | "unknown",
    #     "is_weed": bool}

Design choices:
  - Match is case- and whitespace-insensitive on the species binomial.
  - Any species not in the data returns the honest defaults
    ({"common_name": None, "status": "unknown", "is_weed": False}); annotate
    never raises.
  - status is deliberately conservative. We assert "invasive" only for the
    official Connecticut Invasive Plant List (CIPWG / CGS 22a-381d) and
    "introduced" only for a hand-verified set of unambiguously non-native lawn
    weeds. For the long tail we say "unknown" rather than guess native vs.
    introduced — a wrong "native" label is worse than an honest "unknown".
    "ornamental" (a planted garden/landscape plant) is likewise asserted only
    for the curated cultivated scope added in Workstream B, where cultivation
    is a known fact rather than an inference — and never overrides "invasive".

Data files (built offline, loaded here):
  - data/attributes.csv     species, status, is_weed, cipwg_category, source
  - data/common_names.csv   species, common_name   (from iNaturalist)
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional, TypedDict

_ROOT = Path(__file__).resolve().parent.parent
_ATTRIBUTES_CSV = _ROOT / "data" / "attributes.csv"
_COMMON_NAMES_CSV = _ROOT / "data" / "common_names.csv"

_VALID_STATUS = {"native", "introduced", "invasive", "ornamental", "unknown"}


class Annotation(TypedDict):
    common_name: Optional[str]
    status: str
    is_weed: bool
    hazard: Optional[str]


# Group nouns for a genus ("oak", "maple") are DERIVED from the last word of its
# species' common names — 'northern red oak' and 'white oak' both end in "oak".
# Only accepted when the species agree, so a genus with mixed names falls back
# to its scientific name rather than inventing a misleading one.
_GENUS_NOUN_MIN_AGREEMENT = 0.6


def _normalize(species: str) -> str:
    """Canonical key: lowercase, collapsed internal whitespace, stripped."""
    return " ".join(species.split()).lower()


def _load_common_names() -> dict[str, str]:
    names: dict[str, str] = {}
    if not _COMMON_NAMES_CSV.exists():
        return names
    with open(_COMMON_NAMES_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            species = (row.get("species") or "").strip()
            common = (row.get("common_name") or "").strip()
            if species and common:
                names[_normalize(species)] = common
    return names


def _load_status() -> tuple[dict[str, tuple[str, bool]], dict[str, str]]:
    status: dict[str, tuple[str, bool]] = {}
    hazards: dict[str, str] = {}
    if not _ATTRIBUTES_CSV.exists():
        return status, hazards
    with open(_ATTRIBUTES_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            species = (row.get("species") or "").strip()
            if not species:
                continue
            st = (row.get("status") or "unknown").strip().lower()
            if st not in _VALID_STATUS:
                st = "unknown"
            is_weed = (row.get("is_weed") or "").strip().lower() in {"true", "1", "yes"}
            status[_normalize(species)] = (st, is_weed)
            hazard = (row.get("hazard") or "").strip()
            if hazard:
                hazards[_normalize(species)] = hazard
    return status, hazards


def _load_genus_nouns(names: dict[str, str]) -> dict[str, str]:
    """Map genus -> group noun ("Quercus" -> "oak"), derived from its species'
    common names. Requires agreement so mixed genera keep their Latin name."""
    from collections import Counter, defaultdict

    by_genus: dict[str, list[str]] = defaultdict(list)
    for key, common in names.items():
        genus = key.split()[0] if key else ""
        last = common.strip().lower().split()
        if genus and last:
            by_genus[genus].append(last[-1])

    nouns: dict[str, str] = {}
    for genus, words in by_genus.items():
        word, count = Counter(words).most_common(1)[0]
        if len(words) == 1 or count / len(words) >= _GENUS_NOUN_MIN_AGREEMENT:
            nouns[genus] = word
    return nouns


# Loaded once at import — never re-read per request.
_COMMON_NAMES: dict[str, str] = _load_common_names()
_STATUS, _HAZARDS = _load_status()
_GENUS_NOUNS: dict[str, str] = _load_genus_nouns(_COMMON_NAMES)


def genus_label(genus: str) -> str:
    """Human-facing name for a genus: its group noun when the species agree on
    one ("oak"), otherwise the scientific name itself."""
    if not genus:
        return ""
    return _GENUS_NOUNS.get(_normalize(genus), genus)


def annotate(species: str) -> Annotation:
    """Return common name + CT status + weed flag for a Latin species name.

    Safe for any input: unknown species, typos, empty strings, or None all yield
    the honest default annotation instead of raising.
    """
    if not species:
        return {"common_name": None, "status": "unknown", "is_weed": False,
                "hazard": None}

    key = _normalize(species)
    status, is_weed = _STATUS.get(key, ("unknown", False))
    common_name = _COMMON_NAMES.get(key) or None
    return {"common_name": common_name, "status": status, "is_weed": is_weed,
            "hazard": _HAZARDS.get(key)}


# Small module-level stats, handy for debugging / health reporting.
def _coverage() -> dict[str, int]:
    invasive = sum(1 for s, _ in _STATUS.values() if s == "invasive")
    introduced = sum(1 for s, _ in _STATUS.values() if s == "introduced")
    return {
        "common_names": len(_COMMON_NAMES),
        "status_rows": len(_STATUS),
        "invasive": invasive,
        "introduced": introduced,
    }


if __name__ == "__main__":
    print("coverage:", _coverage())
    for s in ["Alliaria petiolata", "Acer rubrum", "Notaplant xyzzy", "", "  alliaria   PETIOLATA "]:
        print(f"{s!r:32} -> {annotate(s)}")
