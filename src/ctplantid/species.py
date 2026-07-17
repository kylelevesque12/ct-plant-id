"""CT species checklist helpers: name normalization, taxonomy, and the
head/mid/tail tiers that structure every metric in this project.

Deliberately dependency-free (pure Python) so it's fast to test and usable
in the data pipeline before any ML libraries are installed.
"""
from collections import Counter

# Image-count thresholds that sort species into tiers. Provisional — retune
# once the real iNaturalist counts are in (see GOALS.md phase 1).
HEAD_MIN = 100   # >= this many images: well-sampled "head"
TAIL_MAX = 20    # < this many images: sparse "tail"


def normalize_name(name: str) -> str:
    """Canonical form for matching: trimmed, lowercased, single-spaced.
    'Acer  Rubrum ' and 'acer rubrum' collapse to the same key."""
    return " ".join(name.strip().lower().split())


def genus_of(species: str) -> str:
    """Genus = the first token of a binomial name. 'Acer rubrum' -> 'acer'.
    Genus-level fallback uses this when species is uncertain."""
    norm = normalize_name(species)
    return norm.split(" ")[0] if norm else ""


def tier_for_count(count: int) -> str:
    """Which data-availability tier a species falls in, from its image count."""
    if count >= HEAD_MIN:
        return "head"
    if count < TAIL_MAX:
        return "tail"
    return "mid"


def assign_tiers(counts: dict[str, int]) -> dict[str, str]:
    """Map every species -> its tier. Input is {species: image_count}."""
    return {sp: tier_for_count(n) for sp, n in counts.items()}


def tier_summary(counts: dict[str, int]) -> dict[str, int]:
    """How many species sit in each tier — the head/mid/tail shape of the
    dataset that the data card reports."""
    return dict(Counter(tier_for_count(n) for n in counts.values()))
