"""Observation-keyed train/val/test splitting — the leakage guard.

All photos from one iNaturalist observation must land in the SAME split, or
near-identical photos of the same plant leak across train and test and
inflate accuracy (see docs). The assignment is deterministic from the
observation id (hashed), so it's reproducible AND stable as new data
arrives: adding observations never reshuffles the ones already placed.
"""
import hashlib

_BUCKETS = 10_000


def split_for_observation(obs_id, val_frac: float = 0.15,
                          test_frac: float = 0.15) -> str:
    """Deterministically map an observation id -> 'train' | 'val' | 'test'.

    Hash the id into [0, 1); the first `test_frac` is test, the next
    `val_frac` is val, the rest is train. Same id always lands the same
    place, independent of dataset size or insertion order."""
    h = int(hashlib.md5(str(obs_id).encode()).hexdigest(), 16) % _BUCKETS
    frac = h / _BUCKETS
    if frac < test_frac:
        return "test"
    if frac < test_frac + val_frac:
        return "val"
    return "train"


def assign_splits(records: list[dict], val_frac: float = 0.15,
                  test_frac: float = 0.15) -> list[dict]:
    """Return records with a 'split' field set from their 'observation_id'.
    Every record sharing an observation_id gets the same split."""
    return [{**r, "split": split_for_observation(
        r["observation_id"], val_frac, test_frac)} for r in records]


def split_counts(records: list[dict]) -> dict[str, int]:
    counts = {"train": 0, "val": 0, "test": 0}
    for r in records:
        counts[r["split"]] = counts.get(r["split"], 0) + 1
    return counts
