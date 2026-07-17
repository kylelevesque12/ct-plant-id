from ctplantid import species


def test_normalize_name():
    assert species.normalize_name("  Acer   Rubrum ") == "acer rubrum"
    assert species.normalize_name("ACER rubrum") == "acer rubrum"


def test_genus_of():
    assert species.genus_of("Acer rubrum") == "acer"
    assert species.genus_of("  Quercus   Alba ") == "quercus"
    assert species.genus_of("") == ""


def test_tier_thresholds():
    # head at/above HEAD_MIN, tail below TAIL_MAX, mid between.
    assert species.tier_for_count(species.HEAD_MIN) == "head"
    assert species.tier_for_count(species.HEAD_MIN + 500) == "head"
    assert species.tier_for_count(species.TAIL_MAX - 1) == "tail"
    assert species.tier_for_count(species.TAIL_MAX) == "mid"
    assert species.tier_for_count(species.HEAD_MIN - 1) == "mid"


def test_assign_and_summary():
    counts = {"a b": 500, "c d": 40, "e f": 3, "g h": 5}
    tiers = species.assign_tiers(counts)
    assert tiers == {"a b": "head", "c d": "mid", "e f": "tail", "g h": "tail"}
    assert species.tier_summary(counts) == {"head": 1, "mid": 1, "tail": 2}
