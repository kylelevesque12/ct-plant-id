from collections import defaultdict

from ctplantid import splits


def test_split_is_deterministic():
    a = splits.split_for_observation(12345)
    b = splits.split_for_observation(12345)
    assert a == b
    assert a in {"train", "val", "test"}


def test_no_observation_straddles_splits():
    # Several photos per observation; the whole point is they never split.
    records = []
    for obs in range(500):
        for photo in range(4):  # 4 photos each
            records.append({"observation_id": obs, "photo_id": f"{obs}-{photo}"})
    assigned = splits.assign_splits(records)

    by_obs = defaultdict(set)
    for r in assigned:
        by_obs[r["observation_id"]].add(r["split"])
    assert all(len(s) == 1 for s in by_obs.values()), \
        "an observation's photos landed in more than one split — leakage!"


def test_all_three_splits_populated_and_roughly_sized():
    records = [{"observation_id": i, "photo_id": i} for i in range(3000)]
    counts = splits.split_counts(splits.assign_splits(records))
    assert counts["train"] > 0 and counts["val"] > 0 and counts["test"] > 0
    # ~70/15/15 with tolerance for hashing noise.
    assert 0.60 < counts["train"] / 3000 < 0.80
    assert 0.08 < counts["test"] / 3000 < 0.22


def test_new_data_does_not_reshuffle_existing():
    # Stability: an id's split doesn't change when more data is added.
    before = splits.split_for_observation(999999)
    _ = splits.assign_splits([{"observation_id": i} for i in range(10000)])
    after = splits.split_for_observation(999999)
    assert before == after
