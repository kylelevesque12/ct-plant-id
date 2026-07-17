from ctplantid import metrics


PREDS = [
    ["acer rubrum", "acer saccharum", "quercus alba"],  # truth top-1
    ["quercus alba", "quercus rubra", "acer rubrum"],    # truth at rank 2
    ["betula lenta", "acer rubrum", "quercus alba"],     # truth not in top-3
]
TRUTHS = ["acer rubrum", "quercus rubra", "fagus grandifolia"]


def test_top_k_accuracy():
    assert metrics.top_k_accuracy(PREDS, TRUTHS, 1) == 1 / 3   # only ex 0
    assert metrics.top_k_accuracy(PREDS, TRUTHS, 3) == 2 / 3   # ex 0 and 1
    assert metrics.top_k_accuracy([], [], 5) == 0.0


def test_genus_accuracy():
    # ex0 acer==acer hit; ex1 pred genus quercus==quercus hit; ex2 betula!=fagus.
    assert metrics.genus_accuracy(PREDS, TRUTHS) == 2 / 3


def test_stratified_reports_each_tier():
    tiers = {"acer rubrum": "head", "quercus rubra": "mid",
             "fagus grandifolia": "tail"}
    report = metrics.stratified(PREDS, TRUTHS, tiers, k=3)
    assert report["overall"]["n"] == 3
    assert report["head"]["n"] == 1 and report["head"]["top1"] == 1.0
    assert report["mid"]["top1"] == 0.0 and report["mid"]["top3"] == 1.0
    assert report["tail"]["top3"] == 0.0
    # A tier with no members reports n=0, not a crash.
    empty = metrics.stratified(PREDS, TRUTHS, {}, k=3)
    assert empty["head"]["n"] == 0
