"""Evaluation metrics for long-tailed species classification.

The headline numbers for this project (see GOALS.md): top-k accuracy,
genus-level accuracy, and both reported stratified by head/mid/tail tier so
a good head average can't hide a broken tail. Pure Python — operates on
predicted label lists, so it's testable without torch.
"""
from .species import genus_of


def top_k_accuracy(preds: list[list[str]], truths: list[str], k: int) -> float:
    """Fraction of examples whose true label is in the top-k predictions.
    preds[i] is a ranked list of candidate species for example i."""
    if not truths:
        return 0.0
    hits = sum(1 for cand, y in zip(preds, truths) if y in cand[:k])
    return hits / len(truths)


def genus_accuracy(preds: list[list[str]], truths: list[str]) -> float:
    """Top-1 accuracy at the genus level: the predicted species' genus
    matches the true species' genus. Useful even when the exact species is
    wrong, and the basis for the genus fallback in the app."""
    if not truths:
        return 0.0
    hits = sum(1 for cand, y in zip(preds, truths)
               if cand and genus_of(cand[0]) == genus_of(y))
    return hits / len(truths)


def stratified(preds: list[list[str]], truths: list[str],
               tiers: dict[str, str], k: int = 5) -> dict[str, dict]:
    """Break top-1, top-k, and genus accuracy out by tier.

    tiers maps species -> 'head'|'mid'|'tail'. Returns
    {tier: {n, top1, topk, genus}} plus an 'overall' row. This is the shape
    an ML goal's eval report must produce.
    """
    groups: dict[str, list[int]] = {"head": [], "mid": [], "tail": [], "overall": []}
    for i, y in enumerate(truths):
        groups["overall"].append(i)
        t = tiers.get(y)
        if t in groups:
            groups[t].append(i)

    out = {}
    for tier, idxs in groups.items():
        if not idxs:
            out[tier] = {"n": 0, "top1": None, f"top{k}": None, "genus": None}
            continue
        p = [preds[i] for i in idxs]
        y = [truths[i] for i in idxs]
        out[tier] = {
            "n": len(idxs),
            "top1": round(top_k_accuracy(p, y, 1), 4),
            f"top{k}": round(top_k_accuracy(p, y, k), 4),
            "genus": round(genus_accuracy(p, y), 4),
        }
    return out
