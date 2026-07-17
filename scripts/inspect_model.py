"""Answer the question 'can I see and evaluate the pretrained model?' — yes.

Loads a pretrained backbone, prints its architecture and parameter counts,
then swaps in a CT-sized classification head to show what fine-tuning sets
up. Run: .venv/bin/python scripts/inspect_model.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import timm  # noqa: E402

BACKBONE = "tf_efficientnetv2_s"
CT_SPECIES = 2000  # stand-in for the comprehensive CT checklist size


def main():
    # Try real pretrained weights; fall back to random init (architecture is
    # identical) if there's no network for the download.
    try:
        model = timm.create_model(BACKBONE, pretrained=True)
        weights = "pretrained (real weights downloaded)"
    except Exception as e:
        model = timm.create_model(BACKBONE, pretrained=False)
        weights = f"random init (no download: {type(e).__name__})"

    total = sum(p.numel() for p in model.parameters())
    head = model.get_classifier()
    print(f"backbone: {BACKBONE}  [{weights}]")
    print(f"total parameters: {total:,}")
    print(f"original classifier head: {head}")
    print(f"  -> maps {head.in_features} features to {head.out_features} classes")

    # This is the fine-tuning move: same backbone, new head sized to CT.
    ct_model = timm.create_model(BACKBONE, pretrained=False, num_classes=CT_SPECIES)
    ct_head = ct_model.get_classifier()
    print(f"\nafter swapping the head for {CT_SPECIES} CT species:")
    print(f"  -> maps {ct_head.in_features} features to {ct_head.out_features} classes")

    print("\nfirst layers of the architecture (fully inspectable):")
    print("\n".join(str(model).splitlines()[:12]))


if __name__ == "__main__":
    main()
