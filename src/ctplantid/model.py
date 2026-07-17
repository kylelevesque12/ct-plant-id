"""Load a pretrained backbone and adapt it to the CT species set.

timm is imported lazily inside the functions so the rest of the package
(and the unit tests) don't require torch to be installed. This is the file
that does the fine-tuning setup: keep the pretrained feature extractor,
swap the classification head for one sized to the CT checklist.
"""

# A good default: EfficientNetV2-S is a strong accuracy/size trade-off and
# has iNaturalist-pretrained checkpoints available. Swap freely.
DEFAULT_BACKBONE = "tf_efficientnetv2_s"


def build_model(num_classes: int, backbone: str = DEFAULT_BACKBONE,
                pretrained: bool = True):
    """Return a timm model with its head resized to `num_classes` CT species.

    The backbone weights come pretrained; timm's `num_classes` argument
    replaces the final classifier so it outputs CT species instead of the
    backbone's original classes. Freeze the backbone or train it at a small
    LR during fine-tuning (see GOALS.md phase 2).
    """
    import timm
    return timm.create_model(backbone, pretrained=pretrained,
                             num_classes=num_classes)


def describe(model) -> dict:
    """Human-readable summary of a model: parameter counts and the shape of
    its classifier head. Handy for confirming the head was swapped."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    head = model.get_classifier()
    out_features = getattr(head, "out_features", None)
    return {
        "total_params": total,
        "trainable_params": trainable,
        "classifier_out_features": out_features,
    }
