"""Vision Transformer classifier for defect classification."""

import torch
import torch.nn as nn
import timm


class ViTDefectClassifier(nn.Module):
    """Thin wrapper around a timm ViT backbone."""

    def __init__(
        self,
        num_classes: int = 5,
        drop_path_rate: float = 0.1,
        pretrained: bool = True,
        model_name: str = "vit_base_patch16_224",
    ):
        super().__init__()
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_path_rate=drop_path_rate,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def get_feature_extractor(self):
        blocks = getattr(self.model, "blocks", None)
        return blocks[-1] if blocks is not None else self.model


def build_vit(
    num_classes: int = 5,
    drop_path_rate: float = 0.1,
    pretrained: bool = True,
) -> ViTDefectClassifier:
    return ViTDefectClassifier(
        num_classes=num_classes,
        drop_path_rate=drop_path_rate,
        pretrained=pretrained,
    )
