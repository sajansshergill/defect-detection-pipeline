"""EfficientNet classifier for manufacturing defect classification."""

import torch
import torch.nn as nn
import timm


class EfficientNetDefectClassifier(nn.Module):
    """Thin wrapper around a timm EfficientNet backbone."""

    def __init__(
        self,
        num_classes: int = 5,
        dropout: float = 0.3,
        pretrained: bool = True,
        model_name: str = "efficientnet_b0",
    ):
        super().__init__()
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def get_feature_extractor(self):
        return self.model.conv_head


def build_efficientnet(
    num_classes: int = 5,
    dropout: float = 0.3,
    pretrained: bool = True,
) -> EfficientNetDefectClassifier:
    return EfficientNetDefectClassifier(
        num_classes=num_classes,
        dropout=dropout,
        pretrained=pretrained,
    )
