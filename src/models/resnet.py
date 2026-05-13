"""
ResNet-50 fine-tuned for defect classification.
Captures local textire features - good baseline for scratch/crack detection.
"""

import torch.nn as nn
import torch
from torchvision import models


class ResNetDefectClassifier(nn.Module):
    def __init__(self, num_classes: int = 5, dropout: float = 0.3, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = models.resnet50(weights=weights)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )
        self.model = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def get_feature_extractor(self):
        return self.model.layer4


def build_resnet(
    num_classes: int = 5,
    dropout: float = 0.3,
    pretrained: bool = True,
) -> ResNetDefectClassifier:
    return ResNetDefectClassifier(
        num_classes=num_classes,
        dropout=dropout,
        pretrained=pretrained,
    )