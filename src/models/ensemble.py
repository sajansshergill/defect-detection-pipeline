"""Soft-voting ensemble utilities."""

from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional

import torch
import torch.nn as nn

from src.models.efficientnet import build_efficientnet
from src.models.resnet import build_resnet
from src.models.vit import build_vit


MODEL_BUILDERS = {
    "resnet": build_resnet,
    "efficientnet": build_efficientnet,
    "vit": build_vit,
}


def build_model(
    model_name: str,
    num_classes: int = 5,
    pretrained: bool = True,
    **kwargs,
) -> nn.Module:
    """Build one supported classifier by name."""
    key = model_name.lower()
    if key not in MODEL_BUILDERS:
        raise ValueError(f"Unsupported model '{model_name}'. Choose from {sorted(MODEL_BUILDERS)}")
    return MODEL_BUILDERS[key](num_classes=num_classes, pretrained=pretrained, **kwargs)


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    device: Optional[torch.device] = None,
) -> nn.Module:
    """Load a checkpoint saved by the training script."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    return model.to(device)


class SoftVotingEnsemble(nn.Module):
    """Average model probabilities with optional per-model weights."""

    def __init__(
        self,
        models: Mapping[str, nn.Module],
        weights: Optional[Mapping[str, float]] = None,
    ):
        super().__init__()
        self.outputs_probabilities = True
        if not models:
            raise ValueError("SoftVotingEnsemble requires at least one model.")
        self.models = nn.ModuleDict(models)
        if weights is None:
            weights = {name: 1.0 for name in models}
        weight_tensor = torch.tensor(
            [float(weights.get(name, 1.0)) for name in self.models.keys()],
            dtype=torch.float32,
        )
        self.register_buffer("weights", weight_tensor / weight_tensor.sum())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        probs = []
        for model in self.models.values():
            probs.append(torch.softmax(model(x), dim=1))
        stacked = torch.stack(probs, dim=0)
        weighted = stacked * self.weights.view(-1, 1, 1)
        return weighted.sum(dim=0)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x).argmax(dim=1)


def load_ensemble(
    checkpoint_paths: Mapping[str, str],
    num_classes: int = 5,
    device: Optional[torch.device] = None,
    weights: Optional[Mapping[str, float]] = None,
) -> SoftVotingEnsemble:
    """Build and load an ensemble from {model_name: checkpoint_path}."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded: Dict[str, nn.Module] = {}
    for name, path in checkpoint_paths.items():
        if not Path(path).exists():
            raise FileNotFoundError(f"Checkpoint not found for {name}: {path}")
        model = build_model(name, num_classes=num_classes, pretrained=False)
        loaded[name] = load_checkpoint(model, path, device).eval()
    return SoftVotingEnsemble(loaded, weights=weights).to(device).eval()


def parse_checkpoint_args(items: Optional[Iterable[str]]) -> Dict[str, str]:
    """Parse CLI entries like resnet=artifacts/resnet_best.pt."""
    parsed: Dict[str, str] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Expected checkpoint argument NAME=PATH, got '{item}'")
        name, path = item.split("=", 1)
        parsed[name.strip()] = path.strip()
    return parsed
