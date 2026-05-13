"""Grad-CAM generation helpers."""

from typing import Optional, Tuple

import cv2
import numpy as np
import torch

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
except ImportError:  # pragma: no cover - exercised only when optional package is absent
    GradCAM = None
    ClassifierOutputTarget = None
    show_cam_on_image = None

from src.data.augmentations import IMAGENET_MEAN, IMAGENET_STD


def _target_layer(model: torch.nn.Module):
    if hasattr(model, "get_feature_extractor"):
        return model.get_feature_extractor()
    wrapped = getattr(model, "model", None)
    if wrapped is not None and hasattr(wrapped, "layer4"):
        return wrapped.layer4
    raise ValueError("Could not infer target layer. Pass target_layer explicitly.")


def denormalize_image(image_tensor: torch.Tensor) -> np.ndarray:
    """Convert a normalized CHW tensor into an RGB float image in [0, 1]."""
    image = image_tensor.detach().cpu().numpy().transpose(1, 2, 0)
    image = image * np.asarray(IMAGENET_STD) + np.asarray(IMAGENET_MEAN)
    return np.clip(image, 0.0, 1.0).astype(np.float32)


def generate_gradcam(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    target_class: Optional[int] = None,
    target_layer: Optional[torch.nn.Module] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (heatmap, overlay) for one normalized image tensor."""
    if GradCAM is None:
        raise ImportError("Install grad-cam to use Grad-CAM explanations.")
    model.eval()
    if image_tensor.ndim == 3:
        input_tensor = image_tensor.unsqueeze(0)
    else:
        input_tensor = image_tensor
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)

    if target_class is None:
        with torch.no_grad():
            target_class = int(model(input_tensor).argmax(dim=1).item())

    cam = GradCAM(model=model, target_layers=[target_layer or _target_layer(model)])
    grayscale_cam = cam(
        input_tensor=input_tensor,
        targets=[ClassifierOutputTarget(target_class)],
    )[0]
    rgb_image = denormalize_image(input_tensor[0])
    overlay = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)
    return grayscale_cam, overlay


def save_gradcam_overlay(overlay: np.ndarray, output_path: str) -> None:
    """Save an RGB Grad-CAM overlay to disk."""
    cv2.imwrite(output_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
