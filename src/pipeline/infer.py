"""Run inference for one image or a directory of images."""

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch
import yaml
from PIL import Image

from src.data.augmentations import get_val_transforms
from src.models.ensemble import build_model, load_checkpoint, load_ensemble
from src.models.ensemble import parse_checkpoint_args


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_params(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def iter_images(path: str) -> Iterable[Path]:
    input_path = Path(path)
    if input_path.is_file():
        yield input_path
        return
    for image_path in sorted(input_path.rglob("*")):
        if image_path.suffix.lower() in IMAGE_EXTENSIONS:
            yield image_path


def preprocess(image_path: Path, img_size: int) -> torch.Tensor:
    image = np.asarray(Image.open(image_path).convert("RGB"))
    tensor = get_val_transforms(img_size)(image=image)["image"]
    return tensor.unsqueeze(0)


@torch.no_grad()
def predict_image(
    model,
    image_path: Path,
    class_names: List[str],
    img_size: int,
    threshold: float,
    device: torch.device,
) -> Dict:
    tensor = preprocess(image_path, img_size).to(device)
    outputs = model(tensor)
    probs = outputs if getattr(model, "outputs_probabilities", False) else outputs.softmax(dim=1)
    probs = probs.squeeze(0).cpu().numpy()
    predicted_idx = int(probs.argmax())
    defect_score = float(1.0 - probs[0])
    if defect_score < threshold:
        predicted_idx = 0
    return {
        "image": str(image_path),
        "prediction": class_names[predicted_idx],
        "prediction_id": predicted_idx,
        "defect_score": defect_score,
        "probabilities": {
            class_name: float(prob)
            for class_name, prob in zip(class_names, probs)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run defect inference.")
    parser.add_argument("--input", required=True, help="Image file or directory.")
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--model", choices=["resnet", "efficientnet", "vit"])
    parser.add_argument("--checkpoint", help="Single-model checkpoint.")
    parser.add_argument("--checkpoints", nargs="*", help="Ensemble NAME=PATH checkpoint entries.")
    parser.add_argument("--mode", choices=["single", "ensemble"], default="single")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--output", default="artifacts/inference/predictions.json")
    args = parser.parse_args()

    params = load_params(args.params)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = params["classes"]["names"]
    img_size = int(params["data"]["img_size"])

    if args.mode == "ensemble":
        model = load_ensemble(
            parse_checkpoint_args(args.checkpoints),
            num_classes=params["classes"]["num_classes"],
            device=device,
        )
    else:
        if not args.model or not args.checkpoint:
            raise ValueError("--model and --checkpoint are required in single mode.")
        model = build_model(
            args.model,
            num_classes=params["classes"]["num_classes"],
            pretrained=False,
        )
        model = load_checkpoint(model, args.checkpoint, device).eval()

    threshold = args.threshold
    if threshold is None:
        threshold = float(params["ensemble"].get("threshold", 0.5))

    predictions = [
        predict_image(model, path, class_names, img_size, threshold, device)
        for path in iter_images(args.input)
    ]
    if not predictions:
        raise RuntimeError(f"No images found at {args.input}")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(predictions, fh, indent=2)
    print(json.dumps(predictions, indent=2))


if __name__ == "__main__":
    main()
