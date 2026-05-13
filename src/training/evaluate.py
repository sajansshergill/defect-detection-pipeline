"""Evaluate a single model or soft-voting ensemble."""

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import classification_report, confusion_matrix

from src.data.dataset import build_dataloaders
from src.models.ensemble import build_model, load_checkpoint, load_ensemble
from src.models.ensemble import parse_checkpoint_args


def load_params(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@torch.no_grad()
def collect_predictions(model, loader, device: torch.device):
    model.eval()
    labels: List[int] = []
    preds: List[int] = []
    defect_scores: List[float] = []
    all_probs: List[np.ndarray] = []

    for images, targets in loader:
        images = images.to(device)
        outputs = model(images)
        probs = outputs if getattr(model, "outputs_probabilities", False) else outputs.softmax(dim=1)
        labels.extend(targets.numpy().tolist())
        preds.extend(probs.argmax(dim=1).cpu().numpy().tolist())
        defect_scores.extend((1.0 - probs[:, 0]).cpu().numpy().tolist())
        all_probs.extend(probs.cpu().numpy())
    return labels, preds, defect_scores, all_probs


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate defect classifier checkpoints.")
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--model", choices=["resnet", "efficientnet", "vit"])
    parser.add_argument("--checkpoint", help="Checkpoint for single-model evaluation.")
    parser.add_argument("--mode", choices=["single", "ensemble"], default="single")
    parser.add_argument(
        "--checkpoints",
        nargs="*",
        help="Ensemble checkpoints as NAME=PATH, for example resnet=artifacts/resnet.pt",
    )
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--output-dir", default="artifacts/evaluation")
    args = parser.parse_args()

    params = load_params(args.params)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = params["data"]
    loaders = build_dataloaders(
        root=data["root"],
        categories=data["categories"],
        num_classes=params["classes"]["num_classes"],
        img_size=data["img_size"],
        batch_size=params["training"]["batch_size"],
        num_workers=data.get("num_workers", 4),
        train_split=data.get("train_split", 0.7),
        val_split=data.get("val_split", 0.15),
        seed=params["training"].get("seed", 42),
    )
    loader = loaders[1] if args.split == "val" else loaders[2]
    if len(loader.dataset) == 0:
        raise RuntimeError(f"No {args.split} images found under {data['root']}.")

    if args.mode == "ensemble":
        checkpoint_paths = parse_checkpoint_args(args.checkpoints)
        model = load_ensemble(
            checkpoint_paths,
            num_classes=params["classes"]["num_classes"],
            device=device,
        )
        run_name = "ensemble"
    else:
        if not args.model or not args.checkpoint:
            raise ValueError("--model and --checkpoint are required for single mode.")
        model = build_model(
            args.model,
            num_classes=params["classes"]["num_classes"],
            pretrained=False,
        )
        model = load_checkpoint(model, args.checkpoint, device).eval()
        run_name = args.model

    labels, preds, defect_scores, probs = collect_predictions(model, loader, device)
    class_names = params["classes"]["names"]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report = classification_report(
        labels,
        preds,
        labels=list(range(params["classes"]["num_classes"])),
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )
    matrix = confusion_matrix(
        labels,
        preds,
        labels=list(range(params["classes"]["num_classes"])),
    )
    with open(output_dir / f"{run_name}_{args.split}_report.yaml", "w", encoding="utf-8") as fh:
        yaml.safe_dump(report, fh)
    pd.DataFrame(matrix, index=class_names, columns=class_names).to_csv(
        output_dir / f"{run_name}_{args.split}_confusion_matrix.csv"
    )
    pred_df = pd.DataFrame(
        {
            "label": labels,
            "prediction": preds,
            "defect_score": defect_scores,
            "probabilities": [np.asarray(prob).round(6).tolist() for prob in probs],
        }
    )
    pred_df.to_csv(output_dir / f"{run_name}_{args.split}_predictions.csv", index=False)
    print(
        classification_report(
            labels,
            preds,
            labels=list(range(params["classes"]["num_classes"])),
            target_names=class_names,
            zero_division=0,
        )
    )


if __name__ == "__main__":
    main()
