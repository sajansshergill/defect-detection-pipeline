"""Train one defect-classification model."""

import argparse
import random
from pathlib import Path
from typing import Dict, Tuple

import mlflow
import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import accuracy_score, f1_score, recall_score
from tqdm import tqdm

from src.data.dataset import build_dataloaders
from src.models.ensemble import build_model


def load_params(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def model_kwargs(model_name: str, params: Dict) -> Dict:
    config = params["models"].get(model_name, {})
    if model_name in {"resnet", "efficientnet"}:
        return {"dropout": float(config.get("dropout", 0.3))}
    if model_name == "vit":
        return {"drop_path_rate": float(config.get("drop_path_rate", 0.1))}
    return {}


def run_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer = None,
) -> Tuple[float, Dict[str, float]]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    labels = []
    preds = []

    for images, targets in tqdm(loader, leave=False):
        images = images.to(device)
        targets = targets.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, targets)
            if training:
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * images.size(0)
        labels.extend(targets.detach().cpu().numpy().tolist())
        preds.extend(logits.argmax(dim=1).detach().cpu().numpy().tolist())

    denom = max(1, len(loader.dataset))
    metrics = {
        "accuracy": accuracy_score(labels, preds) if labels else 0.0,
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0)
        if labels
        else 0.0,
        "defect_recall": recall_score(
            np.asarray(labels) != 0,
            np.asarray(preds) != 0,
            zero_division=0,
        )
        if labels
        else 0.0,
    }
    return total_loss / denom, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a defect classifier.")
    parser.add_argument("--model", choices=["resnet", "efficientnet", "vit"], required=True)
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--output-dir", default="artifacts/checkpoints")
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()

    params = load_params(args.params)
    set_seed(int(params["training"].get("seed", 42)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = params["data"]
    train_loader, val_loader, _, class_weights = build_dataloaders(
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
    if len(train_loader.dataset) == 0:
        raise RuntimeError(
            f"No training images found under {data['root']}. "
            "Download MVTec AD or update params.yaml:data.root."
        )

    model = build_model(
        args.model,
        num_classes=params["classes"]["num_classes"],
        pretrained=not args.no_pretrained,
        **model_kwargs(args.model, params),
    ).to(device)
    class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    cfg = params["models"][args.model]
    lr = args.lr or float(cfg["lr"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=float(cfg.get("weight_decay", 0.0)),
    )
    epochs = args.epochs or int(params["training"]["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
    mlflow.set_experiment(params["mlflow"]["experiment_name"])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_f1 = -1.0
    best_path = output_dir / f"{args.model}_best.pt"

    with mlflow.start_run(run_name=f"{args.model}_training"):
        mlflow.log_params({"model": args.model, "lr": lr, "epochs": epochs})
        for epoch in range(1, epochs + 1):
            train_loss, train_metrics = run_epoch(
                model, train_loader, criterion, device, optimizer
            )
            val_loss, val_metrics = run_epoch(model, val_loader, criterion, device)
            scheduler.step()

            metrics = {
                "train_loss": train_loss,
                "val_loss": val_loss,
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"val_{k}": v for k, v in val_metrics.items()},
            }
            mlflow.log_metrics(metrics, step=epoch)
            print(f"epoch={epoch}/{epochs} {metrics}")

            if val_metrics["macro_f1"] > best_f1:
                best_f1 = val_metrics["macro_f1"]
                torch.save(
                    {
                        "model_name": args.model,
                        "model_state_dict": model.state_dict(),
                        "classes": params["classes"],
                        "img_size": data["img_size"],
                        "val_metrics": val_metrics,
                    },
                    best_path,
                )
        mlflow.log_artifact(str(best_path))
    print(f"Saved best checkpoint: {best_path}")


if __name__ == "__main__":
    main()
