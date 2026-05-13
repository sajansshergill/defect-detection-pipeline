"""Business-oriented threshold tuning for defect/no-defect decisions."""

import argparse
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import f1_score, precision_score, recall_score


def tune_threshold(
    y_true: Iterable[int],
    defect_scores: Iterable[float],
    min_recall: float = 0.95,
    min_precision: float = 0.10,
) -> Dict[str, float]:
    """Choose the highest-F1 threshold that still satisfies recall needs."""
    y_binary = np.asarray([int(label != 0) for label in y_true])
    scores = np.asarray(list(defect_scores), dtype=np.float32)
    best = {
        "threshold": 0.5,
        "f1": -1.0,
        "precision": 0.0,
        "recall": 0.0,
    }
    fallback = best.copy()

    for threshold in np.linspace(0.01, 0.99, 99):
        pred = (scores >= threshold).astype(int)
        precision = precision_score(y_binary, pred, zero_division=0)
        recall = recall_score(y_binary, pred, zero_division=0)
        f1 = f1_score(y_binary, pred, zero_division=0)
        candidate = {
            "threshold": float(threshold),
            "f1": float(f1),
            "precision": float(precision),
            "recall": float(recall),
        }
        if recall > fallback["recall"] or (
            recall == fallback["recall"] and f1 > fallback["f1"]
        ):
            fallback = candidate
        if recall >= min_recall and precision >= min_precision and f1 > best["f1"]:
            best = candidate

    return best if best["f1"] >= 0 else fallback


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune defect threshold from predictions CSV.")
    parser.add_argument("--predictions", required=True, help="CSV with label and defect_score columns.")
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--output", default="artifacts/threshold.yaml")
    parser.add_argument("--min-recall", type=float, default=0.95)
    parser.add_argument("--min-precision", type=float, default=0.10)
    args = parser.parse_args()

    df = pd.read_csv(args.predictions)
    result = tune_threshold(
        df["label"].to_numpy(),
        df["defect_score"].to_numpy(),
        min_recall=args.min_recall,
        min_precision=args.min_precision,
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        yaml.safe_dump({"ensemble": result}, fh)
    print(result)


if __name__ == "__main__":
    main()
