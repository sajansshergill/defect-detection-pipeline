# Manufacturing Defect Detection Pipeline

End-to-end computer vision pipeline for manufacturing surface defect classification.
It trains ResNet, EfficientNet, and ViT classifiers, evaluates a soft-voting ensemble,
tunes a recall-oriented defect threshold, generates Grad-CAM explanations, and serves
predictions through a Streamlit dashboard.

## Problem

Manufacturing QA teams need to catch defects before products ship. This project classifies
surface images into five classes: `good`, `scratch`, `dent`, `stain`, and `crack`.
For operations, missed defects are more costly than false alarms, so the binary
defect threshold is configurable in `params.yaml`.

## Dataset

The loader expects MVTec AD-style data under `data/mvtec_ad`:

```text
data/mvtec_ad/
  bottle/
    train/good/
    test/good/
    test/broken_large/
  wood/
  metal_nut/
  leather/
  tile/
```

Update `params.yaml` if your dataset root or categories differ.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train

```bash
python src/training/train.py --model resnet
python src/training/train.py --model efficientnet
python src/training/train.py --model vit
```

Checkpoints are written to `artifacts/checkpoints`. MLflow logs are written to
`mlruns`:

```bash
mlflow ui --backend-store-uri mlruns
```

## Evaluate And Tune

```bash
python src/training/evaluate.py \
  --mode ensemble \
  --checkpoints \
  resnet=artifacts/checkpoints/resnet_best.pt \
  efficientnet=artifacts/checkpoints/efficientnet_best.pt \
  vit=artifacts/checkpoints/vit_best.pt

python src/training/threshold_tuning.py \
  --predictions artifacts/evaluation/ensemble_test_predictions.csv
```

## Inference

```bash
python src/pipeline/infer.py \
  --mode single \
  --model resnet \
  --checkpoint artifacts/checkpoints/resnet_best.pt \
  --input path/to/image_or_directory
```

## Dashboard

```bash
streamlit run src/dashboard/app.py
```

Or run the dashboard and MLflow UI with Docker:

```bash
docker-compose up --build
```

Dashboard: `http://localhost:8501`
MLflow: `http://localhost:5000`

## DVC Pipeline

```bash
dvc repro
dvc dag
```

