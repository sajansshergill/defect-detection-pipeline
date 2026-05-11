# 🔍 Manufacturing Defect Detection Pipeline
A multi-model computer vision ensemble for industrial surfact detect classification —— built to mirror SAS's applied AI use cases in manufacturing quality control.

## 📌 Problem Statement
Manufacturing lines generate thousands of product surfact images daily. Manual inspection is slow, inconsistent, and expensive. This project builds and end-to-end computer vision pipeline that:
- Classifies surface defects across 5 categories: **scratch, dent, stain, crack, none**
- Ensemble three deep learning architectures for robust predictions
- Surfaces explainability via Grad-CAM heatmaps overlaid on defect regions
- Exposes results through an interactive Streamlit dashboard for QA operators

## 🏗️ Architecture Overview

Raw Images (MVTec AD Dataset)
        │
        ▼
┌──────────────────────┐
│   Data Preprocessing  │  ← Augmentation, normalization, class-weighted sampling
└──────────┬───────────┘
           │
     ┌─────┼──────────┐
     ▼     ▼          ▼
ResNet-50  EfficientNet-B4  ViT-B/16
(Texture)  (Scale-aware)   (Global attention)
     │     │          │
     └─────┴──────────┘
           │
     Ensemble Layer (soft voting with learned weights)
           │
     ┌─────▼──────────────┐
     │  Prediction +       │
     │  Grad-CAM Heatmap   │
     └─────┬──────────────┘
           │
     Streamlit Dashboard


## 📁 Project Structure
<img width="754" height="958" alt="image" src="https://github.com/user-attachments/assets/8c3b4164-c76d-48ad-8ce0-ba6f6f7d9801" />

## 🗂️ Dataset
This project uses the MVTec Anomaly Detection (MVTec AD) dataset —— the standard benchamrk for indutrial defect detection.

<img width="582" height="576" alt="image" src="https://github.com/user-attachments/assets/9d621dd9-cbc4-4d68-a0d1-3c9e0429f9af" />

## Download and place at:
data/
└── mvtec_ad/
    ├── bottle/
    │   ├── train/good/
    │   └── test/broken_large/
    ├── wood/
    └── ...

## ⚙️ Setup
1. Clone the repo
bashgit clone https://github.com/sajanshergill/manufacturing-defect-detection.git
cd manufacturing-defect-detection

2. Create environment
bashpython -m venv venv
source venv/bin/activate
pip install -r requirements.txt

3. Pull data with DVC
bashdvc pull

4. Or run via Docker
bashdocker-compose up --build

## 🚀 Training
**Train all three models**
bashpython src/training/train.py --model resnet --epochs 30 --lr 1e-4
python src/training/train.py --model efficientnet --epochs 30 --lr 5e-5
python src/training/train.py --model vit --epochs 25 --lr 3e-5

**Ensemble evaluation**
bashpython src/training/evaluate.py --mode ensemble

All experiments are automatically tracked in **MLflow**:
mlflow ui
### Open http://localhost:5000

## 🎯 Threshold Tuning Strategy
Standard F1-optimal threshold is **not used here** —— this is an intentional business decision.

In manufacturing, a **missed defect (false negative)** reaching the customer is catastrophically more costly than a false alarm (false positive) triggering a manual reinspection.
<img width="536" height="364" alt="image" src="https://github.com/user-attachments/assets/452062b3-0727-4e8b-8dff-6f8c946f1e00" />

**Recall is the primary optimization target.** The threshold is treated as a configurable business parameter in params.yml, not a statis model artifact.

## 📊 Results
<img width="602" height="452" alt="image" src="https://github.com/user-attachments/assets/8312097f-2ca5-4929-ae0f-27cb9e328ef5" />

Evaluated on held-out MVTec AD test set across 5 defect categories. Ensemble uses learned soft-voting weights optimized on the validation split.

## 🧠 Explainability
Every prediction ships with a Grad-CAM heatmap highlighting the image region that most influenced the classifiaction decision. This is critical for operator trust —— QA teams can visually verify that the model is attending to the actual defect region, not spurios background artifacts.

from src.explainability.gradcam import generate_gradcam
heatmap = generate_gradcam(model, image_tensor, target_class=2)

## 📱 Streamlit Dashboard
streamlit run src/dashboard/app.py

**Features:**



