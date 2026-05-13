"""Streamlit dashboard for QA defect inspection."""

import tempfile
from pathlib import Path
import sys

import numpy as np
import streamlit as st
import torch
import yaml
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.augmentations import get_val_transforms
from src.explainability.gradcam import generate_gradcam
from src.models.ensemble import build_model, load_checkpoint, load_ensemble
from src.models.ensemble import parse_checkpoint_args
from src.pipeline.infer import predict_image


@st.cache_data
def load_params(path: str = "params.yaml"):
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@st.cache_resource
def load_single_model(model_name: str, checkpoint: str, num_classes: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_name, num_classes=num_classes, pretrained=False)
    return load_checkpoint(model, checkpoint, device).eval(), device


@st.cache_resource
def load_ensemble_model(entries, num_classes: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return load_ensemble(parse_checkpoint_args(entries), num_classes, device), device


def main() -> None:
    st.set_page_config(page_title="Defect Detection QA", layout="wide")
    params = load_params()
    class_names = params["classes"]["names"]
    img_size = int(params["data"]["img_size"])

    st.title("Manufacturing Defect Detection")
    st.caption("Upload a product surface image to classify defects and inspect Grad-CAM.")

    with st.sidebar:
        mode = st.radio("Inference mode", ["single", "ensemble"])
        threshold = st.slider(
            "Defect threshold",
            min_value=0.0,
            max_value=1.0,
            value=float(params["ensemble"].get("threshold", 0.5)),
            step=0.01,
        )
        if mode == "single":
            model_name = st.selectbox("Model", ["resnet", "efficientnet", "vit"])
            checkpoint = st.text_input(
                "Checkpoint path",
                value=f"artifacts/checkpoints/{model_name}_best.pt",
            )
        else:
            checkpoint_text = st.text_area(
                "Ensemble checkpoints, one NAME=PATH per line",
                value=(
                    "resnet=artifacts/checkpoints/resnet_best.pt\n"
                    "efficientnet=artifacts/checkpoints/efficientnet_best.pt\n"
                    "vit=artifacts/checkpoints/vit_best.pt"
                ),
            )

    uploaded = st.file_uploader("Upload image", type=["png", "jpg", "jpeg", "bmp", "webp"])
    if uploaded is None:
        st.info("Upload an image to start.")
        return

    suffix = Path(uploaded.name).suffix or ".png"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.getvalue())
        image_path = Path(tmp.name)

    left, right = st.columns(2)
    with left:
        st.image(Image.open(image_path), caption="Input image", use_container_width=True)

    try:
        if mode == "single":
            model, device = load_single_model(
                model_name,
                checkpoint,
                params["classes"]["num_classes"],
            )
        else:
            entries = [line.strip() for line in checkpoint_text.splitlines() if line.strip()]
            model, device = load_ensemble_model(entries, params["classes"]["num_classes"])

        result = predict_image(
            model,
            image_path,
            class_names,
            img_size,
            threshold,
            device,
        )
        with right:
            st.metric("Prediction", result["prediction"])
            st.metric("Defect score", f"{result['defect_score']:.3f}")
            st.bar_chart(result["probabilities"])

        if mode == "single":
            image = np.asarray(Image.open(image_path).convert("RGB"))
            tensor = get_val_transforms(img_size)(image=image)["image"]
            _, overlay = generate_gradcam(
                model,
                tensor,
                target_class=result["prediction_id"],
            )
            st.image(overlay, caption="Grad-CAM", use_container_width=True)
        else:
            st.caption("Grad-CAM is shown for single-model mode only.")
    except Exception as exc:
        st.error(str(exc))
        st.stop()


if __name__ == "__main__":
    main()
