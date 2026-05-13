"""Streamlit dashboard for QA defect inspection."""

import sys
import tempfile
from pathlib import Path

import streamlit as st
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@st.cache_data
def load_params(path: str = "params.yaml"):
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@st.cache_resource
def load_single_model(model_name: str, checkpoint: str, num_classes: int):
    import torch

    from src.models.ensemble import build_model, load_checkpoint

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_name, num_classes=num_classes, pretrained=False)
    return load_checkpoint(model, checkpoint, device).eval(), device


@st.cache_resource
def load_ensemble_model(entries, num_classes: int):
    import torch

    from src.models.ensemble import load_ensemble, parse_checkpoint_args

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return load_ensemble(parse_checkpoint_args(entries), num_classes, device), device


def render_project_overview() -> None:
    st.subheader("Project Overview")
    st.write(
        "This app demonstrates a manufacturing surface defect detection pipeline "
        "with training, evaluation, inference, threshold tuning, and explainability."
    )
    col1, col2, col3 = st.columns(3)
    col1.metric("Classes", "5")
    col2.metric("Models", "ResNet / EfficientNet / ViT")
    col3.metric("Serving", "Streamlit")

    st.subheader("Pipeline")
    st.code(
        "MVTec-style images -> augmentations -> model training -> ensemble "
        "evaluation -> threshold tuning -> dashboard inference",
        language="text",
    )

    st.subheader("Demo Metrics")
    st.bar_chart(
        {
            "macro_f1": [0.82, 0.85, 0.84, 0.88],
            "defect_recall": [0.91, 0.93, 0.90, 0.96],
        },
        x_label="Model index",
        y_label="Score",
    )


def render_demo_prediction(uploaded, class_names, threshold: float) -> None:
    from PIL import Image

    left, right = st.columns(2)
    if uploaded is not None:
        image = Image.open(uploaded).convert("RGB")
        with left:
            st.image(image, caption="Uploaded image", use_container_width=True)

        width, height = image.size
        brightness = sum(image.resize((1, 1)).getpixel((0, 0))) / (3 * 255)
        defect_score = max(0.05, min(0.95, 1.0 - brightness))
        prediction = "good" if defect_score < threshold else "scratch"
        probabilities = {
            "good": round(1.0 - defect_score, 3),
            "scratch": round(defect_score * 0.45, 3),
            "dent": round(defect_score * 0.20, 3),
            "stain": round(defect_score * 0.20, 3),
            "crack": round(defect_score * 0.15, 3),
        }

        with right:
            st.metric("Demo prediction", prediction)
            st.metric("Demo defect score", f"{defect_score:.3f}")
            st.caption(f"Image size: {width}x{height}")
            st.bar_chart(probabilities)
        st.info("Demo mode uses a lightweight heuristic so the app works without checkpoints.")
    else:
        st.info("Upload an image for a lightweight demo prediction.")
        st.write("Class labels:", ", ".join(class_names))


def main() -> None:
    st.set_page_config(page_title="Defect Detection QA", layout="wide")
    params = load_params()
    class_names = params["classes"]["names"]
    img_size = int(params["data"]["img_size"])

    st.title("Manufacturing Defect Detection")
    st.caption("Upload a product surface image to classify defects and inspect Grad-CAM.")

    with st.sidebar:
        mode = st.radio("Inference mode", ["demo", "single", "ensemble"])
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

    render_project_overview()
    uploaded = st.file_uploader("Upload image", type=["png", "jpg", "jpeg", "bmp", "webp"])
    if mode == "demo":
        render_demo_prediction(uploaded, class_names, threshold)
        return

    if uploaded is None:
        st.info("Upload an image to start real checkpoint inference.")
        return

    from PIL import Image

    from src.pipeline.infer import predict_image

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
            import numpy as np

            from src.data.augmentations import get_val_transforms
            from src.explainability.gradcam import generate_gradcam

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
        st.error("Real inference could not start.")
        st.exception(exc)
        st.stop()


if __name__ == "__main__":
    main()
