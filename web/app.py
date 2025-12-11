"""
ROI dashboard + CNN vs CNIP comparison + CLIP zero-shot demo.
"""

from __future__ import annotations

from itertools import islice
from pathlib import Path
from typing import Iterable

import torch
import streamlit as st
from PIL import Image

DATA_ROOT = Path("data")
CSV_DIR = DATA_ROOT / "csv_file"
IMAGE_ROOT = DATA_ROOT / "data2"
DEFAULT_IMAGE_DIR = IMAGE_ROOT / "Ret-3_test"


def safe_list_images(directory: Path, limit: int = 12) -> list[Path]:
    """Return a small list of images without walking the whole tree."""
    if not directory.exists():
        return []
    candidates = directory.glob("*.jpg")
    return list(islice(candidates, limit))


def list_csv_files() -> list[Path]:
    if not CSV_DIR.exists():
        return []
    return sorted(CSV_DIR.glob("**/*.csv"))


@st.cache_resource(show_spinner=False)
def load_clip():
    """Lazy import to avoid crashing when transformers extras are missing."""
    try:
        from transformers import CLIPModel, CLIPProcessor
    except Exception as exc:  # pylint: disable=broad-except
        st.error(f"transformers import failed ({exc}). Install/upgrade transformers[torch] and restart.")
        return None

    try:
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    except Exception as exc:  # pylint: disable=broad-except
        st.error(f"CLIP weights/processor load failed: {exc}")
        return None
    return model, processor


def run_clip_zero_shot(image: Image.Image, prompts: list[str]) -> list[tuple[str, float]]:
    """Zero-shot scoring using CLIP; returns (prompt, probability)."""
    loaded = load_clip()
    if loaded is None:
        return []
    model, processor = loaded
    inputs = processor(
        text=prompts,
        images=image,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits_per_image[0]
        probs = logits.softmax(dim=0)
    return list(zip(prompts, probs.tolist()))


def calculate_roi(
    volume: int,
    cnn_accuracy: float,
    cnip_accuracy: float,
    value_per_sample: float,
    cnn_cost_per_k: float,
    cnip_cost_per_k: float,
) -> dict[str, float]:
    """Simple ROI: extra value from accuracy uplift minus extra cost."""
    cnn_value = volume * cnn_accuracy * value_per_sample
    cnip_value = volume * cnip_accuracy * value_per_sample
    delta_value = cnip_value - cnn_value
    cnn_cost = (volume / 1000) * cnn_cost_per_k
    cnip_cost = (volume / 1000) * cnip_cost_per_k
    delta_cost = cnip_cost - cnn_cost
    roi = 0.0 if cnip_cost == 0 else (cnip_value - cnip_cost) / cnip_cost
    return {
        "cnn_value": cnn_value,
        "cnip_value": cnip_value,
        "delta_value": delta_value,
        "cnn_cost": cnn_cost,
        "cnip_cost": cnip_cost,
        "delta_cost": delta_cost,
        "roi": roi,
    }


def render_roi_calculator() -> None:
    st.subheader("ROI (Deep Learning Return on Investment)")
    col1, col2, col3 = st.columns(3)
    with col1:
        volume = st.number_input("Images per month", min_value=1, value=50000, step=1000)
        value_per_sample = st.number_input("Business value per correct item", min_value=0.0, value=120.0, step=10.0)
    with col2:
        cnn_accuracy = st.number_input("CNN accuracy", min_value=0.0, max_value=1.0, value=0.82, step=0.01)
        cnip_accuracy = st.number_input("CNIP accuracy", min_value=0.0, max_value=1.0, value=0.89, step=0.01)
    with col3:
        cnn_cost_per_k = st.number_input("CNN cost per 1k images", min_value=0.0, value=4200.0, step=100.0)
        cnip_cost_per_k = st.number_input("CNIP cost per 1k images", min_value=0.0, value=4800.0, step=100.0)

    roi = calculate_roi(
        volume=volume,
        cnn_accuracy=cnn_accuracy,
        cnip_accuracy=cnip_accuracy,
        value_per_sample=value_per_sample,
        cnn_cost_per_k=cnn_cost_per_k,
        cnip_cost_per_k=cnip_cost_per_k,
    )

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Added value", f"{roi['delta_value']:,.0f}")
    col_b.metric("Cost delta", f"{roi['delta_cost']:,.0f}")
    col_c.metric("ROI ( (value-cost)/cost )", f"{roi['roi']:.2f}x")
    payback = "n/a" if roi["delta_cost"] <= 0 else f"{roi['delta_cost'] / max(roi['delta_value'], 1):.2f} months"
    col_d.metric("Payback (rough)", payback)

    st.caption("This ROI block is a simplified example. Replace with your true cost/value numbers.")


def render_model_comparison() -> None:
    st.subheader("CNN vs CNIP performance (4 targets)")
    st.write("Compare accuracy, recall, F1, and latency side-by-side.")

    col_inputs = st.columns(4)
    metrics = ["Accuracy", "Recall", "F1", "Latency (ms)"]
    defaults_cnn = [0.82, 0.80, 0.81, 45.0]
    defaults_cnip = [0.89, 0.88, 0.88, 38.0]
    cnn_vals = []
    cnip_vals = []
    for idx, metric in enumerate(metrics):
        with col_inputs[idx]:
            cnn_vals.append(
                st.number_input(f"CNN {metric}", value=defaults_cnn[idx], step=0.01 if idx < 3 else 1.0)
            )
            cnip_vals.append(
                st.number_input(f"CNIP {metric}", value=defaults_cnip[idx], step=0.01 if idx < 3 else 1.0)
            )

    col_cards = st.columns(4)
    for idx, metric in enumerate(metrics):
        delta = cnip_vals[idx] - cnn_vals[idx]
        delta_label = f"{delta:+.2f}" if idx < 3 else f"{delta:+.0f} ms"
        value = f"{cnip_vals[idx]:.2f}" if idx < 3 else f"{cnip_vals[idx]:.0f} ms"
        col_cards[idx].metric(f"{metric} (CNIP)", value, delta=delta_label)

    st.caption("CNIP here denotes a CLIP + rotation-oriented enhanced model used for comparison.")


def render_market_models() -> None:
    st.subheader("Market/available models (incl. RemoteCLIP)")
    st.markdown(
        "- RemoteCLIP: https://github.com/ChenDelong1999/RemoteCLIP\n"
        "- OpenAI CLIP: https://openai.com/index/clip/\n"
        "- CNIP/CNN: in-house experimental models"
    )
    table = [
        {"Model": "CNN", "Task": "Classification", "Top-1 Acc": 0.82, "F1": 0.81, "Notes": "Baseline conv classifier"},
        {"Model": "CNIP", "Task": "Multimodal classification", "Top-1 Acc": 0.89, "F1": 0.88, "Notes": "CLIP-based + Ro-CIT"},
        {"Model": "RemoteCLIP", "Task": "Remote-sensing text-image", "Top-1 Acc": 0.87, "F1": 0.86, "Notes": "RSICD/RSITMD zero-shot"},
        {"Model": "OpenAI CLIP ViT-B/32", "Task": "Zero-shot classification", "Top-1 Acc": 0.85, "F1": 0.84, "Notes": "General-domain pretrain"},
    ]
    st.dataframe(table, hide_index=True)
    st.caption("Numbers are demo placeholders; swap in your real benchmarks.")


def render_clip_demo() -> None:
    st.subheader("CLIP zero-shot demo (image description/classification)")
    default_prompts = [
        "a satellite image of an airport with multiple airplanes",
        "a satellite image with no planes",
        "a coastal city from above",
        "a barren desert landscape",
    ]
    prompt_text = st.text_area("Text prompts (one per line)", "\n".join(default_prompts), height=120)
    prompts = [p.strip() for p in prompt_text.splitlines() if p.strip()]

    sample_images = safe_list_images(DEFAULT_IMAGE_DIR, limit=8)
    image_names = [img.name for img in sample_images]
    upload_label = "(upload instead)"
    selected_name = st.selectbox("data/data2/Ret-3_test sample", [upload_label] + image_names, index=1 if image_names else 0)
    uploaded = st.file_uploader("Upload image (.png/.jpg)", type=["png", "jpg", "jpeg"])

    image_to_use: Image.Image | None = None
    if uploaded:
        image_to_use = Image.open(uploaded).convert("RGB")
    elif selected_name != upload_label and sample_images:
        image_path = DEFAULT_IMAGE_DIR / selected_name
        image_to_use = Image.open(image_path).convert("RGB")

    if image_to_use and prompts:
        col_img, col_scores = st.columns([1, 1])
        with col_img:
            st.image(image_to_use, caption="input image", width="stretch")
        with col_scores:
            with st.spinner("Running CLIP zero-shot..."):
                scores = run_clip_zero_shot(image_to_use, prompts)
            if not scores:
                st.warning("CLIP could not run. Check transformers installation or see error above.")
            else:
                st.write("Probabilities per prompt")
                st.dataframe(
                    [{"prompt": p, "probability": f"{prob:.3f}"} for p, prob in scores],
                    hide_index=True,
                )
    else:
        st.info("Add prompts and pick a sample/uploaded image to see zero-shot results.")


def render_dataset_glance() -> None:
    st.subheader("Data locations at a glance")
    csv_files = list_csv_files()
    st.write(f"CSV metadata files found: {len(csv_files)} (under data/csv_file)")
    if csv_files:
        st.dataframe([{"file": str(p.relative_to(DATA_ROOT)), "size_kb": round(p.stat().st_size / 1024, 1)} for p in csv_files][:10], hide_index=True)
    images = safe_list_images(DEFAULT_IMAGE_DIR, limit=6)
    st.write(f"Sample images (data/data2/Ret-3_test): showing {len(images)}")
    cols = st.columns(6)
    for idx, img_path in enumerate(images):
        with cols[idx]:
            st.image(str(img_path), caption=img_path.name, width="stretch")


def main() -> None:
    st.set_page_config(page_title="ROI + CLIP Dashboard", layout="wide")
    st.title("ROI / CNN vs CNIP / CLIP Zero-shot Dashboard")
    st.write("Compute ROI, compare CNN vs CNIP, and run CLIP zero-shot in one place.")

    render_roi_calculator()
    st.markdown("---")
    render_model_comparison()
    st.markdown("---")
    render_market_models()
    st.markdown("---")
    render_clip_demo()
    st.markdown("---")
    render_dataset_glance()


if __name__ == "__main__":
    main()
