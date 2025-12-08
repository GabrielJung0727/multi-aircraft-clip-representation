"""
Streamlit UI skeleton so we can quickly inspect PlanesNet predictions.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st
from PIL import Image

DATA_DIR = Path("data/planesnet/planesnet")


def list_samples(limit: int = 12) -> list[Path]:
    return sorted(DATA_DIR.glob("*.png"))[:limit]


def main() -> None:
    st.title("PlanesNet Quick Look")
    st.write("Streaming from the locally extracted PlanesNet dataset.")

    if not DATA_DIR.exists():
        st.error(f"Could not find {DATA_DIR}. Please extract PlanesNet first.")
        return

    samples = list_samples()
    cols = st.columns(3)
    for idx, image_path in enumerate(samples):
        with cols[idx % 3]:
            img = Image.open(image_path)
            st.image(img, caption=image_path.name, use_column_width=True)


if __name__ == "__main__":
    main()
