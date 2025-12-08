"""
Utility script that generates placeholder plots for the PlanesNet dataset.

Once HRPlanes and FGVC Aircraft downloads complete, extend this module to
cover the remaining plot types listed in README.md.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
from torchvision import transforms

from datasets.planesnet_dataset import PlanesNetDataset


PLOT_DIR = Path("results/plots")


def ensure_dirs() -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)


def barplot_dataset_stats(dataset: PlanesNetDataset) -> None:
    counts = Counter(label for _, label, _ in dataset)
    labels = ["no-plane", "plane"]
    values = [counts.get(0, 0), counts.get(1, 0)]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, values, color=["#5c6ac4", "#47b881"])
    ax.set_ylabel("Samples")
    ax.set_title("PlanesNet class distribution")
    for xpos, val in enumerate(values):
        ax.text(xpos, val + 10, str(val), ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "barplot_dataset_stats.png", dpi=200)
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    dataset = PlanesNetDataset(root="data/planesnet", split="train", transform=transform)
    barplot_dataset_stats(dataset)
    print(f"Saved plots to {PLOT_DIR.resolve()}")


if __name__ == "__main__":  # pragma: no cover - script entry
    main()
