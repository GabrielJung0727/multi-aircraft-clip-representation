"""
Visualization utilities for the aircraft representation learning project.

Generates dataset composition plots and training/validation curves from the
JSON logs written by `train_multidataset.py`.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
from torchvision import transforms

from datasets.planesnet_dataset import PlanesNetDataset


PLOT_DIR = Path("results/plots")
LOG_DIR = Path("results/logs")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate plots for training progress and dataset stats.")
    parser.add_argument("--log-file", type=Path, default=None, help="Path to a specific training log JSON.")
    return parser.parse_args()


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


def load_training_history(log_file: Optional[Path]) -> Optional[List[Dict]]:
    target = log_file
    if target is None:
        candidates = sorted(LOG_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        target = candidates[0] if candidates else None
    if target is None or not target.exists():
        return None
    with target.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def plot_accuracy_curves(history: List[Dict]) -> None:
    epochs = [record["epoch"] for record in history]
    datasets = history[0]["train"].keys()

    fig, axes = plt.subplots(len(datasets), 1, figsize=(8, 4 * len(datasets)), sharex=True)
    if hasattr(axes, "flatten"):
        axes = axes.flatten().tolist()
    elif not isinstance(axes, list):
        axes = [axes]

    for ax, dataset in zip(axes, datasets):
        train_acc = [record["train"][dataset]["accuracy"] for record in history]
        val_epochs = [record["epoch"] for record in history if dataset in record["val"]]
        val_acc = [record["val"][dataset]["accuracy"] for record in history if dataset in record["val"]]
        ax.plot(epochs[: len(train_acc)], train_acc, label="Train", marker="o")
        ax.plot(val_epochs, val_acc, label="Validation", marker="s")
        ax.set_title(f"{dataset.upper()} accuracy")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1.0)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()

    axes[-1].set_xlabel("Epoch")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "accuracy_curves.png", dpi=200)
    plt.close(fig)


def plot_validation_summary(history: List[Dict]) -> None:
    latest = history[-1]["val"]
    datasets = list(latest.keys())
    accuracies = [latest[name]["accuracy"] for name in datasets]
    losses = [latest[name]["loss"] for name in datasets]

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax2 = ax1.twinx()
    ax1.bar(datasets, accuracies, color="#2d9bf0", alpha=0.7, label="Accuracy")
    ax2.plot(datasets, losses, color="#ef476f", marker="o", label="Loss")
    ax1.set_ylabel("Accuracy")
    ax2.set_ylabel("Loss")
    ax1.set_ylim(0, 1.0)
    ax1.set_title("Validation snapshot (last epoch)")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "validation_summary.png", dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    ensure_dirs()

    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    dataset = PlanesNetDataset(root="data/planesnet", split="train", transform=transform)
    barplot_dataset_stats(dataset)

    history = load_training_history(args.log_file)
    if history:
        plot_accuracy_curves(history)
        plot_validation_summary(history)
        print(f"Generated plots from log with {len(history)} epochs.")
    else:
        print("No training logs found; generated dataset plot only.")


if __name__ == "__main__":  # pragma: no cover - script entry
    main()
