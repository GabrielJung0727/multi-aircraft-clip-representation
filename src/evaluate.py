"""
Quick evaluation helper for any trained checkpoint.

This currently supports PlanesNet only because the other datasets are still
downloading.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets.planesnet_dataset import PlanesNetDataset
from models.classifier_heads import LinearClassifier
from models.clip_backbones import CLIPBackbone


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate checkpoints on PlanesNet.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/planesnet"))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    dataset = PlanesNetDataset(root=args.data_dir, split="val", transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    backbone = CLIPBackbone()
    head = LinearClassifier()
    model = torch.nn.Sequential(backbone, head).to(args.device)
    model.eval()

    correct, total = 0, 0
    with torch.no_grad():
        for images, labels, _ in dataloader:
            images = images.to(args.device)
            labels = labels.to(args.device)
            logits = model(images)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.numel()

    accuracy = correct / max(total, 1)
    print(f"Validation accuracy: {accuracy:.4f} ({correct}/{total})")


if __name__ == "__main__":  # pragma: no cover - script entry
    main()
