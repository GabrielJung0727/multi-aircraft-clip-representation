"""
Entry point for CLIP-style representation learning on PlanesNet.

The script intentionally keeps the training loop light so it can serve as a
smoke test while the heavier datasets finish downloading.
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
    parser = argparse.ArgumentParser(description="Train a CLIP-style head on PlanesNet.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/planesnet"), help="Root path for PlanesNet.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--split", type=str, default="train")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )
    dataset = PlanesNetDataset(root=args.data_dir, split=args.split, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    backbone = CLIPBackbone()
    head = LinearClassifier()
    model = torch.nn.Sequential(backbone, head).to(args.device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    model.train()
    for epoch in range(args.epochs):
        running_loss = 0.0
        for images, labels, _ in dataloader:
            images = images.to(args.device)
            labels = labels.to(args.device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / max(len(dataloader), 1)
        print(f"[Epoch {epoch+1}/{args.epochs}] loss={avg_loss:.4f}")


if __name__ == "__main__":  # pragma: no cover - script entry
    main()
