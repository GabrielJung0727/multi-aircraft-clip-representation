"""
Entry point for CLIP-style representation learning on PlanesNet.

The script intentionally keeps the training loop light so it can serve as a
smoke test while the heavier datasets finish downloading.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Optional, Tuple

import torch
from rich.console import Console
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.auto import tqdm

from datasets.planesnet_dataset import PlanesNetDataset
from models.classifier_heads import LinearClassifier
from models.clip_backbones import CLIPBackbone


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a CLIP-style head on PlanesNet.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/planesnet"), help="Root path for PlanesNet.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--val-fraction", type=float, default=0.1, help="Fraction of data reserved for validation.")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=224)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_transforms(image_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
            transforms.ToTensor(),
            normalize,
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize(image_size + 32),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train_transform, val_transform


def prepare_datasets(args: argparse.Namespace) -> Tuple[PlanesNetDataset, Optional[PlanesNetDataset]]:
    base_dataset = PlanesNetDataset(root=args.data_dir, split=args.split, transform=None)
    train_transform, val_transform = build_transforms(args.image_size)

    total_samples = len(base_dataset)
    val_size = int(total_samples * args.val_fraction) if args.val_fraction > 0 else 0

    generator = torch.Generator().manual_seed(args.seed)
    permutation = torch.randperm(total_samples, generator=generator).tolist()

    val_indices = permutation[:val_size] if val_size > 0 else []
    train_indices = permutation[val_size:] if val_size > 0 else permutation

    train_dataset = PlanesNetDataset(
        root=args.data_dir,
        split=args.split,
        transform=train_transform,
        indices=train_indices,
    )
    val_dataset = (
        PlanesNetDataset(root=args.data_dir, split=args.split, transform=val_transform, indices=val_indices)
        if val_indices
        else None
    )
    return train_dataset, val_dataset


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    running_correct = 0
    total_samples = 0

    progress_bar = tqdm(dataloader, desc="Batches", leave=False)
    for images, labels, _ in progress_bar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels.long())
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        preds = torch.argmax(logits, dim=1)
        running_correct += (preds == labels).sum().item()
        total_samples += batch_size

        avg_loss = running_loss / max(total_samples, 1)
        avg_acc = running_correct / max(total_samples, 1)
        progress_bar.set_postfix({"loss": f"{avg_loss:.4f}", "acc": f"{avg_acc:.3f}"})

    return running_loss / max(total_samples, 1), running_correct / max(total_samples, 1)


def evaluate(
    model: torch.nn.Module,
    dataloader: Optional[DataLoader],
    criterion: torch.nn.Module,
    device: torch.device,
) -> Tuple[Optional[float], Optional[float]]:
    if dataloader is None:
        return None, None

    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels, _ in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels.long())

            batch = labels.size(0)
            loss_sum += loss.item() * batch
            correct += (torch.argmax(logits, dim=1) == labels).sum().item()
            total += batch

    avg_loss = loss_sum / max(total, 1)
    avg_acc = correct / max(total, 1)
    return avg_loss, avg_acc


def log_epoch(console: Console, epoch: int, train_metrics: Tuple[float, float], val_metrics: Tuple[Optional[float], Optional[float]]) -> None:
    table = Table(title=f"Epoch {epoch}", show_lines=True)
    table.add_column("Split", justify="center")
    table.add_column("Loss", justify="right")
    table.add_column("Accuracy", justify="right")

    train_loss, train_acc = train_metrics
    table.add_row("Train", f"{train_loss:.4f}", f"{train_acc:.3f}")

    val_loss, val_acc = val_metrics
    if val_loss is not None and val_acc is not None:
        table.add_row("Validation", f"{val_loss:.4f}", f"{val_acc:.3f}")
    else:
        table.add_row("Validation", "—", "—")

    console.print(table)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    console = Console()
    train_dataset, val_dataset = prepare_datasets(args)
    console.print(f"Loaded {len(train_dataset)} training samples", style="bold green")
    if val_dataset:
        console.print(f"Loaded {len(val_dataset)} validation samples", style="bold blue")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = (
        DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        if val_dataset
        else None
    )

    backbone = CLIPBackbone()
    head = LinearClassifier()
    model = torch.nn.Sequential(backbone, head).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    progress = Progress(
        TextColumn("[bold blue]Epoch[/]"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
    )

    with progress:
        epoch_task = progress.add_task("training", total=args.epochs)
        for epoch in range(1, args.epochs + 1):
            train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, args.grad_clip)
            val_metrics = evaluate(model, val_loader, criterion, device)
            log_epoch(console, epoch, train_metrics, val_metrics)
            progress.advance(epoch_task)


if __name__ == "__main__":  # pragma: no cover - script entry
    main()
