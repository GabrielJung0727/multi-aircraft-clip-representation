"""
Unified training script that mixes PlanesNet, HRPlanes, and FGVC Aircraft.

It combines CLIP-style contrastive learning (image/text), Ro-CIT augmentation,
U-Net auxiliary segmentation, DiT refinement, and an in-house hashing text encoder
instead of any off-the-shelf pretrained model.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
from models.text_encoder import HashingTextEncoder

from datasets.fgvc_dataset import FGVCAircraftDataset
from datasets.hrplanes_dataset import HRPlanesDataset
from datasets.planesnet_dataset import PlanesNetDataset
from datasets.seg7_dataset import Seg7Dataset
from datasets.det10_dataset import Det10Dataset
from models.classifier_heads import LinearClassifier
from models.clip_backbones import BackboneConfig, CLIPBackbone, DiTBackbone
from models.unet_decoder import UNetDecoder


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
Image.MAX_IMAGE_PIXELS = None  # disable PIL decompression bomb guard for large remote-sensing tiles


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-dataset aircraft training pipeline.")
    parser.add_argument("--planesnet-dir", type=Path, default=Path("data/planesnet"))
    parser.add_argument("--hrplanes-dir", type=Path, default=Path("data/hrplanes"))
    parser.add_argument("--fgvc-dir", type=Path, default=Path("data/fgvc_aircraft"))
    parser.add_argument("--seg7-dir", type=Path, default=Path("data/data2/Seg-7"))
    parser.add_argument("--seg7-csv", type=Path, default=Path("data/csv_file/train/Seg-4.csv"))
    parser.add_argument("--det10-dir", type=Path, default=Path("data/data2/Det-10_part1"))
    parser.add_argument("--det10-csv", type=Path, default=Path("data/csv_file/train/Det-10.csv"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--vision-device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--text-device", type=str, default="cpu")
    parser.add_argument("--image-size", type=int, default=224, help="Square resize/crop for all datasets.")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seg-weight", type=float, default=0.3)
    parser.add_argument("--clip-weight", type=float, default=0.5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-dir", type=Path, default=Path("results/logs"))
    parser.add_argument("--max-angle", type=float, default=35.0)
    parser.add_argument("--experiment-name", type=str, default="multidataset")
    parser.add_argument("--text-vocab-size", type=int, default=8192)
    parser.add_argument("--text-embed-dim", type=int, default=768, help="Text embedding dim (also used for image projection).")
    parser.add_argument("--backbone-width", type=int, default=64, help="Base channel width for the CNN stem.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_indices(length: int, val_fraction: float, seed: int) -> Tuple[List[int], List[int]]:
    indices = list(range(length))
    random.Random(seed).shuffle(indices)
    val_size = int(length * val_fraction)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    return train_indices, val_indices


def build_classic_transforms(image_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    normalize = transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD)
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.02),
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


class TextPromptEncoder:
    """Self-contained text encoder (hashing tokenizer + lightweight transformer)."""

    def __init__(self, vocab_size: int, embed_dim: int, device: torch.device, target_device: torch.device) -> None:
        self.encoder = HashingTextEncoder(vocab_size=vocab_size, embed_dim=embed_dim, device=device)
        self.target_device = target_device

    @torch.no_grad()
    def encode(self, texts: Sequence[str]) -> torch.Tensor:
        embeddings = self.encoder.encode(list(texts)).to(self.target_device)
        return embeddings


def rotation_oriented_cit(
    images: torch.Tensor,
    masks: Optional[torch.Tensor] = None,
    max_angle: float = 30.0,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Applies the same affine parameters to the image batch (and optional masks)."""
    batch = images.shape[0]
    augmented_images = []
    augmented_masks: List[torch.Tensor] = [] if masks is not None else None

    for idx in range(batch):
        angle = random.uniform(-max_angle, max_angle)
        translate = (
            random.uniform(-10, 10),
            random.uniform(-10, 10),
        )
        scale = random.uniform(0.9, 1.1)
        shear = random.uniform(-5, 5)

        img = TF.affine(images[idx], angle=angle, translate=translate, scale=scale, shear=shear)
        augmented_images.append(img)

        if masks is not None and augmented_masks is not None:
            mask = TF.affine(
                masks[idx],
                angle=angle,
                translate=translate,
                scale=scale,
                shear=shear,
                interpolation=InterpolationMode.NEAREST,
            )
            augmented_masks.append(mask)

    stacked_images = torch.stack(augmented_images)
    stacked_masks = torch.stack(augmented_masks) if augmented_masks is not None else None
    return stacked_images, stacked_masks


class MultiTaskAircraftModel(torch.nn.Module):
    def __init__(self, num_fgvc_classes: int, backbone_cfg: Optional[BackboneConfig] = None) -> None:
        super().__init__()
        cfg = backbone_cfg or BackboneConfig()
        self.backbone = CLIPBackbone(config=cfg)
        self.embed_dim = cfg.embed_dim
        self.dit = DiTBackbone(embed_dim=self.embed_dim)
        encoder_channels = tuple(self.backbone.stage_channels[::-1])
        self.seg_decoder = UNetDecoder(encoder_channels=encoder_channels, num_classes=1)
        self.classifiers = torch.nn.ModuleDict(
            {
                "planesnet": LinearClassifier(in_dim=self.embed_dim, num_classes=2),
                "fgvc": LinearClassifier(in_dim=self.embed_dim, num_classes=num_fgvc_classes, hidden_dim=512),
                "hrplanes": LinearClassifier(in_dim=self.embed_dim, num_classes=2),
                "seg7": LinearClassifier(in_dim=self.embed_dim, num_classes=2),
                "det10": LinearClassifier(in_dim=self.embed_dim, num_classes=2),
            }
        )

    def encode(self, images: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        pooled, pyramid = self.backbone.forward_features(images)
        projected = self.backbone.proj(pooled)
        tokens = projected.unsqueeze(-1).unsqueeze(-1)
        refined = self.dit(tokens)
        return refined, pyramid


def clip_contrastive_loss(image_features: torch.Tensor, text_features: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    image_features = F.normalize(image_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)
    logits = image_features @ text_features.t() / temperature
    targets = torch.arange(len(image_features), device=image_features.device)
    return (F.cross_entropy(logits, targets) + F.cross_entropy(logits.t(), targets)) * 0.5


def prepare_dataloaders(
    args: argparse.Namespace,
) -> Tuple[Dict[str, DataLoader], Dict[str, DataLoader], Dict[str, int]]:
    set_seed(args.seed)
    train_transform, val_transform = build_classic_transforms(args.image_size)

    planes_full = PlanesNetDataset(root=args.planesnet_dir, transform=train_transform)
    planes_train_idx, planes_val_idx = split_indices(len(planes_full), args.val_fraction, args.seed)
    planes_train = PlanesNetDataset(root=args.planesnet_dir, transform=train_transform, indices=planes_train_idx, split="train")
    planes_val = PlanesNetDataset(root=args.planesnet_dir, transform=val_transform, indices=planes_val_idx, split="val")

    hrplanes_full = HRPlanesDataset(
        root=args.hrplanes_dir,
        image_size=args.image_size,
        transform=transforms.Compose(
            [
                transforms.Resize((args.image_size, args.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
            ]
        ),
    )
    hr_train_idx, hr_val_idx = split_indices(len(hrplanes_full), args.val_fraction, args.seed + 1)
    hrplanes_train = HRPlanesDataset(root=args.hrplanes_dir, image_size=256, transform=hrplanes_full.transform, indices=hr_train_idx, split="train")
    hrplanes_val = HRPlanesDataset(root=args.hrplanes_dir, image_size=256, transform=hrplanes_full.transform, indices=hr_val_idx, split="val")

    fgvc_map_dataset = FGVCAircraftDataset(root=args.fgvc_dir)
    class_map = fgvc_map_dataset.class_to_idx
    fgvc_train = FGVCAircraftDataset(root=args.fgvc_dir, subset="train", transform=train_transform, class_to_idx=class_map)
    fgvc_val = FGVCAircraftDataset(root=args.fgvc_dir, subset="val", transform=val_transform, class_to_idx=class_map)

    seg7_full = Seg7Dataset(csv_path=args.seg7_csv, image_root=args.seg7_dir, transform=train_transform)
    seg7_train_idx, seg7_val_idx = split_indices(len(seg7_full), args.val_fraction, args.seed + 2)
    seg7_train = Seg7Dataset(csv_path=args.seg7_csv, image_root=args.seg7_dir, transform=train_transform, indices=seg7_train_idx, split="train")
    seg7_val = Seg7Dataset(csv_path=args.seg7_csv, image_root=args.seg7_dir, transform=val_transform, indices=seg7_val_idx, split="val")

    det10_full = Det10Dataset(csv_path=args.det10_csv, image_root=args.det10_dir, transform=train_transform)
    det10_train_idx, det10_val_idx = split_indices(len(det10_full), args.val_fraction, args.seed + 3)
    det10_train = Det10Dataset(csv_path=args.det10_csv, image_root=args.det10_dir, transform=train_transform, indices=det10_train_idx, split="train")
    det10_val = Det10Dataset(csv_path=args.det10_csv, image_root=args.det10_dir, transform=val_transform, indices=det10_val_idx, split="val")

    datasets = {
        "planesnet": (planes_train, planes_val),
        "hrplanes": (hrplanes_train, hrplanes_val),
        "fgvc": (fgvc_train, fgvc_val),
        "seg7": (seg7_train, seg7_val),
        "det10": (det10_train, det10_val),
    }

    train_loaders = {
        name: DataLoader(ds_pair[0], batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
        for name, ds_pair in datasets.items()
    }
    val_loaders = {
        name: DataLoader(ds_pair[1], batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        for name, ds_pair in datasets.items()
    }
    return train_loaders, val_loaders, class_map


def train():
    args = parse_args()
    console = Console()
    set_seed(args.seed)

    vision_device = torch.device(args.vision_device)
    text_device = torch.device(args.text_device)

    train_loaders, val_loaders, class_map = prepare_dataloaders(args)
    backbone_cfg = BackboneConfig(embed_dim=args.text_embed_dim, width=args.backbone_width)
    model = MultiTaskAircraftModel(num_fgvc_classes=len(class_map), backbone_cfg=backbone_cfg).to(vision_device)

    criterion_cls = torch.nn.CrossEntropyLoss()
    criterion_seg = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    text_encoder = TextPromptEncoder(
        vocab_size=args.text_vocab_size,
        embed_dim=backbone_cfg.embed_dim,
        device=text_device,
        target_device=vision_device,
    )

    args.log_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.log_dir / f"{args.experiment_name}_{int(time.time())}.json"
    history: List[Dict[str, float]] = []

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), TimeElapsedColumn(), console=console) as progress:
        train_task = progress.add_task("[green]Training multi-dataset model...", total=args.epochs)
        for epoch in range(1, args.epochs + 1):
            stats = defaultdict(lambda: {"loss": 0.0, "cls": 0.0, "seg": 0.0, "clip": 0.0, "total": 0.0, "correct": 0, "count": 0})
            iterators = {name: iter(loader) for name, loader in train_loaders.items()}
            schedule = []
            for name, loader in train_loaders.items():
                schedule.extend([name] * len(loader))
            random.shuffle(schedule)

            model.train()
            for dataset_name in schedule:
                loader = train_loaders[dataset_name]
                iterator = iterators[dataset_name]
                try:
                    images, labels, meta = next(iterator)
                except StopIteration:
                    iterator = iter(loader)
                    images, labels, meta = next(iterator)
                iterators[dataset_name] = iterator

                masks = meta.get("mask")
                if dataset_name in {"planesnet", "hrplanes"}:
                    images, masks = rotation_oriented_cit(images, masks, max_angle=args.max_angle)

                images = images.to(vision_device, non_blocking=True)
                labels = labels.to(vision_device)
                mask_tensor = masks.to(vision_device) if masks is not None else None

                optimizer.zero_grad()
                image_features, pyramid = model.encode(images)
                logits = model.classifiers[dataset_name](image_features)
                cls_loss = criterion_cls(logits, labels.long())

                seg_loss_value = torch.tensor(0.0, device=vision_device)
                if dataset_name == "hrplanes" and mask_tensor is not None:
                    seg_logits = model.seg_decoder(pyramid[: len(model.seg_decoder.up_blocks) + 1])
                    seg_logits = F.interpolate(seg_logits, size=mask_tensor.shape[-2:], mode="bilinear", align_corners=False)
                    seg_loss_value = criterion_seg(seg_logits, mask_tensor)

                text_meta = meta["text_label"]
                text_labels = list(text_meta) if isinstance(text_meta, list) else [text_meta]
                text_embeddings = text_encoder.encode(text_labels)
                clip_loss_value = clip_contrastive_loss(image_features, text_embeddings)

                total_loss = cls_loss + args.clip_weight * clip_loss_value
                if dataset_name == "hrplanes" and mask_tensor is not None:
                    total_loss += args.seg_weight * seg_loss_value

                total_loss.backward()

                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

                preds = torch.argmax(logits, dim=1)
                stats_entry = stats[dataset_name]
                batch_size = labels.size(0)
                stats_entry["loss"] += cls_loss.item() * batch_size
                stats_entry["cls"] += cls_loss.item() * batch_size
                stats_entry["seg"] += seg_loss_value.item() * batch_size
                stats_entry["clip"] += clip_loss_value.item() * batch_size
                stats_entry["total"] += total_loss.item() * batch_size
                stats_entry["correct"] += (preds == labels).sum().item()
                stats_entry["count"] += batch_size

            val_metrics = evaluate(model, val_loaders, criterion_cls, criterion_seg, vision_device)
            train_summary = {}
            for name, v in stats.items():
                count = max(v["count"], 1)
                train_summary[name] = {
                    "cls_loss": v["cls"] / count,
                    "seg_loss": v["seg"] / count,
                    "clip_loss": v["clip"] / count,
                    "total_loss": v["total"] / count,
                    "accuracy": v["correct"] / count,
                }
            record = {
                "epoch": epoch,
                "train": train_summary,
                "val": val_metrics,
            }
            history.append(record)
            with log_path.open("w", encoding="utf-8") as fp:
                json.dump(history, fp, indent=2)

            console.print(f"[Epoch {epoch}] Validation metrics: {val_metrics}")
            progress.advance(train_task)

    console.print(f"Training complete. Logs saved to {log_path}")


@torch.no_grad()
def evaluate(
    model: MultiTaskAircraftModel,
    val_loaders: Dict[str, DataLoader],
    criterion_cls: torch.nn.Module,
    criterion_seg: torch.nn.Module,
    device: torch.device,
) -> Dict[str, Dict[str, float]]:
    model.eval()
    results: Dict[str, Dict[str, float]] = {}

    for dataset_name, loader in val_loaders.items():
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        seg_loss_total = 0.0
        for images, labels, meta in loader:
            images = images.to(device)
            labels = labels.to(device)
            masks = meta.get("mask")
            mask_tensor = masks.to(device) if masks is not None else None

            image_features, pyramid = model.encode(images)
            logits = model.classifiers[dataset_name](image_features)
            cls_loss = criterion_cls(logits, labels.long())
            total_loss += cls_loss.item() * labels.size(0)
            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

            if dataset_name == "hrplanes" and mask_tensor is not None:
                seg_logits = model.seg_decoder(pyramid[: len(model.seg_decoder.up_blocks) + 1])
                seg_logits = F.interpolate(seg_logits, size=mask_tensor.shape[-2:], mode="bilinear", align_corners=False)
                seg_loss_total += criterion_seg(seg_logits, mask_tensor).item()

        results[dataset_name] = {
            "loss": total_loss / max(total_samples, 1),
            "accuracy": total_correct / max(total_samples, 1),
            "seg_loss": seg_loss_total / max(len(loader), 1),
        }

    return results


if __name__ == "__main__":
    train()
