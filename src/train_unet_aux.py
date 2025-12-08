"""
Auxiliary training script for U-Net transfer learning.

The HRPlanes dataset is not yet available so this script only validates that
the pipeline wires together once the features (from CLIP/DiT) are ready.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import torch

from models.unet_decoder import UNetDecoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the UNet auxiliary decoder.")
    parser.add_argument("--feature-shape", type=int, nargs=3, default=(512, 14, 14))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=1)
    return parser.parse_args()


def dummy_encoder_features(shape: Tuple[int, int, int], batch_size: int = 2) -> Tuple[torch.Tensor, ...]:
    """Generate fake encoder features so we can validate the decoder wiring."""
    c, h, w = shape
    return (
        torch.randn(batch_size, c, h, w),
        torch.randn(batch_size, c // 2, h * 2, w * 2),
        torch.randn(batch_size, c // 4, h * 4, w * 4),
        torch.randn(batch_size, c // 8, h * 8, w * 8),
    )


def main() -> None:
    args = parse_args()
    decoder = UNetDecoder(encoder_channels=(512, 256, 128, 64), num_classes=1).to(args.device)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-3)
    criterion = torch.nn.BCEWithLogitsLoss()

    for epoch in range(args.epochs):
        features = tuple(f.to(args.device) for f in dummy_encoder_features(tuple(args.feature_shape)))
        target = torch.rand(features[-1].shape[0], 1, features[-1].shape[2], features[-1].shape[3]).to(args.device)
        optimizer.zero_grad()
        logits = decoder(features)
        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()
        print(f"[UNet Epoch {epoch+1}/{args.epochs}] loss={loss.item():.4f}")


if __name__ == "__main__":  # pragma: no cover - script entry
    main()
