"""
Simplified U-Net style decoder block used for transfer learning experiments.
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - linear combination of ops
        return self.net(x)


class UNetDecoder(nn.Module):
    """Tiny U-Net decoder that expects encoder feature maps as tuples."""

    def __init__(self, num_classes: int = 1) -> None:
        super().__init__()
        self.up_blocks = nn.ModuleList(
            [
                nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
                nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
                nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            ]
        )
        self.conv_blocks = nn.ModuleList(
            [
                ConvBlock(512, 256),
                ConvBlock(256, 128),
                ConvBlock(128, 64),
            ]
        )
        self.head = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, encoder_features: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        x = encoder_features[0]
        skips = encoder_features[1:]
        for up, conv, skip in zip(self.up_blocks, self.conv_blocks, skips):
            x = up(x)
            x = torch.cat([x, skip], dim=1)
            x = conv(x)
        return self.head(x)


__all__ = ["UNetDecoder"]
