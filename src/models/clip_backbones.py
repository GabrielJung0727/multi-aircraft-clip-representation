"""
Model utilities for CLIP/DiT style backbones.

The actual CLIP weights will be loaded later via `transformers` or
`open_clip`. For now we keep lightweight nn.Module definitions so the rest
of the codebase can import consistent entrypoints.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn


@dataclass
class BackboneConfig:
    embed_dim: int = 512
    freeze_encoder: bool = False
    width: int = 64
    depth: int = 4


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.block(x) + x)


class CLIPBackbone(nn.Module):
    """CNN-based backbone inspired by CLIP's ResNet stem."""

    def __init__(self, config: Optional[BackboneConfig] = None) -> None:
        super().__init__()
        cfg = config or BackboneConfig()

        self.stem = nn.Sequential(
            nn.Conv2d(3, cfg.width, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(cfg.width),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        stages = []
        channels = cfg.width
        for stage_idx in range(cfg.depth):
            next_channels = channels * 2 if stage_idx > 0 else channels
            stages.append(
                nn.Sequential(
                    nn.Conv2d(channels, next_channels, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(next_channels),
                    nn.GELU(),
                    ResidualBlock(next_channels),
                )
            )
            channels = next_channels

        self.stages = nn.Sequential(*stages)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(channels, cfg.embed_dim)

        if cfg.freeze_encoder:
            for param in self.parameters():
                param.requires_grad = False
            for param in self.proj.parameters():
                param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.stem(x)
        feats = self.stages(feats)
        feats = self.avg_pool(feats).flatten(1)
        return self.proj(feats)


class DiTBackbone(nn.Module):
    """Placeholder for a Diffusion Transformer style encoder."""

    def __init__(self, embed_dim: int = 512, depth: int = 4) -> None:
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.zeros(1, embed_dim))
        self.layers = nn.ModuleList([nn.Linear(embed_dim, embed_dim) for _ in range(depth)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x.mean(dim=(2, 3))
        out = out + self.pos_embedding
        for layer in self.layers:
            out = torch.tanh(layer(out))
        return out


__all__ = ["CLIPBackbone", "DiTBackbone", "BackboneConfig"]
