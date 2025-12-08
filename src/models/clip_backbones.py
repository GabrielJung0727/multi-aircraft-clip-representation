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
    freeze_encoder: bool = True


class CLIPBackbone(nn.Module):
    """Very small stand-in for a CLIP vision encoder."""

    def __init__(self, config: Optional[BackboneConfig] = None) -> None:
        super().__init__()
        cfg = config or BackboneConfig()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(64, cfg.embed_dim)
        if cfg.freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - placeholder behaviour
        feats = self.encoder(x).flatten(1)
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
