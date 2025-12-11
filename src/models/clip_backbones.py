"""
Model utilities for CLIP/DiT style backbones.

The actual CLIP weights will be loaded later via `transformers` or
`open_clip`. For now we keep lightweight nn.Module definitions so the rest
of the codebase can import consistent entrypoints.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from torch import nn


@dataclass
class BackboneConfig:
    embed_dim: int = 512
    freeze_encoder: bool = False
    width: int = 64
    depth: int = 4
    drop_path_rate: float = 0.1
    use_se: bool = True
    attn_heads: int = 8


def drop_path(x: torch.Tensor, drop_prob: float, training: bool) -> torch.Tensor:
    """Stochastic depth per sample."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return drop_path(x, self.drop_prob, self.training)


class SqueezeExcite(nn.Module):
    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.net(x)
        return x * scale


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, drop_prob: float = 0.0, use_se: bool = True) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False, groups=channels)
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SqueezeExcite(channels) if use_se else nn.Identity()
        self.drop_path = DropPath(drop_prob)
        self.act2 = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.se(x)
        x = self.drop_path(x) + residual
        return self.act2(x)


class CLIPBackbone(nn.Module):
    """CNN-based backbone inspired by CLIP's ResNet stem."""

    def __init__(self, config: Optional[BackboneConfig] = None) -> None:
        super().__init__()
        cfg = config or BackboneConfig()

        self.cfg = cfg
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
            drop_prob = cfg.drop_path_rate * (stage_idx + 1) / max(cfg.depth, 1)
            stages.append(
                nn.Sequential(
                    nn.Conv2d(channels, next_channels, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(next_channels),
                    nn.GELU(),
                    ResidualBlock(next_channels, drop_prob=drop_prob, use_se=cfg.use_se),
                )
            )
            channels = next_channels

        self.stage_channels = [cfg.width] + [cfg.width * (2 ** i) for i in range(1, cfg.depth)]
        self.stages = nn.ModuleList(stages)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(channels, cfg.embed_dim)

        if cfg.freeze_encoder:
            for param in self.parameters():
                param.requires_grad = False
            for param in self.proj.parameters():
                param.requires_grad = True

    def forward_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        feats = self.stem(x)
        pyramid: List[torch.Tensor] = []
        for stage in self.stages:
            feats = stage(feats)
            pyramid.append(feats)
        pooled = self.avg_pool(feats).flatten(1)
        return pooled, tuple(pyramid[::-1])  # deepest first for decoder convenience

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled, _ = self.forward_features(x)
        return self.proj(pooled)


class DiTBackbone(nn.Module):
    """Lightweight transformer refinement over pooled tokens (DiT-style)."""

    def __init__(self, embed_dim: int = 512, depth: int = 4, num_heads: int = 8, mlp_ratio: float = 4.0, drop_path_rate: float = 0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1024 + 1, embed_dim))  # supports up to 32x32 tokens

        layers: List[nn.Module] = []
        for idx in range(depth):
            drop_prob = drop_path_rate * (idx + 1) / max(depth, 1)
            layers.append(
                TransformerBlock(embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, drop_path=drop_prob)
            )
        self.layers = nn.ModuleList(layers)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) -> tokens
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)  # (B, HW, C)
        cls = self.cls_token.expand(b, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        max_pos = min(tokens.size(1), self.pos_embed.size(1))
        tokens = tokens + self.pos_embed[:, :max_pos, :]

        for blk in self.layers:
            tokens = blk(tokens)
        tokens = self.norm(tokens)
        cls_out = tokens[:, 0]
        return cls_out


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 8, mlp_ratio: float = 4.0, drop: float = 0.0, attn_drop: float = 0.0, drop_path: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=attn_drop, batch_first=True)
        self.drop_path = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False)
        x = x + self.drop_path(attn_out)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


__all__ = ["CLIPBackbone", "DiTBackbone", "BackboneConfig"]
