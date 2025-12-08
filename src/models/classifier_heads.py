"""
Classifier heads shared between PlanesNet / HRPlanes / FGVC.
"""

from __future__ import annotations

import torch
from torch import nn


class LinearClassifier(nn.Module):
    """Single linear layer with dropout to keep experiments lightweight."""

    def __init__(self, in_dim: int = 512, num_classes: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_dim, num_classes))

    def forward(self, features: torch.Tensor) -> torch.Tensor:  # pragma: no cover - simple sequential module
        return self.net(features)


__all__ = ["LinearClassifier"]
