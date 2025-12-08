"""
Classifier heads shared between PlanesNet / HRPlanes / FGVC.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class LinearClassifier(nn.Module):
    """Two-layer classifier with dropout for better convergence."""

    def __init__(
        self,
        in_dim: int = 512,
        num_classes: int = 2,
        dropout: float = 0.1,
        hidden_dim: Optional[int] = 256,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Dropout(dropout)]
        last_dim = in_dim

        if hidden_dim and hidden_dim > 0:
            layers.extend(
                [
                    nn.Linear(in_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            last_dim = hidden_dim

        layers.append(nn.Linear(last_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:  # pragma: no cover - simple sequential module
        return self.net(features)


__all__ = ["LinearClassifier"]
