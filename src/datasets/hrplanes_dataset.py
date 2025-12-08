"""
Placeholder dataset wrapper for HRPlanes / HRPlanesv2.

Once the dataset download completes, replace this stub with an actual
implementation that mirrors the signature of `PlanesNetDataset`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from torch.utils.data import Dataset


class HRPlanesDataset(Dataset):
    """
    Stub version that makes the training scripts importable even before the
    HRPlanes files arrive on disk.
    """

    def __init__(self, root: str | Path, transform: Optional[Callable] = None) -> None:
        raise RuntimeError(
            "HRPlanesDataset is not yet implemented because the HRPlanes archive "
            "has not been fully downloaded. Re-run the setup once the data is available."
        )


__all__ = ["HRPlanesDataset"]
