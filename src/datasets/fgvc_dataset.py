"""
Placeholder dataset wrapper for the FGVC Aircraft benchmark.

The dataset is still downloading, so this module intentionally raises a
descriptive error to keep callers aware of the missing dependency.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from torch.utils.data import Dataset


class FGVCAircraftDataset(Dataset):
    """Stub dataset that will be fully implemented after the download finishes."""

    def __init__(self, root: str | Path, transform: Optional[Callable] = None) -> None:
        raise RuntimeError(
            "FGVCAircraftDataset is waiting for the FGVC Aircraft files to finish downloading."
        )


__all__ = ["FGVCAircraftDataset"]
