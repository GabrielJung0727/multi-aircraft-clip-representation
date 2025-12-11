"""
Lightweight loader for Seg-7 (iSAID/Vaihingen patches + captions).

Assumptions:
- Images live in `image_root` (e.g., data/data2/Seg-7).
- CSV has columns: title, filename.
- Binary label is inferred: 1 if the caption mentions "plane" (any case), else 0.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset


class Seg7Dataset(Dataset):
    def __init__(
        self,
        csv_path: Path,
        image_root: Path,
        transform: Optional[Callable] = None,
        indices: Optional[List[int]] = None,
        split: str = "train",
        sep: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.image_root = image_root
        self.transform = transform
        self.split = split

        # Auto-detect separator to avoid parsing errors (commas vs tabs).
        df = pd.read_csv(csv_path, sep=sep, engine="python")
        self.titles: List[str] = df["title"].tolist()
        self.filenames: List[str] = df["filename"].tolist()

        if indices is not None:
            self.titles = [self.titles[i] for i in indices]
            self.filenames = [self.filenames[i] for i in indices]

    def __len__(self) -> int:
        return len(self.filenames)

    def _infer_label(self, text: str) -> int:
        return 1 if re.search(r"plane", text, re.IGNORECASE) else 0

    def __getitem__(self, idx: int) -> Tuple:
        img_path = self.image_root / self.filenames[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Provide a dummy mask so collate works uniformly across datasets.
        h, w = image.shape[-2], image.shape[-1]
        mask_tensor = torch.zeros((1, h, w), dtype=torch.float32)

        text_label = self.titles[idx]
        label = self._infer_label(text_label)
        meta: Dict[str, object] = {"text_label": text_label, "mask": mask_tensor}
        return image, label, meta


__all__ = ["Seg7Dataset"]
