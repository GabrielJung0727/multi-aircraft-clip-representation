"""
Loader for Det-10_part1 captions + images (binary aircraft presence inferred from text).

Expected CSV columns: title, filename
Images live under image_root (e.g., data/data2/Det-10_part1)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset


AIR_KEYWORDS = re.compile(r"(plane|aircraft|airplane|jet|drone|helicopter|airport)", re.IGNORECASE)


class Det10Dataset(Dataset):
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

        df = pd.read_csv(csv_path, sep=sep, engine="python")
        titles: List[str] = df["title"].tolist()
        files: List[str] = df["filename"].tolist()

        # Keep only rows with existing images.
        valid_titles: List[str] = []
        valid_files: List[str] = []
        for t, f in zip(titles, files):
            if (image_root / f).exists():
                valid_titles.append(t)
                valid_files.append(f)

        self.titles = valid_titles
        self.filenames = valid_files

        if indices is not None:
            self.titles = [self.titles[i] for i in indices]
            self.filenames = [self.filenames[i] for i in indices]

    def __len__(self) -> int:
        return len(self.filenames)

    def _infer_label(self, text: str) -> int:
        return 1 if AIR_KEYWORDS.search(text) else 0

    def __getitem__(self, idx: int) -> Tuple:
        img_path = self.image_root / self.filenames[idx]
        if not img_path.exists():
            # Fallback: return a zero image to avoid crashing long runs.
            image = Image.new("RGB", (256, 256), color=(0, 0, 0))
        else:
            image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        h, w = image.shape[-2], image.shape[-1]
        mask_tensor = torch.zeros((1, h, w), dtype=torch.float32)

        text_label = self.titles[idx]
        label = self._infer_label(text_label)
        meta: Dict[str, object] = {"text_label": text_label, "mask": mask_tensor}
        return image, label, meta


__all__ = ["Det10Dataset"]
