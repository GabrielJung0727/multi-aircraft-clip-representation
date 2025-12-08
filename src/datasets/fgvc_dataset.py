"""
FGVC Aircraft dataset loader that uses the Kaggle CSV mirrors.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


Split = str
Target = int


@dataclass
class FGVCSample:
    path: Path
    class_name: str
    label: Target


class FGVCAircraftDataset(Dataset):
    """
    Lightweight FGVC Aircraft loader.

    Parameters
    ----------
    root:
        Folder containing `train.csv`, `val.csv`, `test.csv` and the
        unpacked `fgvc-aircraft-2013b` archive.
    subset:
        One of `train`, `val`, `test`.
    transform:
        Image transform pipeline.
    class_to_idx:
        Optional mapping to ensure consistent label IDs across splits.
    indices:
        Optional subset indices for controlled splits.
    """

    def __init__(
        self,
        root: str | Path,
        subset: Split = "train",
        transform: Optional[transforms.Compose] = None,
        class_to_idx: Optional[Dict[str, int]] = None,
        indices: Optional[Sequence[int]] = None,
    ) -> None:
        self.root = Path(root)
        self.subset = subset
        self.transform = transform or transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
            ]
        )

        self.images_dir = (
            self.root / "fgvc-aircraft-2013b" / "fgvc-aircraft-2013b" / "data" / "images"
        )
        if not self.images_dir.exists():
            raise FileNotFoundError(f"FGVC Aircraft images directory missing: {self.images_dir}")

        self.class_to_idx = class_to_idx or self._build_label_map(self.root / "train.csv")
        self.samples = self._load_split(self.root / f"{subset}.csv")
        if indices is not None:
            self.samples = [self.samples[i] for i in indices]

    @staticmethod
    def _build_label_map(csv_path: Path) -> Dict[str, int]:
        class_to_idx: Dict[str, int] = {}
        with csv_path.open("r", encoding="utf-8") as fp:
            reader = csv.DictReader(fp)
            for row in reader:
                class_name = row["Classes"]
                if class_name not in class_to_idx:
                    class_to_idx[class_name] = len(class_to_idx)
        return class_to_idx

    def _load_split(self, csv_path: Path) -> List[FGVCSample]:
        samples: List[FGVCSample] = []
        with csv_path.open("r", encoding="utf-8") as fp:
            reader = csv.DictReader(fp)
            for row in reader:
                filename = row["filename"]
                class_name = row["Classes"]
                label = self.class_to_idx[class_name]
                path = self.images_dir / filename
                if not path.exists():
                    continue
                samples.append(FGVCSample(path=path, class_name=class_name, label=label))
        if not samples:
            raise RuntimeError(f"No FGVC samples found for subset={self.subset}")
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        image = Image.open(sample.path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        meta: Dict[str, object] = {
            "path": str(sample.path),
            "dataset": "fgvc",
            "class_name": sample.class_name,
            "split": self.subset,
            "text_label": f"photo of {sample.class_name} aircraft",
        }
        return image, sample.label, meta


__all__ = ["FGVCAircraftDataset"]
