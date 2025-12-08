"""
HRPlanes dataset loader.

Each JPEG image has a paired text file that stores normalized bounding boxes
in YOLO format (`class x_center y_center width height`). We convert those
boxes into binary masks so the UNet auxiliary task can train on segmentation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw
from torchvision import transforms
from torch.utils.data import Dataset
import torch


Split = str
Target = int


@dataclass
class HRPlanesSample:
    path: Path
    boxes: List[Tuple[float, float, float, float]]
    label: Target


class HRPlanesDataset(Dataset):
    """
    Loads the HRPlanes/HRPlanesv2 dataset.

    Parameters
    ----------
    root:
        Path to the folder that contains the JPG/TXT pairs.
    split:
        Logical split identifier. If `indices` is provided the split string is
        only used for metadata tags.
    transform:
        Callable applied to the image tensor.
    image_size:
        Resolution (square) used to resize every sample before tensor conversion.
    indices:
        Optional subset indices to mimic custom splits.
    """

    def __init__(
        self,
        root: str | Path,
        split: Split = "train",
        transform: Optional[transforms.Compose] = None,
        image_size: int = 512,
        indices: Optional[Sequence[int]] = None,
    ) -> None:
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"HRPlanes root directory not found: {self.root}")

        self.transform = transform or transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )
        self.mask_resize = transforms.Resize((image_size, image_size), transforms.InterpolationMode.NEAREST)
        self.split = split

        self.samples = self._discover_samples()
        if indices is not None:
            self.samples = [self.samples[i] for i in indices]

    def _discover_samples(self) -> List[HRPlanesSample]:
        samples: List[HRPlanesSample] = []
        for image_path in sorted(self.root.glob("*.jpg")):
            txt_path = image_path.with_suffix(".txt")
            boxes = self._read_boxes(txt_path) if txt_path.exists() else []
            label = 1 if boxes else 0
            samples.append(HRPlanesSample(path=image_path, boxes=boxes, label=label))
        if not samples:
            raise RuntimeError(f"No HRPlanes images found under {self.root}")
        return samples

    @staticmethod
    def _read_boxes(txt_path: Path) -> List[Tuple[float, float, float, float]]:
        boxes: List[Tuple[float, float, float, float]] = []
        with txt_path.open("r", encoding="utf-8") as fp:
            for line in fp:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                _, x_center, y_center, width, height = parts
                boxes.append((float(x_center), float(y_center), float(width), float(height)))
        return boxes

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        image = Image.open(sample.path).convert("RGB")
        mask = self._boxes_to_mask(image.size, sample.boxes)

        image = self.transform(image)
        mask = self.mask_resize(mask)
        mask_tensor = torch.from_numpy(np.array(mask, dtype=np.float32) / 255.0).unsqueeze(0)

        meta: Dict[str, object] = {
            "path": str(sample.path),
            "dataset": "hrplanes",
            "boxes": sample.boxes,
            "split": self.split,
            "text_label": "high-resolution aerial imagery with aircraft" if sample.label else "aerial background scene",
        }

        return image, sample.label, {**meta, "mask": mask_tensor}

    @staticmethod
    def _boxes_to_mask(image_size: Tuple[int, int], boxes: List[Tuple[float, float, float, float]]) -> Image.Image:
        width, height = image_size
        mask = Image.new("L", (width, height), color=0)
        draw = ImageDraw.Draw(mask)
        for x_center, y_center, box_w, box_h in boxes:
            w = box_w * width
            h = box_h * height
            cx = x_center * width
            cy = y_center * height
            x0 = max(cx - w / 2, 0)
            y0 = max(cy - h / 2, 0)
            x1 = min(cx + w / 2, width)
            y1 = min(cy + h / 2, height)
            draw.rectangle([x0, y0, x1, y1], fill=255)
        return mask


__all__ = ["HRPlanesDataset"]
