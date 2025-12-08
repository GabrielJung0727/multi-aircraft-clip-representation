"""
Dataset helpers for the PlanesNet satellite imagery corpus.

The loader below focuses on the subset that has already been downloaded:
`data/planesnet/{planesnet, scenes, planesnet.json}`.
The JSON metadata is optional—if it is missing or if particular fields do
not exist, the loader gracefully falls back to inferring the label from the
file name prefix (0__ -> no-plane, 1__ -> plane).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from PIL import Image
from torch.utils.data import Dataset


Split = str
Target = int


@dataclass(frozen=True)
class PlanesNetSample:
    """Small container to keep every sample consistent throughout the pipeline."""

    path: Path
    label: Target
    split: Split
    meta: Dict[str, str]


class PlanesNetDataset(Dataset):
    """
    Minimal PyTorch dataset that understands the local PlanesNet folder layout.

    Parameters
    ----------
    root:
        Path to `data/planesnet`.
    split:
        Arbitrary split identifier. If the metadata JSON exposes a `split`
        field, rows will be filtered accordingly. Otherwise every image is
        treated as part of the given split.
    transform:
        Callable applied to the PIL image before returning it.
    include_scenes:
        When True, loads the large `scenes/*.png` files instead of the cropped
        `planesnet/*.png` patches. This is handy for visualization notebooks.
    metadata_file:
        Optional override for the JSON file. Defaults to `planesnet.json`.
    """

    def __init__(
        self,
        root: str | Path,
        split: Split = "train",
        transform: Optional[Callable] = None,
        include_scenes: bool = False,
        metadata_file: Optional[str | Path] = None,
    ) -> None:
        self.root = Path(root)
        self.transform = transform
        self.images_dir = self.root / ("scenes" if include_scenes else "planesnet")
        if not self.images_dir.exists():
            raise FileNotFoundError(
                f"Expected directory {self.images_dir} to exist. "
                "Verify that the PlanesNet archive is extracted under data/planesnet."
            )

        metadata_path = Path(metadata_file) if metadata_file else self.root / "planesnet.json"
        self.samples: List[PlanesNetSample] = self._build_samples(metadata_path, split)

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #
    def _build_samples(self, metadata_path: Path, split: Split) -> List[PlanesNetSample]:
        rows = self._read_metadata(metadata_path) if metadata_path.exists() else None
        if rows:
            filtered = [row for row in rows if row.get("split", split) == split]
            candidates = filtered if filtered else rows
            samples = [
                self._sample_from_metadata(row, split=split) for row in candidates if row.get("filename")
            ]
        else:
            samples = []

        if not samples:
            # Fall back to the directory listing if metadata is missing or malformed.
            for path in sorted(self.images_dir.glob("*.png")):
                samples.append(
                    PlanesNetSample(
                        path=path,
                        label=self._label_from_name(path.name),
                        split=split,
                        meta={"source": "filesystem"},
                    )
                )

        return samples

    def _sample_from_metadata(self, row: Dict[str, str], split: Split) -> PlanesNetSample:
        filename = row.get("filename") or row.get("image_id") or row.get("patch_id")
        if not filename:
            raise ValueError("Metadata row is missing the filename/image identifier")

        path = self.images_dir / filename
        if not path.exists():
            # Some JSON files keep a relative path such as "planesnet/xxx.png".
            alt = self.root / filename
            if alt.exists():
                path = alt
            else:
                raise FileNotFoundError(f"{filename} referenced in metadata but not found on disk.")

        label_value = row.get("label") or row.get("tag") or row.get("category")
        label = self._label_from_name(filename) if label_value is None else int(label_value in ("plane", 1, "1"))

        return PlanesNetSample(path=path, label=label, split=row.get("split", split), meta=row)

    @staticmethod
    def _read_metadata(metadata_path: Path) -> Optional[List[Dict[str, str]]]:
        try:
            with metadata_path.open("r", encoding="utf-8") as fp:
                raw = json.load(fp)
        except json.JSONDecodeError:
            return None

        candidates: Sequence = []
        if isinstance(raw, list):
            candidates = raw
        elif isinstance(raw, dict):
            for key in ("items", "entries", "data", "rows"):
                val = raw.get(key)
                if isinstance(val, list):
                    candidates = val
                    break

        return [row for row in candidates if isinstance(row, dict)] or None

    @staticmethod
    def _label_from_name(filename: str) -> Target:
        return 1 if filename.startswith("1__") else 0

    # --------------------------------------------------------------------- #
    # Dataset protocol
    # --------------------------------------------------------------------- #
    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Image.Image, Target, Dict[str, str]]:
        sample = self.samples[idx]
        image = Image.open(sample.path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, sample.label, sample.meta

    def __repr__(self) -> str:
        return f"PlanesNetDataset(num_samples={len(self)}, root='{self.root}', dir='{self.images_dir.name}')"


__all__ = ["PlanesNetDataset", "PlanesNetSample"]
