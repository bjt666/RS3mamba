"""Paired TIFF tiles, train-only normalization and path-independent manifests.

The source image convention is HWC, 256 x 256 x 4, Blue/Green/Red/NIR.
No display stretch, division by 255 or automatic RGB/BGR conversion is applied.
Torch is imported lazily so data preparation also runs in a CPU-only environment
containing only NumPy, tifffile and Pillow.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import tifffile

if TYPE_CHECKING:
    import torch

DEFAULT_CHANNELS = (3, 2, 1)
DEFAULT_BANDS = ("NIR", "R", "G")
SOURCE_BANDS = ("B", "G", "R", "NIR")
TILE_SHAPE = (256, 256)


def validate_channels(channels: tuple[int, ...]) -> tuple[int, ...]:
    channels = tuple(channels)
    if (len(channels) != 3 or len(set(channels)) != 3
            or any(type(channel) is not int or channel not in range(4) for channel in channels)):
        raise ValueError("Select three distinct zero-based channels from B,G,R,NIR (0,1,2,3).")
    return channels


def image_files(directory: Path) -> list[Path]:
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {directory}")
    paths = sorted(path for path in directory.iterdir()
                   if path.is_file() and path.suffix.lower() in {".tif", ".tiff"})
    if not paths:
        raise FileNotFoundError(f"No TIFF images found in {directory}")
    return paths


def paired_files(root: Path, split: str) -> list[tuple[Path, Path]]:
    root = Path(root)
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Expected train, val or test split, got {split!r}.")
    images = {path.name: path for path in image_files(root / "images" / split)}
    masks = {path.name: path for path in image_files(root / "masks" / split)}
    if images.keys() != masks.keys():
        raise ValueError(
            f"Unpaired files in {split}: missing masks={sorted(images.keys() - masks.keys())[:5]}, "
            f"missing images={sorted(masks.keys() - images.keys())[:5]}"
        )
    return [(images[name], masks[name]) for name in sorted(images)]


def read_image(path: Path) -> np.ndarray:
    image = tifffile.imread(path).astype(np.float32)
    expected = (*TILE_SHAPE, 4)
    if image.shape != expected:
        raise ValueError(f"Expected HWC {expected} TIFF in B,G,R,NIR order: {path}; got {image.shape}")
    if not np.isfinite(image).all():
        raise ValueError(f"Nonfinite image values: {path}")
    return image


def read_mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    mask = tifffile.imread(path)
    if mask.ndim != 2 or mask.shape != tuple(shape):
        raise ValueError(f"Expected mask shape {shape}, got {mask.shape}: {path}")
    if not np.isin(mask, [0, 1, 255]).all():
        raise ValueError(f"Binary masks must contain only 0, 1 or 255: {path}")
    # Source 255 is foreground, not ignore. Background 0 remains a valid class.
    return (mask > 0).astype(np.int64)


def normalized_array(image: np.ndarray, channels: tuple[int, ...], mean: np.ndarray,
                     std: np.ndarray) -> np.ndarray:
    channels = validate_channels(channels)
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    if mean.shape != (3,) or std.shape != (3,):
        raise ValueError("Normalization statistics must match the three selected channels.")
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
        raise ValueError("Normalization statistics must be finite, with positive standard deviations.")
    selected = image[..., list(channels)]
    selected = (selected - mean.reshape(1, 1, -1)) / np.maximum(std.reshape(1, 1, -1), 1e-6)
    return np.ascontiguousarray(selected.transpose(2, 0, 1), dtype=np.float32)


def normalize_image(image: np.ndarray, channels: tuple[int, ...], mean: np.ndarray,
                    std: np.ndarray) -> torch.Tensor:
    import torch

    return torch.from_numpy(normalized_array(image, channels, mean, std))


class PairedTiffDataset:
    """Map-style dataset accepted by torch DataLoader without eager torch imports."""

    def __init__(self, root: Path, split: str, mean: np.ndarray, std: np.ndarray,
                 channels: tuple[int, ...] = DEFAULT_CHANNELS, augment: bool = False) -> None:
        self.image_dir = Path(root) / "images" / split
        self.mask_dir = Path(root) / "masks" / split
        self.pairs = paired_files(Path(root), split)
        self.names = [image.name for image, _ in self.pairs]
        self.channels = validate_channels(channels)
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        # Validate once before any DataLoader workers are started.
        normalized_array(np.zeros((*TILE_SHAPE, 4), dtype=np.float32),
                         self.channels, self.mean, self.std)
        self.augment = bool(augment)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict:
        import torch

        image_path, mask_path = self.pairs[index]
        image = read_image(image_path)
        mask = read_mask(mask_path, image.shape[:2])
        if self.augment:
            if random.random() < 0.5:
                image, mask = image[:, ::-1, :], mask[:, ::-1]
            if random.random() < 0.5:
                image, mask = image[::-1, :, :], mask[::-1, :]
            turns = random.randrange(4)
            if turns:
                image = np.rot90(image, turns, axes=(0, 1))
                mask = np.rot90(mask, turns)
        return {
            "image": normalize_image(image, self.channels, self.mean, self.std),
            "mask": torch.from_numpy(np.ascontiguousarray(mask, dtype=np.int64)),
            "name": image_path.name,
            "image_path": str(image_path),
        }


def compute_train_stats(root: Path, channels: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    channels = validate_channels(channels)
    total = np.zeros(len(channels), dtype=np.float64)
    total_sq = np.zeros(len(channels), dtype=np.float64)
    count = 0
    for path in image_files(Path(root) / "images" / "train"):
        image = read_image(path)[..., list(channels)].astype(np.float64)
        total += image.sum(axis=(0, 1))
        total_sq += np.square(image).sum(axis=(0, 1))
        count += image.shape[0] * image.shape[1]
    mean = total / count
    std = np.sqrt(np.maximum(total_sq / count - np.square(mean), 1e-12))
    return mean.astype(np.float32), std.astype(np.float32)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_manifest(root: Path, splits: tuple[str, ...]) -> dict:
    """Record names and content hashes; the fingerprint is independent of root path."""
    root = Path(root)
    records = []
    for split in splits:
        # Detect image/mask mismatch even before training reads the first item.
        pairs = paired_files(root, split)
        for kind, member in (("images", 0), ("masks", 1)):
            for pair in pairs:
                path = pair[member]
                records.append({"path": path.relative_to(root).as_posix(),
                                "size": path.stat().st_size, "sha256": file_sha256(path)})
    fingerprint = hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()
    return {"algorithm": "sha256", "splits": list(splits), "fingerprint": fingerprint, "files": records}
