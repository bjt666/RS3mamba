"""Check the paired GF-1 TIFF dataset and compute normalization from train only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from rs3mamba_baseline.dataset import (
    DEFAULT_CHANNELS, SOURCE_BANDS, TILE_SHAPE, build_manifest, compute_train_stats,
    paired_files, read_image, read_mask, validate_channels,
)


def save_preview(image: np.ndarray, mask: np.ndarray, path: Path,
                 channels: tuple[int, ...]) -> None:
    from PIL import Image, ImageDraw

    selected = image[..., list(channels)]
    low, high = np.percentile(selected, [2, 98], axis=(0, 1))
    rgb = np.uint8(np.clip((selected - low) / np.maximum(high - low, 1e-6), 0, 1) * 255)
    target = np.repeat((mask * 255).astype(np.uint8)[..., None], 3, axis=2)
    overlay = rgb.astype(np.float32)
    foreground = mask.astype(bool)
    overlay[foreground] = 0.55 * overlay[foreground] + 0.45 * np.array([255, 210, 0])
    height, width = mask.shape
    canvas = Image.new("RGB", (width * 3, height + 48), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (title, panel) in enumerate([
        ("False color (display stretch)", rgb), ("Ground truth: white=target", target),
        ("Yellow: target overlay", overlay.astype(np.uint8)),
    ]):
        canvas.paste(Image.fromarray(panel), (index * width, 24))
        draw.text((index * width + 4, 5), title, fill="black")
    draw.text((4, height + 28), "2-98% stretch is for this preview only; training uses raw TIFF values.", fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def inspect_dataset(root: Path, channels: tuple[int, ...] = DEFAULT_CHANNELS,
                    splits: tuple[str, ...] = ("train", "val", "test"),
                    output: Path | None = None, preview_samples: int = 3) -> dict:
    root = Path(root)
    channels = validate_channels(channels)
    if not splits or len(set(splits)) != len(splits):
        raise ValueError("Select nonempty, distinct dataset splits.")
    if preview_samples < 0:
        raise ValueError("preview_samples must be nonnegative.")
    mean, std = compute_train_stats(root, channels)
    hashes: dict[str, list[str]] = {}
    summary = {}
    for split in splits:
        pairs = paired_files(root, split)
        foreground_pixels = 0
        total_pixels = 0
        nonempty_images = 0
        all_zero_images = []
        channel_min = np.full(3, np.inf, dtype=np.float64)
        channel_max = np.full(3, -np.inf, dtype=np.float64)
        for index, (image_path, mask_path) in enumerate(pairs):
            image = read_image(image_path)
            mask = read_mask(mask_path, image.shape[:2])
            selected = np.ascontiguousarray(image[..., list(channels)])
            digest = hashlib.sha256(selected.tobytes()).hexdigest()
            hashes.setdefault(digest, []).append(f"{split}/{image_path.name}")
            pixels = int(mask.sum())
            foreground_pixels += pixels
            total_pixels += int(mask.size)
            nonempty_images += int(pixels > 0)
            if not np.any(selected):
                all_zero_images.append(image_path.name)
            channel_min = np.minimum(channel_min, selected.min(axis=(0, 1)))
            channel_max = np.maximum(channel_max, selected.max(axis=(0, 1)))
            if output is not None and index < preview_samples:
                save_preview(image, mask, Path(output) / "previews" / split / (image_path.name + ".png"), channels)
        summary[split] = {
            "num_images": len(pairs), "shape": [*TILE_SHAPE, 4],
            "foreground_pixels": foreground_pixels, "total_pixels": total_pixels,
            "foreground_fraction": foreground_pixels / total_pixels,
            "images_with_foreground": nonempty_images,
            "all_zero_selected_images": all_zero_images,
            "selected_channel_min": channel_min.tolist(), "selected_channel_max": channel_max.tolist(),
        }
    duplicates = [paths for paths in hashes.values() if len(paths) > 1]
    cross_split_duplicates = [paths for paths in duplicates
                              if len({path.split("/", 1)[0] for path in paths}) > 1]
    report = {
        "data_root": str(root.resolve()), "input_channels": list(channels),
        "input_bands": [SOURCE_BANDS[channel] for channel in channels],
        "source_bands": list(SOURCE_BANDS), "mean": mean.tolist(), "std": std.tolist(),
        "normalization_source": "train", "divide_by_255": False,
        "mask_policy": "0=background; source 1 or 255=foreground; no source ignore value",
        "splits": summary, "duplicate_selected_images": duplicates,
        "cross_split_duplicates": cross_split_duplicates,
        "status": "failed" if cross_split_duplicates else "passed",
        "limitations": "Content hashes do not detect neighboring or overlapping tiles from one scene.",
    }
    if output is not None:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        (output / "stats.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        (output / "dataset_manifest.json").write_text(
            json.dumps(build_manifest(root, splits), indent=2, allow_nan=False), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/data_check"))
    parser.add_argument("--input-channels", type=int, nargs=3, default=list(DEFAULT_CHANNELS),
                        metavar=("C1", "C2", "C3"), help="Zero-based B,G,R,NIR indices; default 3 2 1")
    parser.add_argument("--splits", nargs="+", choices=["train", "val", "test"], default=["train", "val", "test"])
    parser.add_argument("--preview-samples", type=int, default=3)
    args = parser.parse_args()
    report = inspect_dataset(args.data_root, tuple(args.input_channels), tuple(args.splits),
                             args.output_dir, args.preview_samples)
    print(json.dumps({key: report[key] for key in ["status", "input_bands", "mean", "std", "splits",
                                                 "cross_split_duplicates"]}, indent=2, allow_nan=False))
    print(f"Report and previews: {args.output_dir.resolve()}")
    if report["cross_split_duplicates"]:
        raise SystemExit("Cross-split identical selected images detected; resolve the split before training.")


if __name__ == "__main__":
    main()
