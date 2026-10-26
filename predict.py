"""Predict a 256x256 four-band TIFF or folder using saved normalization and weights."""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import tifffile
import torch

from rs3mamba_baseline.checkpoint import load_checkpoint, preprocessing_from_checkpoint
from rs3mamba_baseline.dataset import file_sha256, image_files, normalize_image, read_image, read_mask
from rs3mamba_baseline.engine import resized_logits
from rs3mamba_baseline.metrics import METRIC_POLICY, confusion_matrix, metrics_from_confusion
from rs3mamba_baseline.utils import collect_environment, resolve_device, setup_logger, write_csv, write_json
from rs3mamba_baseline.visualization import save_prediction_visuals


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path, help="Optional TIFF labels with matching filenames")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save-probabilities", action="store_true")
    parser.add_argument("--log-every", type=int, default=20)
    args = parser.parse_args()
    if args.log_every < 1:
        parser.error("log-every must be positive")
    if args.input.is_file():
        if args.input.suffix.lower() not in {".tif", ".tiff"}:
            parser.error("input must be a TIFF file")
        paths = [args.input]
    else:
        paths = image_files(args.input)
    if args.mask_dir:
        missing = [path.name for path in paths if not (args.mask_dir / path.name).is_file()]
        if missing:
            raise FileNotFoundError(f"Missing labels: {missing[:5]}")
    checkpoint = load_checkpoint(args.checkpoint)
    channels, mean, std = preprocessing_from_checkpoint(checkpoint)
    device = resolve_device(args.device)
    run_dir = args.checkpoint.parent.parent if args.checkpoint.parent.name == "checkpoints" else args.checkpoint.parent
    output = args.output_dir or run_dir / "predictions" / f"{datetime.now():%Y%m%d-%H%M%S-%f}"
    if output.exists() and any(path.name != "console.log" for path in output.iterdir()):
        raise FileExistsError(f"Prediction directory is occupied: {output}")
    logger = setup_logger(output / "predict.log")
    from rs3mamba_baseline.model import model_from_checkpoint
    model = model_from_checkpoint(checkpoint).to(device).eval()
    matrix = np.zeros((2, 2), dtype=np.int64)
    per_image, records = [], []
    for index, path in enumerate(paths, 1):
        image = read_image(path)
        images = normalize_image(image, channels, mean, std).unsqueeze(0).to(device)
        logits = resized_logits(model, images, image.shape[:2])
        if not torch.isfinite(logits).all():
            raise FloatingPointError(f"Nonfinite prediction logits: {path}")
        prediction = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
        target = read_mask(args.mask_dir / path.name, image.shape[:2]) if args.mask_dir else None
        save_prediction_visuals(path, prediction, output, target=target)
        record = dict(name=path.name, sha256=file_sha256(path), height=image.shape[0], width=image.shape[1])
        if target is not None:
            cm = confusion_matrix(target, prediction)
            matrix += cm
            per_image.append(dict(name=path.name, **metrics_from_confusion(cm)))
            record["mask_sha256"] = file_sha256(args.mask_dir / path.name)
        if args.save_probabilities:
            probability_dir = output / "probabilities"
            probability_dir.mkdir(parents=True, exist_ok=True)
            tifffile.imwrite(probability_dir / path.name, logits.softmax(1)[0, 1].cpu().numpy().astype(np.float32))
        records.append(record)
        if index == 1 or index % args.log_every == 0 or index == len(paths):
            logger.info("Predicted %s/%s: %s", index, len(paths), path.name)
    report = dict(num_images=len(paths), checkpoint=str(args.checkpoint.resolve()),
                  checkpoint_sha256=file_sha256(args.checkpoint), checkpoint_epoch=checkpoint["epoch"],
                  input_channels=list(channels), input_bands=checkpoint["config"]["input_bands"],
                  mask_encoding=dict(background=0, sargassum=1), probabilities_saved=args.save_probabilities,
                  display="NIR,R,G per-image 2%/98% stretch; display only",
                  georeferencing="Pixel dimensions preserved; GeoTIFF spatial metadata is not copied.", inputs=records)
    if args.mask_dir:
        report.update(metrics=metrics_from_confusion(matrix), confusion_matrix=matrix.tolist(), metric_policy=METRIC_POLICY)
        write_csv(output / "per_image_metrics.csv", per_image)
    write_json(output / "prediction_manifest.json", report)
    write_json(output / "environment.json", collect_environment(Path(__file__).resolve().parent, device))
    logger.info("Predictions saved to %s", output.resolve())


if __name__ == "__main__":
    main()
