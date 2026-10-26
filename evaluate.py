"""Evaluate a saved RS3Mamba checkpoint offline; export metrics and class-index masks."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader

from rs3mamba_baseline.checkpoint import load_checkpoint, preprocessing_from_checkpoint
from rs3mamba_baseline.dataset import PairedTiffDataset, build_manifest, file_sha256
from rs3mamba_baseline.engine import evaluate_model
from rs3mamba_baseline.metrics import METRIC_POLICY
from rs3mamba_baseline.utils import collect_environment, resolve_device, setup_logger, write_csv, write_json
from rs3mamba_baseline.visualization import plot_confusion, save_prediction_visuals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("dataset"))
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--visualization-samples", type=int, default=8)
    args = parser.parse_args()
    if args.batch_size < 1 or args.num_workers < 0 or args.log_every < 1 or args.visualization_samples < 0:
        parser.error("Invalid batch size, worker count, logging interval or visualization count.")
    checkpoint = load_checkpoint(args.checkpoint)
    channels, mean, std = preprocessing_from_checkpoint(checkpoint)
    device = resolve_device(args.device)
    run_dir = args.checkpoint.parent.parent if args.checkpoint.parent.name == "checkpoints" else args.checkpoint.parent
    output = args.output_dir or run_dir / "evaluation" / f"{args.split}-{datetime.now():%Y%m%d-%H%M%S-%f}"
    if output.exists() and any(path.name != "console.log" for path in output.iterdir()):
        raise FileExistsError(f"Evaluation directory is occupied: {output}")
    logger = setup_logger(output / "evaluate.log")
    dataset = PairedTiffDataset(args.data_root, args.split, mean, std, channels)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False,
                        pin_memory=device.type == "cuda")
    from rs3mamba_baseline.model import model_from_checkpoint
    model = model_from_checkpoint(checkpoint).to(device)
    mask_dir = output / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)

    def save_mask(sample):
        # Preserve the TIFF suffix inside the name to avoid .tif/.tiff stem collisions.
        Image.fromarray(sample["prediction"].astype(np.uint8)).save(mask_dir / (sample["name"] + ".png"))

    result = evaluate_model(model, loader, device, nn.CrossEntropyLoss(), f"Evaluate {args.split}", args.log_every,
                            sample_limit=args.visualization_samples, prediction_callback=save_mask)
    report = dict(split=args.split, num_images=result["num_images"], checkpoint=str(args.checkpoint.resolve()),
                  checkpoint_sha256=file_sha256(args.checkpoint), checkpoint_epoch=checkpoint["epoch"],
                  input_channels=list(channels), input_bands=checkpoint["config"]["input_bands"],
                  initialization=checkpoint.get("initialization_report", checkpoint.get("initialization", {})),
                  loss=result["loss"], **result["metrics"], confusion_matrix=result["confusion"].tolist(),
                  metric_policy=METRIC_POLICY, mask_encoding=dict(background=0, sargassum=1))
    write_json(output / "metrics.json", report)
    write_csv(output / "per_image_metrics.csv", result["per_image"])
    write_csv(output / "confusion_matrix.csv", [dict(ground_truth=label, predicted_background=int(result["confusion"][i, 0]),
               predicted_sargassum=int(result["confusion"][i, 1])) for i, label in enumerate(("background", "sargassum"))])
    plot_confusion(result["confusion"], output / "confusion_matrix.png")
    for sample in result["samples"]:
        save_prediction_visuals(Path(sample["image_path"]), sample["prediction"], output / "visualizations", target=sample["target"])
    write_json(output / "dataset_manifest.json", build_manifest(args.data_root, (args.split,)))
    write_json(output / "environment.json", collect_environment(Path(__file__).resolve().parent, device))
    logger.info("Evaluation saved to %s", output.resolve())
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
