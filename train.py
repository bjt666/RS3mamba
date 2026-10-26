"""Train RS3Mamba or resume a full checkpoint at a completed epoch boundary."""
from __future__ import annotations

import argparse
import copy
import logging
import time
from datetime import datetime
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from rs3mamba_baseline.checkpoint import (
    SCHEMA_VERSION, atomic_torch_save, load_checkpoint, make_checkpoint,
    preprocessing_from_checkpoint, restore_rng,
)
from rs3mamba_baseline.config import DEFAULTS, default_config, read_config, save_config, validate_config
from rs3mamba_baseline.dataset import PairedTiffDataset, build_manifest, compute_train_stats
from rs3mamba_baseline.engine import evaluate_model, train_epoch
from rs3mamba_baseline.utils import (
    collect_environment, resolve_device, save_pip_freeze, seed_worker, set_seed,
    setup_logger, utc_now, write_history, write_json,
)
from rs3mamba_baseline.visualization import plot_history, save_prediction_visuals


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--resume", type=Path, help="Full checkpoints/last.pt; settings are restored automatically")
    for field in ("data_root", "output_dir", "cnn_checkpoint", "vmamba_checkpoint"):
        parser.add_argument("--" + field.replace("_", "-"), type=Path)
    for field in ("epochs", "batch_size", "accumulation_steps", "num_workers", "seed", "log_every",
                  "visualization_samples", "max_train_batches", "max_val_batches"):
        parser.add_argument("--" + field.replace("_", "-"), type=int)
    for field in ("encoder_lr", "decoder_lr", "learning_rate", "weight_decay"):
        parser.add_argument("--" + field.replace("_", "-"), type=float)
    parser.add_argument("--device")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false", default=None)
    return parser.parse_args()


def prepare_config(args, checkpoint=None):
    overrides = read_config(args.config)
    overrides.update({key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
                      if key in DEFAULTS and value is not None})
    config = default_config()
    if checkpoint is not None:
        required = {"optimizer", "scheduler", "scaler", "rng_state", "history", "dataset_fingerprint", "best_iou", "best_epoch"}
        if checkpoint.get("schema_version") != SCHEMA_VERSION or not required.issubset(checkpoint):
            raise ValueError("Resume requires a complete RS3Mamba training checkpoint.")
        config.update(checkpoint["config"])
        if args.resume.parent.name == "checkpoints":
            config["output_dir"] = str(args.resume.resolve().parent.parent)
        mutable = {"data_root", "output_dir", "device", "num_workers", "log_every", "visualization_samples"}
        for key, value in overrides.items():
            if key == "output_dir" and value is None:
                continue
            if key not in mutable and value != config[key]:
                raise ValueError(f"Cannot change {key} while resuming. Start a new experiment for different settings.")
        overrides = {key: value for key, value in overrides.items() if not (key == "output_dir" and value is None)}
    config.update(overrides)
    validate_config(config)
    if config["output_dir"] is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        kind = "smoke" if config["max_train_batches"] or config["max_val_batches"] else "baseline"
        config["output_dir"] = f"outputs/rs3mamba-{kind}-seed{config['seed']}-{stamp}"
    for key in ("data_root", "output_dir"):
        config[key] = str(Path(config[key]).resolve())
    return config


def make_optimizer(model, config):
    if config["optimizer"] == "sgd":
        return torch.optim.SGD(model.parameters(), lr=config["learning_rate"], momentum=config["momentum"],
                               weight_decay=config["weight_decay"])
    encoder_ids = {id(parameter) for name in ("backbone", "vssm_encoder", "stem")
                   for parameter in getattr(model, name).parameters()}
    encoder, decoder = [], []
    for parameter in model.parameters():
        if parameter.requires_grad:
            (encoder if id(parameter) in encoder_ids else decoder).append(parameter)
    if not encoder or not decoder:
        raise ValueError("Expected encoder and fusion/decoder parameter groups.")
    return torch.optim.AdamW([dict(params=encoder, lr=config["encoder_lr"], name="encoder"),
                              dict(params=decoder, lr=config["decoder_lr"], name="fusion_decoder")],
                             weight_decay=config["weight_decay"])


def make_scheduler(optimizer, config):
    if config["scheduler"] == "multistep":
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config["milestones"], gamma=config["gamma"])
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])


def main():
    args = parse_args()
    checkpoint = load_checkpoint(args.resume) if args.resume else None
    config = prepare_config(args, checkpoint)
    device = resolve_device(config["device"])
    if config["amp"] and device.type != "cuda":
        raise ValueError("AMP is supported only on CUDA in this project.")
    output = Path(config["output_dir"])
    if checkpoint:
        if args.resume.parent.name != "checkpoints":
            raise ValueError("Resume requires <run>/checkpoints/last.pt (or best.pt). Copy the whole run directory when moving it.")
        if output != args.resume.resolve().parent.parent:
            raise ValueError("Resume in the checkpoint's run directory. Copy the whole run first when moving it.")
    if output.exists() and checkpoint is None and any(path.name not in {"console.log", "nohup.out"} for path in output.iterdir()):
        raise FileExistsError(f"Output directory already contains files: {output}. Use a new directory or --resume.")
    output.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(output / "train.log")
    logger.info("Experiment: %s", output)
    if config["max_train_batches"] or config["max_val_batches"]:
        logger.warning("SMOKE RUN: batch limits are active. Scores are diagnostic, not full validation results.")
    logger.info("Input NIR,R,G; batch=%s accumulation=%s nominal effective batch=%s; AMP=%s",
                config["batch_size"], config["accumulation_steps"], config["batch_size"] * config["accumulation_steps"], config["amp"])
    if checkpoint and config["num_workers"] != checkpoint["config"]["num_workers"]:
        logger.warning("Worker count changed; augmentation RNG sequences can change.")
    set_seed(config["seed"])
    torch.backends.cudnn.benchmark = False
    root = Path(config["data_root"])
    logger.info("Hashing train/val files; test data are not used for training or model selection.")
    manifest = build_manifest(root, ("train", "val"))
    if checkpoint:
        if checkpoint["dataset_fingerprint"] != manifest["fingerprint"]:
            raise ValueError("Train/validation data changed; resume refused.")
        channels, mean, std = preprocessing_from_checkpoint(checkpoint)
    else:
        channels = tuple(config["input_channels"])
        mean, std = compute_train_stats(root, channels)
    train_set = PairedTiffDataset(root, "train", mean, std, channels, augment=True)
    val_set = PairedTiffDataset(root, "val", mean, std, channels)
    generators = {name: torch.Generator().manual_seed(config["seed"] + index) for index, name in enumerate(("train", "val"))}
    common = dict(batch_size=config["batch_size"], num_workers=config["num_workers"], pin_memory=device.type == "cuda",
                  worker_init_fn=seed_worker, persistent_workers=False)
    train_loader = DataLoader(train_set, shuffle=True, generator=generators["train"], **common)
    val_loader = DataLoader(val_set, shuffle=False, generator=generators["val"], **common)
    # Keep --help/data utilities usable without Mamba. Full models import only here.
    from rs3mamba_baseline.model import build_model, model_from_checkpoint
    model = (model_from_checkpoint(checkpoint) if checkpoint else build_model(config)).to(device)
    optimizer = make_optimizer(model, config)
    scheduler = make_scheduler(optimizer, config)
    scaler = torch.cuda.amp.GradScaler(enabled=config["amp"])
    history = copy.deepcopy(checkpoint["history"]) if checkpoint else []
    best_iou = checkpoint["best_iou"] if checkpoint else -1.0
    best_epoch = checkpoint["best_epoch"] if checkpoint else 0
    start_epoch = checkpoint["epoch"] + 1 if checkpoint else 1
    if checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        # state tensors follow the destination parameter device in PyTorch load_state_dict.
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        restore_rng(checkpoint["rng_state"], generators)
    environment = collect_environment(Path(__file__).resolve().parent, device)
    initialization = checkpoint.get("initialization_report", checkpoint.get("initialization", {})) if checkpoint else getattr(model, "initialization_report", {})
    model.initialization_report = initialization
    save_config(output / "config.yaml", config)
    write_json(output / "preprocessing.json", dict(input_channels=list(channels), input_bands=config["input_bands"], mean=mean.tolist(), std=std.tolist()))
    write_json(output / "initialization.json", initialization)
    write_json(output / "dataset_manifest.json", manifest)
    write_json(output / "environment.json", environment)
    write_json(output / "model_summary.json", dict(architecture="RS3Mamba", num_classes=2,
               total_parameters=sum(p.numel() for p in model.parameters()), trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad)))
    save_pip_freeze(output / "requirements-freeze.txt")
    criterion = nn.CrossEntropyLoss()
    started = time.monotonic()
    for epoch in range(start_epoch, config["epochs"] + 1):
        encoder_lr = optimizer.param_groups[0]["lr"]
        decoder_lr = optimizer.param_groups[-1]["lr"]
        training = train_epoch(model, train_loader, optimizer, criterion, device, epoch, config["epochs"],
                               config["log_every"], config["accumulation_steps"], scaler, config["amp"], config["max_train_batches"])
        validation = evaluate_model(model, val_loader, device, criterion, "Validation", config["log_every"],
                                    collect_per_image=False, sample_limit=config["visualization_samples"], max_batches=config["max_val_batches"])
        row = dict(epoch=epoch, train_loss=training["loss"], val_loss=validation["loss"],
                   encoder_lr=encoder_lr, decoder_lr=decoder_lr, optimizer_steps=training["optimizer_steps"],
                   train_images=training["num_images"], val_images=validation["num_images"],
                   **{"val_" + key: value for key, value in validation["metrics"].items()})
        history.append(row)
        score = validation["metrics"]["foreground_iou"]
        if score is None:
            if config["max_val_batches"]:
                logger.warning("The smoke validation subset has undefined foreground IoU; best.pt is not selected.")
            else:
                raise ValueError("Validation foreground IoU is undefined. Check that validation labels contain foreground.")
        improved = score is not None and score > best_iou
        if improved:
            best_iou, best_epoch = score, epoch
        scheduler.step()
        payload = make_checkpoint(model, optimizer, scheduler, scaler, epoch, best_iou, best_epoch,
                                  config, mean, std, history, manifest, generators, environment)
        atomic_torch_save(payload, output / "checkpoints" / "last.pt")
        if improved:
            atomic_torch_save(payload, output / "checkpoints" / "best.pt")
            for sample in validation["samples"]:
                save_prediction_visuals(Path(sample["image_path"]), sample["prediction"], output / "best_val_visualizations", target=sample["target"])
        write_history(output, history)
        plot_history(history, output / "curves")
        logger.info("Epoch %s | train %.6f val %.6f | foreground IoU %s | best %s @%s",
                    epoch, training["loss"], validation["loss"], score, best_iou, best_epoch)
        write_json(output / "summary.json", dict(completed=epoch == config["epochs"], completed_epochs=epoch,
                   total_epochs=config["epochs"], best_epoch=best_epoch, best_foreground_iou=best_iou if best_epoch else None,
                   run_kind="smoke" if config["max_train_batches"] or config["max_val_batches"] else "full",
                   updated_at=utc_now(), session_seconds=time.monotonic() - started))
    if best_epoch:
        logger.info("Finished. Best validation checkpoint: %s", output / "checkpoints" / "best.pt")
    else:
        logger.warning("Finished without a defined foreground IoU; only last.pt is available.")


if __name__ == "__main__":
    main()
