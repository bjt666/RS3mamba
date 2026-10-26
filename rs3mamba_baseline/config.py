"""Explicit, serializable training settings for the fixed RS3Mamba baseline."""
from __future__ import annotations

import copy
import math
from pathlib import Path

import yaml

DEFAULTS = dict(
    data_root="dataset", output_dir=None, epochs=100, batch_size=4,
    accumulation_steps=2, num_workers=4, seed=42, device="cuda", amp=False,
    pretrained=True, cnn_checkpoint="pretrain/semi_weakly_supervised_resnet18-118f1556.pth",
    vmamba_checkpoint="pretrain/vmamba_tiny_e292.pth", input_channels=[3, 2, 1],
    input_bands=["NIR", "R", "G"], num_classes=2, optimizer="adamw", scheduler="cosine",
    encoder_lr=6e-5, decoder_lr=6e-4, learning_rate=0.01, momentum=0.9,
    weight_decay=0.01, milestones=[25, 35, 45], gamma=0.1, loss="cross_entropy",
    log_every=20, visualization_samples=8, max_train_batches=None, max_val_batches=None,
)


def default_config() -> dict:
    return copy.deepcopy(DEFAULTS)


def read_config(path: Path | None) -> dict:
    if path is None:
        return {}
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ValueError("Configuration must be a YAML mapping.")
    unknown = set(value) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
    return value


def save_config(path: Path, config: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")


def validate_config(config: dict) -> None:
    unknown = set(config) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
    for key in ("epochs", "batch_size", "accumulation_steps", "log_every"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer.")
    for key in ("num_workers", "seed", "visualization_samples"):
        if type(config[key]) is not int or config[key] < 0:
            raise ValueError(f"{key} must be a nonnegative integer.")
    for key in ("max_train_batches", "max_val_batches"):
        if config[key] is not None and (type(config[key]) is not int or config[key] < 1):
            raise ValueError(f"{key} must be null or a positive integer.")
    for key in ("encoder_lr", "decoder_lr", "learning_rate", "gamma"):
        if not isinstance(config[key], (int, float)) or not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"{key} must be finite and positive.")
    if not math.isfinite(config["weight_decay"]) or config["weight_decay"] < 0:
        raise ValueError("weight_decay must be finite and nonnegative.")
    if not 0 <= config["momentum"] < 1:
        raise ValueError("momentum must be in [0, 1).")
    if config["optimizer"] not in {"adamw", "sgd"} or config["scheduler"] not in {"cosine", "multistep"}:
        raise ValueError("optimizer must be adamw/sgd and scheduler cosine/multistep.")
    if not isinstance(config["milestones"], list) or any(type(x) is not int or x < 1 for x in config["milestones"]):
        raise ValueError("milestones must be a list of positive epoch numbers.")
    if config["scheduler"] == "multistep" and (config["milestones"] != sorted(set(config["milestones"])) or any(x >= config["epochs"] for x in config["milestones"])):
        raise ValueError("milestones must be increasing and precede the final epoch.")
    if config["input_channels"] != [3, 2, 1] or config["input_bands"] != ["NIR", "R", "G"]:
        raise ValueError("This baseline fixes the source order B,G,R,NIR and model input NIR,R,G.")
    if config["num_classes"] != 2 or config["loss"] != "cross_entropy":
        raise ValueError("The baseline uses two classes with unweighted cross entropy.")
    if any(type(config[key]) is not bool for key in ("amp", "pretrained")):
        raise ValueError("amp and pretrained must be booleans.")
    if not isinstance(config["data_root"], str) or not config["data_root"]:
        raise ValueError("data_root must be a nonempty path.")
    if config["output_dir"] is not None and not isinstance(config["output_dir"], str):
        raise ValueError("output_dir must be a path or null.")
