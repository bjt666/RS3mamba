"""Full epoch-boundary checkpoints, including loader and AMP state."""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch

SCHEMA_VERSION = 1
MODEL_NAME = "rs3mamba-sargassum-v1"


def atomic_torch_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_checkpoint(path: str | Path) -> dict:
    # Full optimizer/RNG state uses pickle; use this only with trusted project checkpoints.
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = {"model", "config", "normalization_mean", "normalization_std", "epoch"}
    if not isinstance(payload, dict) or payload.get("model_name") != MODEL_NAME or not required.issubset(payload):
        raise ValueError(f"Not an RS3Mamba baseline checkpoint: {path}")
    preprocessing_from_checkpoint(payload)
    return payload


def preprocessing_from_checkpoint(checkpoint: dict):
    channels = tuple(checkpoint["config"]["input_channels"])
    mean = np.asarray(checkpoint["normalization_mean"], dtype=np.float32)
    std = np.asarray(checkpoint["normalization_std"], dtype=np.float32)
    if channels != (3, 2, 1) or mean.shape != (3,) or std.shape != (3,):
        raise ValueError("Invalid checkpoint input order or normalization shape.")
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
        raise ValueError("Invalid checkpoint normalization statistics.")
    return channels, mean, std


def capture_rng(generators: dict) -> dict:
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                loaders={name: generator.get_state() for name, generator in generators.items()})


def restore_rng(state: dict, generators: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state["cuda"]:
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("CUDA device count changed; cannot restore saved CUDA RNG states.")
        torch.cuda.set_rng_state_all(state["cuda"])
    for name, generator in generators.items():
        generator.set_state(state["loaders"][name])


def make_checkpoint(model, optimizer, scheduler, scaler, epoch, best_iou, best_epoch,
                    config, mean, std, history, manifest, generators, environment) -> dict:
    return dict(schema_version=SCHEMA_VERSION, model_name=MODEL_NAME, model=model.state_dict(),
                config=config, optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                scaler=scaler.state_dict(), epoch=epoch, best_iou=best_iou, best_epoch=best_epoch,
                normalization_mean=mean.tolist(), normalization_std=std.tolist(), history=history,
                dataset_fingerprint=manifest["fingerprint"], rng_state=capture_rng(generators),
                initialization_report=getattr(model, "initialization_report", {}), environment=environment)
