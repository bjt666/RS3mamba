"""Official RS3Mamba architecture, explicit initialization and offline restoration."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import torch


UPSTREAM_REVISION = "b89998178ab07679c179fbda3b28eefcd0e361eb"
CNN_NAME = "swsl_resnet18"
CNN_FILENAME = "semi_weakly_supervised_resnet18-118f1556.pth"
VMAMBA_FILENAME = "vmamba_tiny_e292.pth"

# Preserve the actual upstream skip-list behavior. It repeats norm.weight and
# therefore still copies patch_embed.norm.bias; this bias is zero in the release.
VMAMBA_SKIPPED = frozenset({
    "norm.weight", "norm.bias", "head.weight", "head.bias",
    "patch_embed.proj.weight", "patch_embed.proj.bias", "patch_embed.norm.weight",
})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: str | Path) -> tuple[dict, dict]:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing pretrained checkpoint: {path}. Run python tools/download_pretrained.py first."
        )
    # Only trusted upstream initialization files should be passed here. Both
    # official files predate the current weights_only serialization defaults.
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint must be a mapping: {path}")
    return payload, {"path": str(path), "sha256": _sha256(path), "bytes": path.stat().st_size}


def _tensor_mapping(payload: dict) -> dict:
    for wrapper in ("state_dict", "model"):
        if wrapper in payload and isinstance(payload[wrapper], dict):
            payload = payload[wrapper]
            break
    if not payload or not all(isinstance(key, str) and torch.is_tensor(value)
                              for key, value in payload.items()):
        raise ValueError("CNN checkpoint does not contain a tensor state dictionary.")
    if all(key.startswith("module.") for key in payload):
        payload = {key[7:]: value for key, value in payload.items()}
    return payload


def _channels(config: dict) -> tuple[int, ...]:
    channels = tuple(config.get("input_channels", (3, 2, 1)))
    if len(channels) != 3 or len(set(channels)) != 3 or any(channel not in range(4) for channel in channels):
        raise ValueError("RS3Mamba requires three distinct source channels from B,G,R,NIR (indices 0..3).")
    return channels


def _load_cnn(model, path: str | Path, channels: tuple[int, ...]) -> dict:
    payload, report = _load(path)
    source = _tensor_mapping(payload)
    expected = model.backbone.state_dict()
    # The official Meta release was serialized before BatchNorm counters were
    # added to state dictionaries. PyTorch's own historical loader initializes
    # these counters to zero as well; all learned tensors and running stats must
    # still be present.
    historical_buffers = {name for name in expected if name.endswith(".num_batches_tracked") and name not in source}
    missing = sorted(set(expected) - set(source) - historical_buffers)
    extra = sorted(set(source) - set(expected) - {"fc.weight", "fc.bias"})
    if missing or extra:
        raise ValueError(f"Wrong/incomplete SWSL ResNet-18 checkpoint: missing={missing[:8]}, unexpected={extra[:8]}")
    if "conv1.weight" not in source or tuple(source["conv1.weight"].shape) != (64, 3, 7, 7):
        raise ValueError("Expected the three-channel SWSL ResNet-18 input convolution.")
    rgb_filters = source["conv1.weight"]
    # Source TIFF is B,G,R,NIR; pretrained convolution is RGB.
    filters = {0: rgb_filters[:, 2], 1: rgb_filters[:, 1],
               2: rgb_filters[:, 0], 3: rgb_filters.mean(dim=1)}
    adapted = torch.stack([filters[channel] for channel in channels], dim=1)
    loaded = {}
    for name, target in expected.items():
        if name in historical_buffers:
            continue
        value = adapted if name == "conv1.weight" else source[name]
        if value.shape != target.shape:
            raise ValueError(f"CNN shape mismatch for {name}: {tuple(value.shape)} vs {tuple(target.shape)}")
        loaded[name] = value
    state = dict(expected)
    state.update(loaded)
    model.backbone.load_state_dict(state, strict=True)
    report.update({"model": CNN_NAME, "loaded_tensors": len(loaded), "expected_tensors": len(expected) - len(historical_buffers),
                   "loaded_numel": sum(value.numel() for value in loaded.values()),
                   "coverage": 1.0, "ignored_source_keys": sorted(set(source) - set(expected)),
                   "initialized_batchnorm_counters": sorted(historical_buffers),
                   "input_convolution": {"source": "RGB", "target_source_indices": list(channels),
                                         "NIR_initialization": "mean of pretrained RGB filters"}})
    return report


def _load_vmamba(model, path: str | Path) -> dict:
    payload, report = _load(path)
    if "model" not in payload or not isinstance(payload["model"], dict):
        raise ValueError("VMamba v0 Tiny e292 checkpoint must contain a 'model' state dictionary.")
    source = payload["model"]
    expected = model.vssm_encoder.state_dict()
    intentionally_new = {name for name in expected if name in VMAMBA_SKIPPED}
    required = set(expected) - intentionally_new
    loaded, skipped, unexpected = {}, [], []
    for name, value in source.items():
        if name in VMAMBA_SKIPPED:
            skipped.append(name)
            continue
        target_name = re.sub(r"^layers\.(\d+)\.downsample\.", r"downsamples.\1.", name)
        if target_name not in expected:
            unexpected.append(name)
            continue
        if not torch.is_tensor(value) or value.shape != expected[target_name].shape:
            shape = tuple(value.shape) if torch.is_tensor(value) else type(value).__name__
            raise ValueError(f"VMamba shape mismatch for {name}: {shape} vs {tuple(expected[target_name].shape)}")
        if target_name in loaded:
            raise ValueError(f"Duplicate VMamba mapping for {target_name}")
        loaded[target_name] = value
    missing = sorted(required - set(loaded))
    if missing or unexpected:
        raise ValueError(
            f"Wrong/incomplete VMamba v0 Tiny e292 checkpoint: missing={missing[:8]}, "
            f"unexpected={unexpected[:8]}. Newer VMamba v2/Tiny checkpoints are incompatible."
        )
    if not loaded:
        raise ValueError("No compatible VMamba tensors were loaded.")
    expected.update(loaded)
    model.vssm_encoder.load_state_dict(expected, strict=True)
    report.update({"model": "VMamba-v0-Tiny-e292", "loaded_tensors": len(loaded),
                   "expected_tensors": len(required), "coverage": len(loaded) / len(required),
                   "loaded_numel": sum(value.numel() for value in loaded.values()),
                   "skipped_source_keys": sorted(skipped),
                   "initialized_encoder_keys": sorted(intentionally_new)})
    return report


def build_model(config: dict):
    """Create two-class official model, loading external weights only if requested.

    All checkpoint paths are local. The upstream architecture is always
    constructed with timm's automatic weight download disabled.
    """
    from vendor.rs3mamba.RS3Mamba import RS3Mamba

    channels = _channels(config)
    num_classes = int(config.get("num_classes", 2))
    if num_classes != 2:
        raise ValueError("This Sargassum baseline uses exactly two classes.")
    model = RS3Mamba(backbone_name=CNN_NAME, pretrained=False, num_classes=num_classes)
    report = {"architecture": "official-RS3Mamba-R18-VMamba-v0-Tiny", "upstream_revision": UPSTREAM_REVISION,
              "pretrained": bool(config.get("pretrained", True)), "input_channels": list(channels),
              "two_branches_receive_same_image": True,
              "unpretrained_parts": ["stem", "Fuse", "decoder", "vssm_encoder.patch_embed.proj",
                                     "vssm_encoder.patch_embed.norm.weight"]}
    if report["pretrained"]:
        report["cnn"] = _load_cnn(model, config.get("cnn_checkpoint", f"pretrain/{CNN_FILENAME}"), channels)
        report["vmamba"] = _load_vmamba(model, config.get("vmamba_checkpoint", f"pretrain/{VMAMBA_FILENAME}"))
    else:
        report["initialization"] = "upstream module initializers, no external checkpoint"
    model.initialization_report = report
    return model


def model_from_checkpoint(checkpoint: dict):
    """Restore a complete trained model with no initialization files or network."""
    config = dict(checkpoint["config"])
    config["pretrained"] = False
    model = build_model(config)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.initialization_report = checkpoint.get("initialization_report", checkpoint.get("initialization", {}))
    return model


def model_parameter_counts(model) -> dict:
    """Unique parameters; upstream aliases must not be counted more than once."""
    return {"total": sum(parameter.numel() for parameter in model.parameters()),
            "trainable": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)}
