"""Download and verify the two upstream initialization files (never committed)."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import urllib.request
from pathlib import Path


WEIGHTS = {
    "cnn": {
        "filename": "semi_weakly_supervised_resnet18-118f1556.pth",
        "url": "https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_weakly_supervised_resnet18-118f1556.pth",
        "sha256": "118f1556c5734939bce4171a5223f613f7ff344080967bb054db2d41613bbb3d",
        "expected_bytes": 46811375,
        "source": "Meta SWSL ResNet-18, CC-BY-NC 4.0",
    },
    "vmamba": {
        "filename": "vmamba_tiny_e292.pth",
        "url": "https://github.com/MzeroMiko/VMamba/releases/download/%23v0cls/vssmtiny_dp01_ckpt_epoch_292.pth",
        "expected_bytes": 91649482,
        "sha256": "dbc0cc4f5ec0e45db5fba7c939d2c7d9b617e891ac10766912a8d604c37c5e47",
        "source": "VMamba official v0 classification release, Tiny epoch 292, ImageNet-1K",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(path: Path, kind: str) -> dict:
    import torch

    info = WEIGHTS[kind]
    size = path.stat().st_size
    if size < 1024 * 1024 or (info.get("expected_bytes") is not None and size != info["expected_bytes"]):
        raise ValueError(f"Unexpected {kind} checkpoint size: {size} bytes")
    digest = sha256(path)
    if digest != info["sha256"]:
        raise ValueError(f"SHA-256 mismatch for {kind}: {digest}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid {kind} checkpoint payload")
    if kind == "vmamba":
        state = payload.get("model", {})
        probes = {"layers.0.blocks.0.self_attention.in_proj.weight": (384, 96),
                  "layers.2.blocks.8.self_attention.in_proj.weight": (1536, 384),
                  "layers.3.blocks.1.self_attention.in_proj.weight": (3072, 768)}
    else:
        state = payload.get("state_dict", payload.get("model", payload))
        if state and all(key.startswith("module.") for key in state):
            state = {key[7:]: value for key, value in state.items()}
        probes = {"conv1.weight": (64, 3, 7, 7), "layer4.1.conv2.weight": (512, 512, 3, 3)}
    for key, shape in probes.items():
        if key not in state or not torch.is_tensor(state[key]) or tuple(state[key].shape) != shape:
            raise ValueError(f"Unexpected architecture in {kind}: {key} should have shape {shape}")
    return {"file": str(path.resolve()), "bytes": size, "sha256": digest, "kind": kind,
            "source": info["source"], "url": info["url"], "state_tensors": len(state)}


def download(path: Path, kind: str, force: bool) -> dict:
    if path.exists() and not force:
        report = verify(path, kind)
        print(f"Verified existing {path.name}", flush=True)
        return report
    partial = path.with_name(path.name + ".part")
    for attempt in range(3):
        try:
            request = urllib.request.Request(WEIGHTS[kind]["url"], headers={"User-Agent": "RS3mamba-baseline/1.0"})
            with urllib.request.urlopen(request, timeout=90) as response, partial.open("wb") as handle:
                content_type = response.headers.get("Content-Type", "")
                if "text/html" in content_type:
                    raise ValueError("The weight server returned HTML instead of a checkpoint")
                for chunk in iter(lambda: response.read(1024 * 1024), b""):
                    handle.write(chunk)
            report = verify(partial, kind)
            os.replace(partial, path)
            report["file"] = str(path.resolve())
            print(f"Downloaded and verified {path.name} ({report['bytes']:,} bytes)", flush=True)
            return report
        except Exception:
            if partial.exists():
                partial.unlink()
            if attempt == 2:
                raise
            time.sleep(attempt + 1)
    raise RuntimeError("Download did not complete")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("pretrain"))
    parser.add_argument("--only", choices=("both", "cnn", "vmamba"), default="both")
    parser.add_argument("--force", action="store_true", help="Atomically replace existing files after verification")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    for kind in (("cnn", "vmamba") if args.only == "both" else (args.only,)):
        path = args.output_dir / WEIGHTS[kind]["filename"]
        reports.append(verify(path, kind) if args.verify_only else download(path, kind, args.force))
    manifest = args.output_dir / "weights_manifest.json"
    if manifest.exists():
        previous = json.loads(manifest.read_text(encoding="utf-8"))
        by_kind = {entry["kind"]: entry for entry in previous if isinstance(entry, dict) and "kind" in entry}
    else:
        by_kind = {}
    by_kind.update({entry["kind"]: entry for entry in reports})
    manifest.write_text(json.dumps(list(by_kind.values()), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(reports, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
