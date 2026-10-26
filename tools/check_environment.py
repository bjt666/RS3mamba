"""Report the Linux training environment and test the real selective-scan CUDA op."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-only", action="store_true", help="Collect information without requiring CUDA")
    parser.add_argument("--model-smoke", action="store_true", help="Also check two-class 256x256 model forward/backward")
    parser.add_argument("--no-pretrained", action="store_true", help="Use module initialization for the optional model check")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    report = {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
              "packages": {}, "imports": {}, "checks": {}}
    for package in ("torch", "torchvision", "numpy", "timm", "einops", "mamba-ssm", "causal-conv1d", "triton", "transformers"):
        try:
            report["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            report["packages"][package] = None
    for module in ("torch", "torchvision", "timm", "mamba_ssm", "causal_conv1d", "selective_scan_cuda"):
        try:
            imported = importlib.import_module(module)
            report["imports"][module] = {"ok": True, "path": getattr(imported, "__file__", None)}
        except Exception as error:
            report["imports"][module] = {"ok": False, "error": str(error)}
    for executable, arguments in (("nvidia-smi", ["--query-gpu=name,driver_version,memory.total", "--format=csv"]),
                                   ("nvcc", ["--version"]), ("g++", ["--version"])):
        if shutil.which(executable):
            completed = subprocess.run([executable, *arguments], capture_output=True, text=True, timeout=30)
            report[executable] = {"returncode": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr}
        else:
            report[executable] = None
    failed = False
    if not args.report_only:
        try:
            import torch
            from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable; a Linux NVIDIA training server is required")
            report["cuda"] = {"torch_cuda": torch.version.cuda, "name": torch.cuda.get_device_name(0),
                              "memory_gib": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2)}
            u = torch.randn(2, 8, 32, device="cuda", requires_grad=True)
            delta = torch.randn_like(u, requires_grad=True)
            A = -torch.ones(8, 4, device="cuda")
            B = torch.randn(2, 4, 32, device="cuda", requires_grad=True)
            C = torch.randn_like(B, requires_grad=True)
            result = selective_scan_fn(u, delta, A, B, C, delta_softplus=True)
            result.square().mean().backward()
            torch.cuda.synchronize()
            if not torch.isfinite(result).all() or u.grad is None or not torch.isfinite(u.grad).all():
                raise RuntimeError("Selective scan produced nonfinite values or no finite gradient")
            report["checks"]["selective_scan_forward_backward"] = "passed"
            if args.model_smoke:
                from rs3mamba_baseline.model import build_model, model_parameter_counts

                model = build_model({"pretrained": not args.no_pretrained, "input_channels": [3, 2, 1]}).cuda().train()
                x = torch.randn(args.batch_size, 3, 256, 256, device="cuda")
                target = torch.randint(0, 2, (args.batch_size, 256, 256), device="cuda")
                logits = model(x)
                if tuple(logits.shape) != (args.batch_size, 2, 256, 256):
                    raise RuntimeError(f"Unexpected logits shape: {tuple(logits.shape)}")
                loss = torch.nn.functional.cross_entropy(logits, target)
                loss.backward()
                torch.cuda.synchronize()
                if not torch.isfinite(loss) or any(parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                                                      for parameter in model.parameters()):
                    raise RuntimeError("Model forward/backward produced nonfinite values")
                report["checks"]["model_forward_backward"] = {"status": "passed", "loss": loss.item(),
                    "logits_shape": list(logits.shape), "peak_cuda_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
                    "parameters": model_parameter_counts(model), "initialization": model.initialization_report}
        except Exception as error:
            report["checks"]["error"] = str(error)
            failed = True
        if any(not value["ok"] for value in report["imports"].values()):
            failed = True
    report["status"] = "report_only" if args.report_only else ("failed" if failed else "passed")
    output = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
    print(output)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
