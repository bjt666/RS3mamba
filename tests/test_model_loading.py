"""Initialization invariants; no CUDA substitute is used in production code."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from rs3mamba_baseline.model import _load_cnn, _load_vmamba, build_model, model_from_checkpoint


ROOT = Path(__file__).resolve().parents[1]


class SmallEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Module()
        self.patch_embed.proj = nn.Conv2d(48, 96, 2)
        self.patch_embed.norm = nn.LayerNorm(96)
        self.layers = nn.ModuleList([nn.Linear(3, 5, bias=False)])
        self.downsamples = nn.ModuleList([nn.Linear(4, 8, bias=False)])


def make_cnn():
    module = nn.Module()
    module.conv1 = nn.Conv2d(3, 64, 7, bias=False)
    module.bn1 = nn.BatchNorm2d(64)
    return module


class LoaderTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def save(self, state, filename="weights.pth"):
        path = self.directory / filename
        torch.save(state, path)
        return path

    def test_nir_filter_mapping_and_old_batchnorm_buffers(self):
        model = SimpleNamespace(backbone=make_cnn())
        source = {key: value.clone() for key, value in model.backbone.state_dict().items()
                  if not key.endswith(".num_batches_tracked")}
        source["conv1.weight"][:, 0].fill_(1)
        source["conv1.weight"][:, 1].fill_(3)
        source["conv1.weight"][:, 2].fill_(8)
        source["fc.weight"] = torch.ones(2, 64)
        source["fc.bias"] = torch.zeros(2)
        report = _load_cnn(model, self.save(source), (3, 2, 1))
        filters = model.backbone.conv1.weight.detach()
        self.assertTrue(torch.allclose(filters[:, 0], torch.full_like(filters[:, 0], 4)))
        self.assertTrue(torch.equal(filters[:, 1], torch.ones_like(filters[:, 1])))
        self.assertTrue(torch.equal(filters[:, 2], torch.full_like(filters[:, 2], 3)))
        self.assertEqual(report["coverage"], 1.0)
        self.assertEqual(report["initialized_batchnorm_counters"], ["bn1.num_batches_tracked"])
        self.assertEqual(model.backbone.bn1.num_batches_tracked.item(), 0)

    def test_cnn_missing_real_tensor_rejected(self):
        model = SimpleNamespace(backbone=make_cnn())
        source = dict(model.backbone.state_dict())
        del source["bn1.running_mean"]
        with self.assertRaisesRegex(ValueError, "incomplete"):
            _load_cnn(model, self.save(source), (3, 2, 1))

    def vmamba_source(self, encoder):
        state = {key.replace("downsamples.0.", "layers.0.downsample."): torch.ones_like(value)
                 for key, value in encoder.state_dict().items()}
        state.update({"head.weight": torch.ones(2, 3), "head.bias": torch.zeros(2),
                      "norm.weight": torch.ones(3), "norm.bias": torch.zeros(3)})
        return state

    def test_vmamba_remapping_and_official_skip_policy(self):
        encoder = SmallEncoder()
        model = SimpleNamespace(vssm_encoder=encoder)
        before_projection = encoder.patch_embed.proj.weight.detach().clone()
        before_norm_weight = encoder.patch_embed.norm.weight.detach().clone()
        source = self.vmamba_source(encoder)
        report = _load_vmamba(model, self.save({"model": source}))
        self.assertTrue(torch.equal(encoder.patch_embed.proj.weight, before_projection))
        self.assertTrue(torch.equal(encoder.patch_embed.norm.weight, before_norm_weight))
        self.assertTrue(torch.equal(encoder.patch_embed.norm.bias, torch.ones_like(encoder.patch_embed.norm.bias)))
        self.assertTrue(torch.equal(encoder.downsamples[0].weight, torch.ones_like(encoder.downsamples[0].weight)))
        self.assertEqual(report["loaded_tensors"], report["expected_tensors"])
        self.assertEqual(report["coverage"], 1.0)

    def test_vmamba_missing_or_wrong_shape_rejected(self):
        model = SimpleNamespace(vssm_encoder=SmallEncoder())
        source = self.vmamba_source(model.vssm_encoder)
        missing = dict(source)
        del missing["layers.0.weight"]
        with self.assertRaisesRegex(ValueError, "incomplete"):
            _load_vmamba(model, self.save({"model": missing}, "missing.pth"))
        source["layers.0.weight"] = torch.zeros(2, 2)
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            _load_vmamba(model, self.save({"model": source}, "shape.pth"))

    def test_restore_never_uses_initialization_paths(self):
        toy = nn.Linear(3, 2)
        saved = {key: value.clone() for key, value in toy.state_dict().items()}
        checkpoint = {"config": {"pretrained": True, "cnn_checkpoint": "missing-cnn", "vmamba_checkpoint": "missing-vmamba"},
                      "model": saved, "initialization_report": {"source": "original-run"}}
        with patch("rs3mamba_baseline.model.build_model", return_value=toy) as constructor:
            restored = model_from_checkpoint(checkpoint)
        self.assertFalse(constructor.call_args.args[0]["pretrained"])
        self.assertTrue(checkpoint["config"]["pretrained"])
        self.assertEqual(restored.initialization_report, {"source": "original-run"})


@unittest.skipUnless(importlib.util.find_spec("timm") is not None and importlib.util.find_spec("einops") is not None
                     and (ROOT / "pretrain/vmamba_tiny_e292.pth").is_file()
                     and (ROOT / "pretrain/semi_weakly_supervised_resnet18-118f1556.pth").is_file(),
                     "Optional real-weight construction test requires downloaded weights and timm/einops")
class OfficialCheckpointTests(unittest.TestCase):
    def test_real_files_cover_all_compatible_backbone_tensors(self):
        # The stub is for importing and constructing modules only. Any attempt
        # to execute selective scan raises immediately. CUDA math is verified by
        # tools/check_environment.py on the actual training server.
        # Isolate imports as well: restoring all of sys.modules in a patch.dict
        # can unload newly imported torch._dynamo modules while leaving their
        # C++ registrations alive, breaking later optimizer tests.
        script = '''
import json, sys, types
module = types.ModuleType("mamba_ssm.ops.selective_scan_interface")
def not_executable(*args, **kwargs):
    raise AssertionError("Test-only import stub cannot execute selective scan")
module.selective_scan_fn = not_executable
module.selective_scan_ref = not_executable
sys.modules["mamba_ssm"] = types.ModuleType("mamba_ssm")
sys.modules["mamba_ssm.ops"] = types.ModuleType("mamba_ssm.ops")
sys.modules[module.__name__] = module
from rs3mamba_baseline.model import build_model
model = build_model({"pretrained": True, "input_channels": [3, 2, 1]})
report = model.initialization_report
print(json.dumps({"cnn": report["cnn"]["loaded_tensors"],
                  "vmamba": report["vmamba"]["loaded_tensors"],
                  "cnn_coverage": report["cnn"]["coverage"],
                  "vmamba_coverage": report["vmamba"]["coverage"],
                  "parameters": sum(p.numel() for p in model.parameters())}))
'''
        completed = subprocess.run([sys.executable, "-c", script], cwd=ROOT, capture_output=True, text=True, timeout=120)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        report = json.loads(completed.stdout.strip().splitlines()[-1])
        self.assertEqual(report["cnn"], 100)
        self.assertEqual(report["vmamba"], 205)
        self.assertEqual(report["cnn_coverage"], 1.0)
        self.assertEqual(report["vmamba_coverage"], 1.0)
        self.assertEqual(report["parameters"], 43324638)


if __name__ == "__main__":
    unittest.main()
