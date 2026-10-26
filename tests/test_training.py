"""CPU checks for accumulation, full checkpoints and immutable resume settings."""
from __future__ import annotations

import argparse
import contextlib
import copy
import logging
import random
import tempfile
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from rs3mamba_baseline.checkpoint import atomic_torch_save, capture_rng, load_checkpoint, make_checkpoint, restore_rng
from rs3mamba_baseline.config import default_config, validate_config
from rs3mamba_baseline.engine import evaluate_model, train_epoch


class Samples:
    def __init__(self):
        generator = torch.Generator().manual_seed(15)
        self.images = torch.randn(5, 3, 4, 4, generator=generator)
        self.masks = torch.randint(0, 2, (5, 4, 4), generator=generator)

    def __len__(self):
        return 5

    def __getitem__(self, index):
        return dict(image=self.images[index], mask=self.masks[index], name=str(index), image_path=str(index))


@contextlib.contextmanager
def closed_test_logger():
    try:
        yield
    finally:
        logger = logging.getLogger("rs3mamba")
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            handler.close()


class TrainingTests(unittest.TestCase):
    def test_accumulation_matches_actual_sample_groups(self):
        torch.manual_seed(42)
        model = nn.Conv2d(3, 2, 1)
        reference = copy.deepcopy(model)
        dataset = Samples()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        result = train_epoch(model, DataLoader(dataset, batch_size=2), optimizer, nn.CrossEntropyLoss(),
                             torch.device("cpu"), 1, 1, accumulation_steps=2)
        expected_optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
        for begin, end in ((0, 4), (4, 5)):
            expected_optimizer.zero_grad()
            nn.functional.cross_entropy(reference(dataset.images[begin:end]), dataset.masks[begin:end]).backward()
            expected_optimizer.step()
        self.assertEqual(result["optimizer_steps"], 2)
        self.assertEqual(result["num_images"], 5)
        for actual, expected in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)

    def test_smoke_limit_flushes_partial_group(self):
        model = nn.Conv2d(3, 2, 1)
        result = train_epoch(model, DataLoader(Samples(), batch_size=2),
                             torch.optim.SGD(model.parameters(), lr=0.1), nn.CrossEntropyLoss(),
                             torch.device("cpu"), 1, 1, accumulation_steps=8, max_batches=1)
        self.assertEqual(result["optimizer_steps"], 1)
        self.assertEqual(result["num_images"], 2)

    def test_evaluation_aggregates_pixels_and_emits_every_mask(self):
        samples = []
        result = evaluate_model(nn.Conv2d(3, 2, 1), DataLoader(Samples(), batch_size=2), torch.device("cpu"),
                                nn.CrossEntropyLoss(), sample_limit=2, prediction_callback=samples.append)
        self.assertEqual(len(samples), 5)
        self.assertEqual(len(result["samples"]), 2)
        self.assertEqual(int(result["confusion"].sum()), 5 * 4 * 4)
        self.assertEqual(result["num_images"], 5)

    def test_checkpoint_and_rng_round_trip(self):
        model = nn.Conv2d(3, 2, 1)
        model.initialization_report = dict(pretrained=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 2)
        scaler = torch.cuda.amp.GradScaler(enabled=False)
        generators = dict(train=torch.Generator().manual_seed(43))
        checkpoint = make_checkpoint(model, optimizer, scheduler, scaler, 1, 0.3, 1, default_config(),
                                     np.zeros(3), np.ones(3), [dict(epoch=1)], dict(fingerprint="abc"), generators, {})
        expected = (random.random(), float(np.random.rand()), torch.rand(2), torch.rand(2, generator=generators["train"]))
        restore_rng(checkpoint["rng_state"], generators)
        actual = (random.random(), float(np.random.rand()), torch.rand(2), torch.rand(2, generator=generators["train"]))
        self.assertEqual(actual[:2], expected[:2])
        torch.testing.assert_close(actual[2], expected[2])
        torch.testing.assert_close(actual[3], expected[3])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last.pt"
            atomic_torch_save(checkpoint, path)
            restored = load_checkpoint(path)
            self.assertEqual(restored["dataset_fingerprint"], "abc")
            self.assertEqual(restored["initialization_report"], {"pretrained": False})
            self.assertFalse(path.with_suffix(".pt.tmp").exists())

    def test_resume_refuses_new_schedule_and_smoke_to_formal(self):
        from train import prepare_config
        checkpoint = dict(schema_version=1, config=default_config(), optimizer={}, scheduler={}, scaler={},
                          rng_state={}, history=[], dataset_fingerprint="abc", best_iou=0.2, best_epoch=1)
        args = argparse.Namespace(config=None, resume=Path("run/checkpoints/last.pt"), epochs=200)
        with self.assertRaisesRegex(ValueError, "Cannot change epochs"):
            prepare_config(args, checkpoint)
        checkpoint["config"]["max_train_batches"] = 2
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text("max_train_batches: null\n", encoding="utf-8")
            args = argparse.Namespace(config=path, resume=Path("run/checkpoints/last.pt"))
            with self.assertRaisesRegex(ValueError, "Cannot change max_train_batches"):
                prepare_config(args, checkpoint)

    def test_cli_train_resume_evaluate_predict_with_dummy_model(self):
        """Exercise serialization/data/CLI glue without substituting a fake production scan."""
        import tifffile
        import train
        import evaluate
        import predict
        from PIL import Image

        class Tiny(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = nn.Conv2d(3, 3, 1)
                self.vssm_encoder = nn.Conv2d(3, 3, 1)
                self.stem = nn.Identity()
                self.decoder = nn.Conv2d(3, 2, 1)
                self.initialization_report = dict(pretrained=False, test_dummy=True)

            def forward(self, images):
                return self.decoder(self.vssm_encoder(self.backbone(images)))

        model_module = types.ModuleType("rs3mamba_baseline.model")
        model_module.build_model = lambda config: Tiny()

        def restore_model(checkpoint):
            model = Tiny()
            model.load_state_dict(checkpoint["model"])
            return model

        model_module.model_from_checkpoint = restore_model
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with tempfile.TemporaryDirectory() as directory, closed_test_logger(), patch.dict(sys.modules, {"rs3mamba_baseline.model": model_module}):
                root = Path(directory)
                dataset = root / "dataset"
                rng = np.random.default_rng(42)
                for split, count in (("train", 4), ("val", 2), ("test", 2)):
                    (dataset / "images" / split).mkdir(parents=True)
                    (dataset / "masks" / split).mkdir(parents=True)
                    for index in range(count):
                        image = rng.normal(size=(256, 256, 4)).astype(np.float32)
                        mask = np.zeros((256, 256), dtype=np.uint8)
                        mask[32:64, 32:64] = 255
                        tifffile.imwrite(dataset / "images" / split / f"{index}.tif", image, photometric="rgb")
                        tifffile.imwrite(dataset / "masks" / split / f"{index}.tif", mask)
                common = ["--data-root", str(dataset), "--epochs", "2", "--batch-size", "2",
                          "--accumulation-steps", "2", "--num-workers", "0", "--device", "cpu",
                          "--no-pretrained", "--visualization-samples", "0"]
                no_pip = patch.object(train, "save_pip_freeze", lambda path: None)
                no_plot = patch.object(train, "plot_history", lambda *args: None)
                with no_pip, no_plot:
                    full = root / "full"
                    with patch.object(sys, "argv", ["train.py", *common, "--output-dir", str(full)]):
                        train.main()
                    resumed = root / "resumed"
                    original_train = train.train_epoch

                    def interrupt_epoch_two(*args, **kwargs):
                        if args[5] == 2:
                            raise RuntimeError("test interruption")
                        return original_train(*args, **kwargs)

                    with patch.object(sys, "argv", ["train.py", *common, "--output-dir", str(resumed)]), patch.object(train, "train_epoch", interrupt_epoch_two):
                        with self.assertRaisesRegex(RuntimeError, "test interruption"):
                            train.main()
                    with patch.object(sys, "argv", ["train.py", "--resume", str(resumed / "checkpoints/last.pt")]):
                        train.main()
                    expected = load_checkpoint(full / "checkpoints/last.pt")
                    actual = load_checkpoint(resumed / "checkpoints/last.pt")
                    self.assertEqual(len(actual["history"]), 2)
                    for key in expected["model"]:
                        torch.testing.assert_close(actual["model"][key], expected["model"][key], rtol=0, atol=0)
                    copied_checkpoint = root / "copied.pt"
                    copied_checkpoint.write_bytes((full / "checkpoints/last.pt").read_bytes())
                    with patch.object(sys, "argv", ["train.py", "--resume", str(copied_checkpoint), "--output-dir", str(full)]):
                        with self.assertRaisesRegex(ValueError, "Resume requires"):
                            train.main()
                evaluation = root / "evaluation"
                with patch.object(sys, "argv", ["evaluate.py", "--checkpoint", str(full / "checkpoints/best.pt"),
                                 "--data-root", str(dataset), "--split", "test", "--output-dir", str(evaluation),
                                 "--num-workers", "0", "--device", "cpu", "--visualization-samples", "0"]):
                    evaluate.main()
                self.assertTrue((evaluation / "metrics.json").is_file())
                with Image.open(evaluation / "masks/0.tif.png") as mask_file:
                    self.assertTrue(set(np.unique(np.asarray(mask_file))) <= {0, 1})
                prediction = root / "prediction"
                with patch.object(sys, "argv", ["predict.py", "--checkpoint", str(full / "checkpoints/best.pt"),
                                 "--input", str(dataset / "images/test/0.tif"), "--output-dir", str(prediction),
                                 "--device", "cpu", "--save-probabilities"]):
                    predict.main()
                self.assertTrue((prediction / "comparisons/0.tif.png").is_file())
                self.assertEqual(tifffile.imread(prediction / "probabilities/0.tif").dtype, np.float32)
        finally:
            torch.set_num_threads(previous_threads)


if __name__ == "__main__":
    unittest.main()
