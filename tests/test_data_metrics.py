"""Data protocol and pixel-aggregated metric checks using real temporary TIFFs."""

from __future__ import annotations

import importlib.util
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import tifffile

from prepare import inspect_dataset
from rs3mamba_baseline.dataset import (
    DEFAULT_CHANNELS, PairedTiffDataset, build_manifest, compute_train_stats,
    normalized_array, paired_files, read_image, read_mask,
)
from rs3mamba_baseline.metrics import confusion_matrix, metrics_from_confusion


class DatasetProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "dataset"
        for split in ("train", "val", "test"):
            for kind in ("images", "masks"):
                (self.root / kind / split).mkdir(parents=True)
        self.base = np.zeros((256, 256, 4), dtype=np.float16)
        self.base[...] = [10, 20, 30, 40]
        self.base[128:] += 2
        self.mask = np.zeros((256, 256), dtype=np.uint8)
        self.mask[20:30, 40:50] = 255
        for index, split in enumerate(("train", "val", "test")):
            self.write_pair(split, "tile.tif", self.base + index * 100, self.mask)

    def tearDown(self):
        self.temporary.cleanup()

    def write_pair(self, split, name, image, mask):
        tifffile.imwrite(self.root / "images" / split / name, image, photometric="minisblack")
        tifffile.imwrite(self.root / "masks" / split / name, mask, photometric="minisblack")

    def test_raw_float16_band_order_and_train_only_statistics(self):
        image = read_image(self.root / "images/train/tile.tif")
        self.assertEqual(image.dtype, np.float32)
        np.testing.assert_array_equal(image[0, 0], [10, 20, 30, 40])
        mean, std = compute_train_stats(self.root, DEFAULT_CHANNELS)
        np.testing.assert_array_equal(mean, [41, 31, 21])
        np.testing.assert_array_equal(std, [1, 1, 1])
        normalized = normalized_array(image, DEFAULT_CHANNELS, mean, std)
        np.testing.assert_array_equal(normalized[:, 0, 0], [-1, -1, -1])
        val = normalized_array(read_image(self.root / "images/val/tile.tif"), DEFAULT_CHANNELS, mean, std)
        np.testing.assert_array_equal(val[:, 0, 0], [99, 99, 99])

    def test_source_255_is_foreground_and_unsupported_labels_fail(self):
        path = self.root / "masks/train/tile.tif"
        mapped = read_mask(path, (256, 256))
        self.assertEqual(int(mapped.sum()), 100)
        self.assertEqual(int(mapped[20, 40]), 1)
        self.mask[0, 0] = 2
        tifffile.imwrite(path, self.mask)
        with self.assertRaisesRegex(ValueError, "0, 1 or 255"):
            read_mask(path, (256, 256))

    def test_shape_and_pairing_validation(self):
        path = self.root / "images/train/odd.tif"
        tifffile.imwrite(path, np.zeros((255, 256, 4), dtype=np.float16), photometric="minisblack")
        with self.assertRaisesRegex(ValueError, "Expected HWC"):
            read_image(path)
        with self.assertRaisesRegex(ValueError, "missing masks"):
            paired_files(self.root, "train")

    def test_fingerprint_survives_move_and_detects_changed_mask(self):
        original = build_manifest(self.root, ("train", "val"))["fingerprint"]
        other = Path(self.temporary.name) / "moved"
        shutil.copytree(self.root, other)
        self.assertEqual(original, build_manifest(other, ("train", "val"))["fingerprint"])
        changed = self.mask.copy()
        changed[0, 0] = 1
        tifffile.imwrite(other / "masks/train/tile.tif", changed)
        self.assertNotEqual(original, build_manifest(other, ("train", "val"))["fingerprint"])

    def test_prepare_detects_identical_selected_channels_across_splits(self):
        # An image can have different Blue values and still duplicate the actual model input.
        duplicate = self.base.copy()
        duplicate[..., 0] += 1000
        self.write_pair("val", "tile.tif", duplicate, self.mask)
        report = inspect_dataset(self.root, output=None, preview_samples=0)
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["cross_split_duplicates"], [["train/tile.tif", "val/tile.tif"]])
        self.assertEqual(report["normalization_source"], "train")

    @unittest.skipUnless(importlib.util.find_spec("torch"), "Torch is optional for CPU-only preparation")
    def test_synchronized_augmentation_and_tensor_output(self):
        image = self.base.copy()
        image[..., 3] = self.mask.astype(np.float16)
        self.write_pair("train", "tile.tif", image, self.mask)
        dataset = PairedTiffDataset(self.root, "train", np.zeros(3), np.ones(3), augment=True)
        with patch("rs3mamba_baseline.dataset.random.random", return_value=0.1), patch(
                "rs3mamba_baseline.dataset.random.randrange", return_value=1):
            sample = dataset[0]
        self.assertEqual(tuple(sample["image"].shape), (3, 256, 256))
        np.testing.assert_array_equal((sample["image"][0].numpy() > 0).astype(np.int64), sample["mask"].numpy())


class MetricTests(unittest.TestCase):
    def test_known_confusion_and_pixel_aggregation(self):
        target = np.array([0, 0, 0, 1, 1, 1])
        prediction = np.array([0, 0, 1, 0, 1, 1])
        matrix = confusion_matrix(target, prediction)
        np.testing.assert_array_equal(matrix, [[2, 1], [1, 2]])
        metrics = metrics_from_confusion(matrix)
        self.assertAlmostEqual(metrics["foreground_iou"], 0.5)
        self.assertAlmostEqual(metrics["dice"], 2 / 3)
        self.assertAlmostEqual(metrics["precision"], 2 / 3)
        summed = matrix + confusion_matrix(np.zeros(100), np.zeros(100))
        self.assertAlmostEqual(metrics_from_confusion(summed)["foreground_iou"], 0.5)

    def test_undefined_and_invalid_inputs(self):
        metrics = metrics_from_confusion(np.array([[10, 0], [0, 0]]))
        self.assertIsNone(metrics["foreground_iou"])
        self.assertIsNone(metrics["dice"])
        self.assertEqual(metrics["miou"], 1)
        with self.assertRaises(ValueError):
            confusion_matrix(np.array([0, 255]), np.array([0, 1]))
        with self.assertRaises(ValueError):
            confusion_matrix(np.array([0, 1]), np.array([[0, 1]]))


if __name__ == "__main__":
    unittest.main()
