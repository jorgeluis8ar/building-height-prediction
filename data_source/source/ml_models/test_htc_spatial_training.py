#!/usr/bin/env python3
"""Contract tests for the five-channel spatial HTC dataset and loader."""

from __future__ import annotations

import csv
from pathlib import Path
import random
import unittest

import numpy as np
import torch

from htc_spatial_training import CityHeightBatchSampler, SpatialHTCDataset, load_stats


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3]
DATASET_DIR = (
    REPO_ROOT
    / "data_source/data/ml_models/generated/htc_dc_net/"
    / "nyc_la_off_nadir_rgb_nir_mask_spatial_v1"
)


def read_ids(name: str) -> list[str]:
    return [
        line.strip()
        for line in (DATASET_DIR / f"{name}.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def manifest() -> dict[str, dict[str, str]]:
    with (DATASET_DIR / "chips_manifest.csv").open("r", newline="", encoding="utf-8") as stream:
        return {row["chip_id"]: row for row in csv.DictReader(stream)}


class SpatialDatasetContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rows = manifest()

    def test_splits_cover_all_chips_without_overlap(self) -> None:
        splits = {name: set(read_ids(name)) for name in ("train", "val", "test")}
        self.assertFalse(splits["train"] & splits["val"])
        self.assertFalse(splits["train"] & splits["test"])
        self.assertFalse(splits["val"] & splits["test"])
        self.assertEqual(set.union(*splits.values()), set(read_ids("all")))

    def test_spatial_blocks_do_not_cross_splits(self) -> None:
        block_splits: dict[str, set[str]] = {}
        for row in self.rows.values():
            block_splits.setdefault(row["spatial_block_id"], set()).add(row["split"])
        self.assertTrue(all(len(splits) == 1 for splits in block_splits.values()))

    def test_stats_are_five_channel_and_train_only(self) -> None:
        mean, std = load_stats(DATASET_DIR)
        self.assertEqual(mean.size, 5)
        self.assertEqual(std.size, 5)
        self.assertTrue(np.all(std > 0))
        import pickle

        with (DATASET_DIR / "stats/image_stats.pickle").open("rb") as stream:
            metadata = pickle.load(stream)
        self.assertEqual(metadata["stats_split"], "train")
        self.assertEqual(metadata["training_chips"], len(read_ids("train")))

    def test_augmented_mask_channel_matches_target_mask(self) -> None:
        rows = [self.rows[chip_id] for chip_id in read_ids("train")[:8]]
        dataset = SpatialHTCDataset(
            DATASET_DIR,
            rows,
            augment_spatial=True,
            augment_spectral=True,
        )
        mean, std = load_stats(DATASET_DIR)
        for seed in range(8):
            random.seed(seed)
            _, normalized_image, target = dataset[seed % len(dataset)]
            recovered_mask = normalized_image[4] * float(std[4]) + float(mean[4])
            self.assertTrue(torch.allclose(recovered_mask, target["mask"][0], atol=1e-5))
            self.assertTrue(torch.all((target["mask"] == 0) | (target["mask"] == 1)))

    def test_balanced_batches_have_requested_composition(self) -> None:
        rows = [self.rows[chip_id] for chip_id in read_ids("train")]
        sampler = CityHeightBatchSampler(rows, seed=20260722, batch_size=8)
        batch = next(iter(sampler))
        counts: dict[tuple[str, str], int] = {}
        for index in batch:
            row = rows[index]
            key = (row["source_city"], row["height_category"])
            counts[key] = counts.get(key, 0) + 1
        for city in ("los_angeles", "new_york_city"):
            self.assertEqual(counts[(city, "lowrise")], 2)
            self.assertEqual(counts[(city, "midrise")], 1)
            self.assertEqual(counts[(city, "highrise")], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
