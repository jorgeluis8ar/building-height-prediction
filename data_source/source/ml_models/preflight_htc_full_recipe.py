#!/usr/bin/env python3
"""Acceptance checks for the four-channel full HTC-DC training recipe."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import rasterio
import torch


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3]
HTC_REPO_DIR = REPO_ROOT / "data_source/source/ml_models/external/HTC-DC-Net"
DEFAULT_DATASET = REPO_ROOT / "data_source/data/ml_models/generated/htc_dc_net/nyc_la_off_nadir_rgb_nir_full_recipe_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--device", choices=["cpu", "mps"], default="mps")
    parser.add_argument(
        "--height-loss-weighting",
        choices=["none", "bins"],
        default="none",
    )
    parser.add_argument("--height-bin-edges", default="3,6,10,25,50")
    parser.add_argument("--height-bin-weights", default="4,3,2,1,3,8")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def split_ids(dataset_dir: Path, name: str) -> list[str]:
    return [
        line.strip()
        for line in (dataset_dir / f"{name}.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def validate_dataset(dataset_dir: Path) -> dict:
    splits = {name: split_ids(dataset_dir, name) for name in ("train", "val", "test")}
    split_sets = {name: set(values) for name, values in splits.items()}
    if any(split_sets[a] & split_sets[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise RuntimeError("Split membership overlaps")
    all_ids = split_ids(dataset_dir, "all")
    if set(all_ids) != set().union(*split_sets.values()):
        raise RuntimeError("Split files do not cover all.txt exactly")
    with (dataset_dir / "chips_manifest.csv").open("r", newline="", encoding="utf-8") as stream:
        manifest = {row["chip_id"]: row for row in csv.DictReader(stream)}
    expected_order = "red;green;blue;nir"
    for chip_id in all_ids:
        row = manifest[chip_id]
        if row["channel_order"] != expected_order or int(row["input_channels"]) != 4:
            raise RuntimeError(f"Wrong manifest channel contract for {chip_id}")
        paths = [
            dataset_dir / "image" / f"{chip_id}_IMG.tif",
            dataset_dir / "mask" / f"{chip_id}_BLG.tif",
            dataset_dir / "ndsm" / f"{chip_id}_AGL.tif",
        ]
        with rasterio.open(paths[0]) as image, rasterio.open(paths[1]) as mask, rasterio.open(paths[2]) as ndsm:
            if image.count != 4 or mask.count != 1 or ndsm.count != 1:
                raise RuntimeError(f"Wrong raster band count for {chip_id}")
            signatures = [(src.crs, src.transform, src.width, src.height, src.bounds, src.res) for src in (image, mask, ndsm)]
            if signatures[0] != signatures[1] or signatures[0] != signatures[2]:
                raise RuntimeError(f"Raster alignment mismatch for {chip_id}")
            if image.shape != (256, 256) or image.res != (3.0, 3.0):
                raise RuntimeError(f"Wrong image grid for {chip_id}: shape={image.shape}, res={image.res}")
            mask_values = np.unique(mask.read(1))
            if not set(mask_values.tolist()).issubset({0, 1, 255}):
                raise RuntimeError(f"Non-binary footprint mask for {chip_id}: {mask_values}")

    stored_mean, stored_std = torch.load(dataset_dir / "image_stats.pickle")
    sum_values = np.zeros(4, dtype="float64")
    sum_squares = np.zeros(4, dtype="float64")
    count = 0
    for chip_id in splits["train"]:
        with rasterio.open(dataset_dir / "image" / f"{chip_id}_IMG.tif") as src:
            values = src.read().astype("float64").reshape(4, -1)
        sum_values += values.sum(axis=1)
        sum_squares += (values**2).sum(axis=1)
        count += values.shape[1]
    computed_mean = sum_values / count
    computed_std = np.sqrt(sum_squares / count - computed_mean**2)
    if not np.allclose(stored_mean, computed_mean, rtol=1e-6, atol=1e-4):
        raise RuntimeError(f"Stored means are not train-only means: {stored_mean} vs {computed_mean}")
    if not np.allclose(stored_std, computed_std, rtol=1e-6, atol=1e-4):
        raise RuntimeError(f"Stored stds are not train-only stds: {stored_std} vs {computed_std}")
    city_counts = {}
    for split, ids in splits.items():
        city_counts[split] = {
            city: sum(manifest[chip_id]["source_city"] == city for chip_id in ids)
            for city in ("los_angeles", "new_york_city")
        }
    return {
        "split_counts": {name: len(values) for name, values in splits.items()},
        "city_counts": city_counts,
        "scenes": sorted({(row["source_city"], row["source_scene_id"]) for row in manifest.values()}),
        "channel_order": ["red", "green", "blue", "nir"],
        "resolution_m": 3.0,
        "image_shape": [4, 256, 256],
        "train_only_mean": computed_mean.tolist(),
        "train_only_std": computed_std.tolist(),
        "alignment_status": "passed_all_244_chips",
    }


def model_config(
    dataset_dir: Path,
    device: str,
    height_loss_weighting: str,
    height_bin_edges: list[float],
    height_bin_weights: list[float],
) -> dict:
    relative = str(dataset_dir.relative_to(REPO_ROOT))
    return {
        "data_dir": relative,
        "model": "htcdc",
        "backbone": "efficientnetb5",
        "in_channels": 4,
        "num_classes": 256,
        "patch_size": 4,
        "fusion_mode": "third",
        "head_tail_cut": True,
        "earlier": True,
        "prob_loss": "gaussian",
        "prob_loss_bg": "uniform",
        "height_loss_weighting": height_loss_weighting,
        "height_bin_edges": height_bin_edges,
        "height_bin_weights": height_bin_weights,
        "background_loss_weight": 0.0,
        "chamfer_weight": 0.01,
        "optimizer": "AdamW",
        "lr": 0.0001,
        "weight_decay": 0.01,
        "device": device,
        "test": False,
    }


def validate_model(
    dataset_dir: Path,
    device: str,
    height_loss_weighting: str,
    height_bin_edges: list[float],
    height_bin_weights: list[float],
) -> dict:
    sys.path.insert(0, str(HTC_REPO_DIR))
    from build import get_model_and_optimizer
    from htc_full_recipe_training import FullRecipeDataset, read_manifest

    cfg = model_config(
        dataset_dir,
        device,
        height_loss_weighting,
        height_bin_edges,
        height_bin_weights,
    )
    manifest = read_manifest(dataset_dir)
    train_ids = split_ids(dataset_dir, "train")[:8]
    dataset = FullRecipeDataset(dataset_dir, [manifest[chip_id] for chip_id in train_ids], False, False)
    batch = [dataset[index] for index in range(8)]
    chip_ids = [item[0] for item in batch]
    image = torch.stack([item[1] for item in batch]).to(device)
    gt = {
        key: torch.stack([item[2][key] for item in batch]).to(device)
        for key in ("ndsm", "mask")
    }
    if image.shape != (8, 4, 256, 256) or gt["mask"].shape != (8, 1, 256, 256):
        raise RuntimeError(f"Wrong smoke shapes: image={image.shape}, mask={gt['mask'].shape}")
    model, optimizer = get_model_and_optimizer(cfg)
    model.to(device)
    stem = model.model.encoder.original_model.conv_stem
    if stem.in_channels != 4:
        raise RuntimeError(f"EfficientNet stem has {stem.in_channels} channels")
    if model.backbone_name != "efficientnetb5" or model.num_bins != 256:
        raise RuntimeError("Backbone/bin configuration mismatch")
    if model.height_loss_weighting != height_loss_weighting:
        raise RuntimeError("Requested height-loss weighting is not active")
    if model.height_bin_edges != height_bin_edges or model.height_bin_weights != height_bin_weights:
        raise RuntimeError("Height-bin edges or weights were not propagated to the model")
    if height_loss_weighting == "bins":
        probe = torch.tensor([[[[1.0, 4.0, 8.0, 15.0, 35.0, 60.0]]]], device=device)
        expected = torch.tensor([4.0, 3.0, 2.0, 1.0, 3.0, 8.0], device=device)
        observed = model.height_weights(probe).flatten()
        if not torch.equal(observed, expected):
            raise RuntimeError(f"Height-bin weight mapping failed: {observed} vs {expected}")
    before = next(parameter for parameter in model.parameters() if parameter.requires_grad).detach().clone()
    model.train()
    losses, prediction = model(image, gt)
    required_families = (
        "mae_", "bin_chamfer_", "cross_entropy_", "loss_probability_", "loss_probability_bg_"
    )
    for family in required_families:
        keys = sorted(key for key in losses if key.startswith(family) and "bg_" not in key[len(family):])
        if family == "loss_probability_":
            keys = sorted(
                key for key in losses
                if key.startswith("loss_probability_") and not key.startswith("loss_probability_bg_")
            )
        if len(keys) != 4:
            raise RuntimeError(f"Expected four {family} losses, found {keys}")
        values = torch.stack([losses[key] for key in keys])
        if not torch.all(torch.isfinite(values)) or not torch.all(values > 0):
            raise RuntimeError(f"Invalid {family} losses: {values}")
    if len(prediction["ndsm_intermediate"]) != 4 or len(prediction["bin"]) != 4:
        raise RuntimeError("Full four-level supervision is inactive")
    if len(prediction["htc_prob"]) != 4 or len(prediction["prob"]) != 4:
        raise RuntimeError("HTC probability outputs are incomplete")
    if any(losses[f"background_mae_{index}"].item() != 0 for index in range(4)):
        raise RuntimeError("Custom background L1 is not disabled")
    optimizer.zero_grad(set_to_none=True)
    losses["loss_total"].backward()
    if any(
        parameter.grad is not None and not torch.all(torch.isfinite(parameter.grad))
        for parameter in model.parameters()
    ):
        raise RuntimeError("Non-finite gradients in smoke step")
    optimizer.step()
    after = next(parameter for parameter in model.parameters() if parameter.requires_grad).detach()
    if torch.equal(before, after):
        raise RuntimeError("Optimizer step did not change parameters")
    return {
        "chip_ids": chip_ids,
        "input_shape": list(image.shape),
        "mask_shape": list(gt["mask"].shape),
        "backbone": model.backbone_name,
        "backbone_runtime_name": "tf_efficientnet_b5_ap",
        "stem_in_channels": stem.in_channels,
        "adaptive_bins": model.num_bins,
        "patch_size": model.patch_size,
        "head_tail_cut": model.head_tail_cut,
        "earlier": model.earlier,
        "supervised_levels": len(prediction["ndsm_intermediate"]),
        "loss_total": float(losses["loss_total"].detach().cpu()),
        "optimizer_weight_decay": optimizer.param_groups[0]["weight_decay"],
        "height_loss_weighting": model.height_loss_weighting,
        "height_bin_edges": model.height_bin_edges,
        "height_bin_weights": model.height_bin_weights,
        "gradient_status": "finite",
        "optimizer_step": "changed_parameters",
    }


def validate_overlap() -> dict:
    from htc_sliding_window_inference import overlap_predict

    image = np.zeros((4, 512, 512), dtype="float32")
    call = {"value": 0}

    def predictor(window: np.ndarray) -> np.ndarray:
        call["value"] += 1
        return np.full(window.shape[1:], call["value"], dtype="float32")

    mean, variance, count = overlap_predict(image, predictor, window=256, stride=128)
    if count.max() != 4 or count.min() != 1 or np.any(variance < 0):
        raise RuntimeError(
            f"Sliding-window acceptance failed: count={count.min()}..{count.max()}, "
            f"variance_min={variance.min()}"
        )
    return {
        "window": 256,
        "stride": 128,
        "prediction_count_min": int(count.min()),
        "prediction_count_max": int(count.max()),
        "variance_min": float(variance.min()),
        "variance_max": float(variance.max()),
    }


def main() -> None:
    args = parse_args()
    os.environ.setdefault("WANDB_MODE", "offline")
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    os.environ.setdefault("TORCH_HOME", "/private/tmp/torch_htc_cache")
    random.seed(20260723)
    np.random.seed(20260723)
    torch.manual_seed(20260723)
    dataset_dir = args.dataset_dir.resolve()
    height_bin_edges = parse_float_list(args.height_bin_edges)
    height_bin_weights = parse_float_list(args.height_bin_weights)
    if len(height_bin_weights) != len(height_bin_edges) + 1:
        raise ValueError("Expected one more height-bin weight than edge")
    report = {
        "dataset": validate_dataset(dataset_dir),
        "model_smoke": validate_model(
            dataset_dir,
            args.device,
            args.height_loss_weighting,
            height_bin_edges,
            height_bin_weights,
        ),
        "sliding_window": validate_overlap(),
        "status": "passed",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
