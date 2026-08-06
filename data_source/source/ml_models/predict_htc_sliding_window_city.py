#!/usr/bin/env python3
"""Run 256/128 overlapping HTC inference over full NYC and LA chip mosaics."""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.merge import merge
import torch
import torch.nn.functional as F


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--device", choices=["cpu", "mps"], default="mps")
    parser.add_argument("--window", type=int, default=256)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--nodata", type=float, default=-9999.0)
    return parser.parse_args()


def read_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def mosaic_city_images(dataset_dir: Path, city: str) -> tuple[np.ndarray, dict]:
    chip_ids = [chip_id for chip_id in read_ids(dataset_dir / "all.txt") if chip_id.startswith(city)]
    paths = [dataset_dir / "image" / f"{chip_id}_IMG.tif" for chip_id in chip_ids]
    sources = [rasterio.open(path) for path in paths]
    try:
        mosaic, transform = merge(sources, nodata=0, method="first")
        profile = sources[0].profile.copy()
    finally:
        for source in sources:
            source.close()
    profile.update(
        width=mosaic.shape[2],
        height=mosaic.shape[1],
        transform=transform,
        count=1,
        dtype="float32",
        compress="deflate",
        predictor=2,
    )
    return mosaic.astype("float32"), profile


def padded_length(length: int, window: int, stride: int) -> int:
    if length <= window:
        return window
    return window + math.ceil((length - window) / stride) * stride


def main() -> None:
    args = parse_args()
    os.environ.setdefault("WANDB_MODE", "offline")
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    os.environ.setdefault("TORCH_HOME", "/private/tmp/torch_htc_cache")
    sys.path.insert(0, str(SCRIPT_PATH.parent))
    from evaluate_htc_checkpoint_on_split import load_model
    from htc_full_recipe_training import load_stats
    from htc_sliding_window_inference import overlap_predict

    run_dir = args.run_dir.resolve()
    dataset_dir = args.dataset_dir.resolve()
    checkpoint = args.checkpoint.resolve()
    model, _ = load_model(checkpoint, dataset_dir, args.device)
    model.eval()
    mean, std = load_stats(dataset_dir)
    mean_tensor = torch.tensor(mean, dtype=torch.float32, device=args.device)[None, :, None, None]
    std_tensor = torch.tensor(std, dtype=torch.float32, device=args.device)[None, :, None, None]

    output_dir = run_dir / f"sliding_window_inference_epoch_{args.epoch:03d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    for city in ("los_angeles", "new_york_city"):
        image, profile = mosaic_city_images(dataset_dir, city)
        original_height, original_width = image.shape[1:]
        padded_height = padded_length(original_height, args.window, args.stride)
        padded_width = padded_length(original_width, args.window, args.stride)
        padded = np.zeros((4, padded_height, padded_width), dtype="float32")
        padded[:, :original_height, :original_width] = image

        def predictor(window: np.ndarray) -> np.ndarray:
            tensor = torch.from_numpy(np.ascontiguousarray(window))[None].to(args.device)
            tensor = (tensor - mean_tensor) / std_tensor
            with torch.no_grad():
                result = model.model(tensor)
                final_prediction = result[1][-1]
                final_prediction = F.interpolate(
                    final_prediction,
                    size=(args.window, args.window),
                    mode="bilinear",
                    align_corners=False,
                )
            return final_prediction[0, 0].detach().cpu().numpy().astype("float32")

        height_mean, height_variance, prediction_count = overlap_predict(
            padded, predictor, window=args.window, stride=args.stride
        )
        height_mean = height_mean[:original_height, :original_width]
        height_variance = height_variance[:original_height, :original_width]
        prediction_count = prediction_count[:original_height, :original_width]
        valid = np.any(image != 0, axis=0)
        height_mean = np.where(valid, height_mean, args.nodata).astype("float32")
        height_variance = np.where(valid, height_variance, args.nodata).astype("float32")
        prediction_count = np.where(valid, prediction_count, 0).astype("uint8")
        if prediction_count.max() > 4:
            raise RuntimeError(f"{city} received more than four predictions per pixel")
        if np.any(height_variance[valid] < 0):
            raise RuntimeError(f"{city} produced negative overlap variance")

        outputs = {
            "height_mean": (height_mean, "float32", args.nodata),
            "height_variance": (height_variance, "float32", args.nodata),
            "prediction_count": (prediction_count, "uint8", 0),
        }
        for suffix, (array, dtype, nodata) in outputs.items():
            output_path = output_dir / f"{city}_{suffix}_epoch_{args.epoch:03d}.tif"
            output_profile = profile.copy()
            output_profile.update(dtype=dtype, nodata=nodata, predictor=2 if dtype == "float32" else 1)
            with rasterio.open(output_path, "w", **output_profile) as destination:
                destination.write(array, 1)
        print(
            f"{city}: shape={original_height}x{original_width} "
            f"count={prediction_count[valid].min()}..{prediction_count[valid].max()} "
            f"mean={height_mean[valid].min():.3f}..{height_mean[valid].max():.3f} "
            f"variance={height_variance[valid].min():.6f}..{height_variance[valid].max():.6f}",
            flush=True,
        )
    print(f"Sliding-window outputs: {output_dir}")


if __name__ == "__main__":
    main()

