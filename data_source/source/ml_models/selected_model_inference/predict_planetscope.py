#!/usr/bin/env python3
"""Predict an AGL-height GeoTIFF from one four-band PlanetScope GeoTIFF."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Optional

import numpy as np
import rasterio
import torch
import torch.nn.functional as F


BUNDLE_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Four-band RGB+NIR GeoTIFF.")
    parser.add_argument("--output", type=Path, required=True, help="Predicted AGL GeoTIFF in meters.")
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--window", type=int, default=256)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--nodata", type=float, default=-9999.0)
    parser.add_argument(
        "--htc-repo-dir",
        type=Path,
        default=None,
        help="Path to data_source/source/ml_models/external/HTC-DC-Net when auto-detection fails.",
    )
    parser.add_argument("--skip-checksum", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def find_htc_repo(explicit: Optional[Path]) -> Path:
    candidates = []
    if explicit is not None:
        candidates.append(explicit)
    environment = os.environ.get("HTC_DC_NET_REPO")
    if environment:
        candidates.append(Path(environment))
    for parent in BUNDLE_DIR.parents:
        candidates.append(parent / "data_source/source/ml_models/external/HTC-DC-Net")
    for candidate in candidates:
        candidate = candidate.resolve()
        if (candidate / "build.py").is_file() and (candidate / "htcdc.py").is_file():
            return candidate
    raise FileNotFoundError(
        "Could not locate HTC-DC-Net source. Pass --htc-repo-dir or set HTC_DC_NET_REPO."
    )


def window_starts(length: int, window: int, stride: int) -> list[int]:
    if length <= window:
        return [0]
    starts = list(range(0, length - window + 1, stride))
    if starts[-1] != length - window:
        starts.append(length - window)
    return starts


def main() -> None:
    args = parse_args()
    if args.window != 256:
        raise ValueError("This model was trained with 256x256 chips; --window must be 256.")
    if not 0 < args.stride <= args.window:
        raise ValueError("--stride must be greater than zero and no larger than --window.")

    manifest = json.loads((BUNDLE_DIR / "inference_manifest.json").read_text(encoding="utf-8"))
    weights_path = BUNDLE_DIR / manifest["files"]["model_weights"]
    stats_path = BUNDLE_DIR / manifest["files"]["image_stats"]
    if not args.skip_checksum:
        actual = sha256(weights_path)
        expected = manifest["model"]["checkpoint_sha256"]
        if actual != expected:
            raise RuntimeError(f"Checkpoint SHA-256 mismatch: expected {expected}, found {actual}")

    mean, std = torch.load(stats_path, map_location="cpu")
    mean_array = np.asarray(mean, dtype="float32")
    std_array = np.asarray(std, dtype="float32")
    if mean_array.shape != (4,) or std_array.shape != (4,) or np.any(std_array <= 0):
        raise RuntimeError("image_stats.pickle must contain four valid RGB+NIR means and standard deviations.")

    checkpoint = torch.load(weights_path, map_location=args.device)
    cfg = dict(checkpoint["cfg"])
    cfg.update(
        device=args.device,
        restore=False,
        in_channels=4,
        data_dir=str(BUNDLE_DIR),
        pretrained_backbone=False,
        efficientnet_repo_dir=str(BUNDLE_DIR / "efficientnet_source"),
    )
    htc_repo = find_htc_repo(args.htc_repo_dir)
    sys.path.insert(0, str(htc_repo))
    from build import get_model_and_optimizer  # noqa: PLC0415

    model, _ = get_model_and_optimizer(cfg)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(args.device)
    model.eval()

    with rasterio.open(args.input) as source:
        if source.count != 4:
            raise RuntimeError(f"Input must contain four bands ordered RGB+NIR; found {source.count}.")
        image = source.read().astype("float32")
        profile = source.profile.copy()
        valid = np.all(np.isfinite(image), axis=0)
        if source.nodata is not None:
            valid &= np.all(image != source.nodata, axis=0)
        valid &= np.any(image != 0, axis=0)

    height, width = image.shape[1:]
    padded_height = max(args.window, args.window + math.ceil(max(0, height - args.window) / args.stride) * args.stride)
    padded_width = max(args.window, args.window + math.ceil(max(0, width - args.window) / args.stride) * args.stride)
    padded = np.zeros((4, padded_height, padded_width), dtype="float32")
    padded[:, :height, :width] = image
    prediction_sum = np.zeros((padded_height, padded_width), dtype="float64")
    prediction_count = np.zeros((padded_height, padded_width), dtype="uint16")
    mean_tensor = torch.from_numpy(mean_array)[None, :, None, None].to(args.device)
    std_tensor = torch.from_numpy(std_array)[None, :, None, None].to(args.device)

    starts_y = window_starts(padded_height, args.window, args.stride)
    starts_x = window_starts(padded_width, args.window, args.stride)
    total = len(starts_y) * len(starts_x)
    completed = 0
    with torch.no_grad():
        for row in starts_y:
            for col in starts_x:
                window = padded[:, row : row + args.window, col : col + args.window]
                tensor = torch.from_numpy(np.ascontiguousarray(window))[None].to(args.device)
                tensor = (tensor - mean_tensor) / std_tensor
                result = model.model(tensor)
                pred = F.interpolate(
                    result[1][-1], size=(args.window, args.window), mode="bilinear", align_corners=False
                )[0, 0].detach().cpu().numpy()
                prediction_sum[row : row + args.window, col : col + args.window] += pred
                prediction_count[row : row + args.window, col : col + args.window] += 1
                completed += 1
                if completed == 1 or completed % 25 == 0 or completed == total:
                    print(f"Predicted windows: {completed}/{total}", flush=True)

    if np.any(prediction_count[:height, :width] == 0):
        raise RuntimeError("Inference left uncovered pixels.")
    prediction = (prediction_sum / np.maximum(prediction_count, 1))[:height, :width].astype("float32")
    prediction = np.maximum(prediction, 0)
    prediction[~valid] = args.nodata
    args.output.parent.mkdir(parents=True, exist_ok=True)
    profile.update(count=1, dtype="float32", nodata=args.nodata, compress="deflate", predictor=2)
    with rasterio.open(args.output, "w", **profile) as destination:
        destination.write(prediction, 1)
        destination.set_band_description(1, "predicted_agl_m")
        destination.update_tags(
            model_id=manifest["model"]["model_id"],
            units="meters",
            channel_order="red,green,blue,nir",
        )
    finite = prediction[prediction != args.nodata]
    print(f"Output: {args.output}")
    print(f"Valid prediction range: {float(finite.min()):.3f} to {float(finite.max()):.3f} m")


if __name__ == "__main__":
    main()
