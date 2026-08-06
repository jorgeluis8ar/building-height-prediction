#!/usr/bin/env python3
"""Create full-city HTC-DC Net prediction QA panels."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3]
DEFAULT_RUN_DIR = (
    REPO_ROOT
    / "data_source/data/ml_models/generated/htc_dc_net/mini_training_runs/"
    / "nyc71_la100_seed20260707_epoch5_repo_params_guarded"
)

CITY_CONFIG = {
    "los_angeles": {
        "label": "Los Angeles",
        "prediction_template": "los_angeles_predicted_ndsm_epoch_{epoch:03d}.tif",
        "image": (
            "data_source/data/height_labels/generated/los_angeles/lidar_ndsm/htc_dc_net/"
            "20231203_182937_07_2488/full_scene/los_angeles_20231203_182937_07_2488_IMG.tif"
        ),
        "target": (
            "data_source/data/height_labels/generated/los_angeles/lidar_ndsm/htc_dc_net/"
            "20231203_182937_07_2488/full_scene/los_angeles_20231203_182937_07_2488_AGL.tif"
        ),
        "mask": (
            "data_source/data/height_labels/generated/los_angeles/lidar_ndsm/htc_dc_net/"
            "20231203_182937_07_2488/full_scene/los_angeles_20231203_182937_07_2488_BLG.tif"
        ),
    },
    "new_york_city": {
        "label": "New York City",
        "prediction_template": "new_york_city_predicted_ndsm_epoch_{epoch:03d}.tif",
        "image": (
            "data_source/data/height_labels/generated/new_york_city/lidar_ndsm/htc_dc_net/"
            "20200122_154449_92_1061/full_scene/new_york_city_20200122_154449_92_1061_IMG.tif"
        ),
        "target": (
            "data_source/data/height_labels/generated/new_york_city/lidar_ndsm/htc_dc_net/"
            "20200122_154449_92_1061/full_scene/new_york_city_20200122_154449_92_1061_AGL.tif"
        ),
        "mask": (
            "data_source/data/height_labels/generated/new_york_city/lidar_ndsm/htc_dc_net/"
            "20200122_154449_92_1061/full_scene/new_york_city_20200122_154449_92_1061_BLG.tif"
        ),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--mosaic-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--epoch", type=int, default=5)
    parser.add_argument("--max-display-pixels", type=int, default=1_600)
    parser.add_argument("--nodata", type=float, default=-9999.0)
    parser.add_argument(
        "--error-same-scale-as-prediction",
        action="store_true",
        help="Plot Prediction - target with the same palette and 0-to-height scale used for Prediction.",
    )
    return parser.parse_args()


def read_prediction(path: Path) -> tuple[np.ndarray, dict]:
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float32")
        nodata = src.nodata
        if nodata is not None:
            arr[arr == nodata] = np.nan
        profile = {
            "crs": src.crs,
            "transform": src.transform,
            "height": src.height,
            "width": src.width,
            "bounds": src.bounds,
        }
    return arr, profile


def read_aligned(path: Path, ref: dict, resampling: Resampling) -> np.ndarray:
    with rasterio.open(path) as src:
        dst = np.full((ref["height"], ref["width"]), np.nan, dtype="float32")
        source_nodata = src.nodata
        destination_nodata = -9999.0
        dst.fill(destination_nodata)
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=source_nodata,
            dst_transform=ref["transform"],
            dst_crs=ref["crs"],
            dst_nodata=destination_nodata,
            resampling=resampling,
        )
        dst[dst == destination_nodata] = np.nan
        return dst


def read_aligned_image(path: Path, ref: dict) -> np.ndarray:
    with rasterio.open(path) as src:
        band_count = min(3, src.count)
        bands = []
        source_nodata = src.nodata
        destination_nodata = -9999.0
        for band_idx in range(1, band_count + 1):
            dst = np.full((ref["height"], ref["width"]), destination_nodata, dtype="float32")
            reproject(
                source=rasterio.band(src, band_idx),
                destination=dst,
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=source_nodata,
                dst_transform=ref["transform"],
                dst_crs=ref["crs"],
                dst_nodata=destination_nodata,
                resampling=Resampling.bilinear,
            )
            dst[dst == destination_nodata] = np.nan
            bands.append(dst)
        if band_count == 1:
            bands = [bands[0], bands[0], bands[0]]
        return np.stack(bands[:3], axis=-1)


def stretch_rgb(rgb: np.ndarray) -> np.ndarray:
    out = np.zeros_like(rgb, dtype="float32")
    for band_idx in range(rgb.shape[-1]):
        band = rgb[..., band_idx]
        valid = np.isfinite(band) & (band > 0)
        if not np.any(valid):
            continue
        lo, hi = np.nanpercentile(band[valid], [2, 98])
        if hi <= lo:
            continue
        out[..., band_idx] = np.clip((band - lo) / (hi - lo), 0, 1)
    alpha = np.any(np.isfinite(rgb) & (rgb > 0), axis=-1)
    out[~alpha] = 1.0
    return out


def downsample(arr: np.ndarray, max_pixels: int) -> np.ndarray:
    factor = max(1, int(np.ceil(max(arr.shape) / max_pixels)))
    if arr.ndim == 3:
        return arr[::factor, ::factor, :]
    return arr[::factor, ::factor]


def finite_percentile(arrays: list[np.ndarray], q: float, default: float) -> float:
    values = []
    for arr in arrays:
        vals = arr[np.isfinite(arr)]
        if vals.size:
            values.append(vals)
    if not values:
        return default
    return float(np.nanpercentile(np.concatenate(values), q))


def metrics(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> dict:
    valid = np.isfinite(pred) & np.isfinite(target) & (mask > 0) & (target > 0)
    if not np.any(valid):
        return {"eval_pixels": 0}
    error = pred[valid] - target[valid]
    return {
        "eval_pixels": int(valid.sum()),
        "mae_m": float(np.mean(np.abs(error))),
        "rmse_m": float(np.sqrt(np.mean(error**2))),
        "bias_m": float(np.mean(error)),
        "pred_mean_m": float(np.mean(pred[valid])),
        "pred_std_m": float(np.std(pred[valid])),
        "pred_min_m": float(np.min(pred[valid])),
        "pred_max_m": float(np.max(pred[valid])),
        "target_mean_m": float(np.mean(target[valid])),
        "target_std_m": float(np.std(target[valid])),
        "target_min_m": float(np.min(target[valid])),
        "target_max_m": float(np.max(target[valid])),
    }


def plot_panel(
    city_slug: str,
    city_label: str,
    epoch: int,
    image: np.ndarray,
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    out_path: Path,
    stats: dict,
    max_display_pixels: int,
    error_same_scale_as_prediction: bool,
) -> None:
    valid = np.isfinite(pred) & np.isfinite(target) & (mask > 0) & (target > 0)
    image_plot = stretch_rgb(image)
    pred_plot = np.where(mask > 0, pred, np.nan)
    target_plot = np.where(mask > 0, target, np.nan)
    error_plot = np.where(valid, pred - target, np.nan)

    image_show = downsample(image_plot, max_display_pixels)
    pred_show = downsample(pred_plot, max_display_pixels)
    target_show = downsample(target_plot, max_display_pixels)
    mask_show = downsample(mask, max_display_pixels)
    error_show = downsample(error_plot, max_display_pixels)

    height_vmax = max(10.0, finite_percentile([pred_plot, target_plot], 99.0, 10.0))
    err_lim = max(10.0, min(250.0, finite_percentile([np.abs(error_plot)], 98.0, 50.0)))

    fig, axes = plt.subplots(1, 5, figsize=(29, 7), constrained_layout=True)
    fig.suptitle(
        (
            f"{city_label} | epoch {epoch:03d} | full city mosaic | "
            f"MAE={stats.get('mae_m', float('nan')):.2f} m | "
            f"RMSE={stats.get('rmse_m', float('nan')):.2f} m | "
            f"bias={stats.get('bias_m', float('nan')):.2f} m | "
            f"pred_std={stats.get('pred_std_m', float('nan')):.2f} m"
        ),
        fontsize=13,
    )

    axes[0].imshow(image_show)
    axes[0].set_title("PlanetScope scene")

    im1 = axes[1].imshow(pred_show, cmap="viridis", vmin=0, vmax=height_vmax)
    axes[1].set_title("Prediction (building mask)")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.02, label="m")

    im2 = axes[2].imshow(target_show, cmap="viridis", vmin=0, vmax=height_vmax)
    axes[2].set_title("Target AGL")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.02, label="m")

    axes[3].imshow(mask_show > 0, cmap="gray", vmin=0, vmax=1)
    axes[3].set_title("Building mask")

    if error_same_scale_as_prediction:
        im4 = axes[4].imshow(error_show, cmap="viridis", vmin=0, vmax=height_vmax)
    else:
        im4 = axes[4].imshow(error_show, cmap="coolwarm", vmin=-err_lim, vmax=err_lim)
    axes[4].set_title("Prediction - target")
    fig.colorbar(im4, ax=axes[4], fraction=0.046, pad=0.02, label="m")

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    mosaic_dir = (args.mosaic_dir or run_dir / f"mosaics_epoch_{args.epoch:03d}").resolve()
    output_dir = (args.output_dir or mosaic_dir / "full_city_qa_panels").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for city_slug, cfg in CITY_CONFIG.items():
        pred_path = mosaic_dir / cfg["prediction_template"].format(epoch=args.epoch)
        image_path = REPO_ROOT / cfg["image"]
        target_path = REPO_ROOT / cfg["target"]
        mask_path = REPO_ROOT / cfg["mask"]
        for path in (pred_path, image_path, target_path, mask_path):
            if not path.exists():
                raise FileNotFoundError(path)

        pred, ref = read_prediction(pred_path)
        image = read_aligned_image(image_path, ref)
        target = read_aligned(target_path, ref, Resampling.nearest)
        mask = read_aligned(mask_path, ref, Resampling.nearest)
        mask = np.nan_to_num(mask, nan=0.0)

        stats = metrics(pred, target, mask)
        out_path = output_dir / f"{city_slug}_full_city_prediction_qa_epoch_{args.epoch:03d}.png"
        plot_panel(
            city_slug=city_slug,
            city_label=cfg["label"],
            epoch=args.epoch,
            image=image,
            pred=pred,
            target=target,
            mask=mask,
            out_path=out_path,
            stats=stats,
            max_display_pixels=args.max_display_pixels,
            error_same_scale_as_prediction=args.error_same_scale_as_prediction,
        )

        summary_rows.append(
            {
                "city": city_slug,
                "prediction_path": str(pred_path.relative_to(REPO_ROOT)),
                "planetscope_image_path": str(image_path.relative_to(REPO_ROOT)),
                "target_agl_path": str(target_path.relative_to(REPO_ROOT)),
                "building_mask_path": str(mask_path.relative_to(REPO_ROOT)),
                "qa_png_path": str(out_path.relative_to(REPO_ROOT)),
                "crs": str(ref["crs"]),
                "width": ref["width"],
                "height": ref["height"],
                **stats,
            }
        )
        print(f"{cfg['label']}: wrote {out_path}")

    summary_path = output_dir / "full_city_prediction_qa_summary.csv"
    write_csv(summary_path, summary_rows)
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
