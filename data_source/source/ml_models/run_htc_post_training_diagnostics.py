#!/usr/bin/env python3
"""Run the standard post-training diagnostics package for HTC-DC Net models.

This wrapper is intentionally conservative: it calls the existing project
diagnostic scripts, checks expected files, and writes all outputs under the
model run's `evaluation/` folder.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3]
PYTHON = sys.executable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--splits",
        default="train,val,test,all",
        help="Comma-separated splits to diagnose.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--collapse-std-threshold", type=float, default=0.05)
    parser.add_argument("--collapse-min-share", type=float, default=0.8)
    parser.add_argument("--height-stat", choices=["mean", "median"], default="median")
    parser.add_argument("--min-component-pixels", type=int, default=3)
    parser.add_argument(
        "--height-bin-edges",
        default="0,10,20,30,40",
        help="Comma-separated LiDAR height bin edges in meters.",
    )
    parser.add_argument("--resolution-meters", type=float, default=3.0)
    parser.add_argument("--prediction-nodata", type=float, default=-9999.0)
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--skip-mosaics", action="store_true")
    parser.add_argument("--skip-full-city-panels", action="store_true")
    return parser.parse_args()


def run_command(command: list[str]) -> None:
    print("+ " + " ".join(command))
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT))


def epoch_label(epoch: int) -> str:
    return f"epoch_{epoch:03d}"


def read_csv(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def resolve_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def export_split_predictions(args: argparse.Namespace, split: str, checkpoint: Path) -> str:
    label = epoch_label(args.epoch)
    output_subdir = f"{split}_predictions_{label}_unmasked"
    if args.skip_export:
        expected_dir = args.run_dir / output_subdir
        if not expected_dir.exists():
            raise FileNotFoundError(expected_dir)
        return output_subdir

    run_command(
        [
            PYTHON,
            rel(SCRIPT_PATH.parent / "evaluate_htc_checkpoint_on_split.py"),
            "--run-dir",
            rel(args.run_dir),
            "--dataset-dir",
            rel(args.dataset_dir),
            "--checkpoint",
            rel(checkpoint),
            "--split",
            split,
            "--output-subdir",
            output_subdir,
            "--output-label",
            label + "_unmasked",
            "--collapse-std-threshold",
            str(args.collapse_std_threshold),
            "--collapse-min-share",
            str(args.collapse_min_share),
            "--prediction-nodata",
            str(args.prediction_nodata),
            "--device",
            args.device,
        ]
    )
    return output_subdir


def mosaic_all_predictions(args: argparse.Namespace, predictions_subdir: str) -> None:
    if args.skip_mosaics:
        return
    label = epoch_label(args.epoch)
    summary_csv = args.run_dir / f"all_predictions_summary_{label}_unmasked.csv"
    if not summary_csv.exists():
        raise FileNotFoundError(summary_csv)

    for masked in [False, True]:
        suffix = "masked" if masked else "unmasked"
        command = [
            PYTHON,
            rel(SCRIPT_PATH.parent / "mosaic_htc_prediction_chips.py"),
            "--run-dir",
            rel(args.run_dir),
            "--summary-csv",
            rel(summary_csv),
            "--output-dir",
            rel(args.run_dir / f"mosaics_all_{label}_{suffix}_3m"),
            "--predictions-subdir",
            predictions_subdir,
            "--epoch",
            str(args.epoch),
            "--data-dir",
            rel(args.dataset_dir),
            "--resolution-meters",
            str(args.resolution_meters),
            "--output-suffix",
            "_3m",
            "--nodata",
            str(args.prediction_nodata),
        ]
        if masked:
            command.append("--mask-predictions")
        run_command(command)


def run_scatter(args: argparse.Namespace, split: str, predictions_subdir: str) -> Path:
    label = epoch_label(args.epoch)
    output_dir = args.run_dir / "evaluation" / f"{split}_building_scatter_{label}"
    run_command(
        [
            PYTHON,
            rel(SCRIPT_PATH.parent / "plot_building_height_scatter.py"),
            "--run-dir",
            rel(args.run_dir),
            "--data-dir",
            rel(args.dataset_dir),
            "--split",
            split,
            "--predictions-subdir",
            predictions_subdir,
            "--output-dir",
            rel(output_dir),
            "--height-stat",
            args.height_stat,
            "--min-component-pixels",
            str(args.min_component_pixels),
        ]
    )
    return output_dir


def create_three_panel_scatter(scatter_dir: Path, split: str, epoch: int) -> Path:
    component_csv = scatter_dir / "building_component_predictions.csv"
    metrics_csv = scatter_dir / "building_scatter_metrics.csv"
    if not component_csv.exists():
        raise FileNotFoundError(component_csv)
    if not metrics_csv.exists():
        raise FileNotFoundError(metrics_csv)

    rows = read_csv(component_csv)
    metrics = {row["group"]: row for row in read_csv(metrics_csv)}
    groups = [
        ("all", "All"),
        ("los_angeles", "Los Angeles"),
        ("new_york_city", "New York City"),
    ]
    all_x = np.array([float(row["lidar_height_m"]) for row in rows], dtype=float)
    all_y = np.array([float(row["predicted_height_m"]) for row in rows], dtype=float)
    max_axis = float(np.nanpercentile(np.concatenate([all_x, all_y]), 99.5))
    max_axis = max(max_axis, 10.0)

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.8), sharex=True, sharey=True, constrained_layout=True)
    fig.suptitle(
        f"Building-Level Predicted Height vs LiDAR Height | {split.title()} sample | epoch {epoch}",
        fontsize=16,
        fontweight="bold",
    )
    rng = np.random.default_rng(20260720)
    for ax, (group, title) in zip(axes, groups):
        group_rows = rows if group == "all" else [row for row in rows if row["source_city"] == group]
        if len(group_rows) > 10000:
            indexes = rng.choice(len(group_rows), size=10000, replace=False)
            plot_rows = [group_rows[int(index)] for index in indexes]
        else:
            plot_rows = group_rows
        x = np.array([float(row["lidar_height_m"]) for row in plot_rows], dtype=float)
        y = np.array([float(row["predicted_height_m"]) for row in plot_rows], dtype=float)
        ax.scatter(x, y, s=8, alpha=0.28, edgecolors="none")
        ax.plot([0, max_axis], [0, max_axis], color="black", linewidth=1, linestyle="--")
        group_metrics = metrics[group]
        intercept = float(group_metrics["intercept"])
        slope = float(group_metrics["slope"])
        line_x = np.array([0, max_axis])
        ax.plot(line_x, intercept + slope * line_x, color="#b2182b", linewidth=2)
        ax.set_title(title)
        ax.set_xlim(0, max_axis)
        ax.set_ylim(0, max_axis)
        ax.grid(alpha=0.2)
        ax.text(
            0.04,
            0.96,
            (
                f"y = {intercept:.2f} + {slope:.3f}x\n"
                f"RMSE = {float(group_metrics['rmse_m']):.2f} m\n"
                f"R2 = {float(group_metrics['r2']):.3f}\n"
                f"N = {int(group_metrics['buildings']):,}"
            ),
            transform=ax.transAxes,
            va="top",
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.75"},
        )
    axes[0].set_ylabel("Predicted building height (m)")
    for ax in axes:
        ax.set_xlabel("LiDAR-derived building height (m)")

    output_path = scatter_dir / f"building_height_scatter_three_panel_{split}_epoch_{epoch:03d}.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def run_height_bins(args: argparse.Namespace, split: str, scatter_dir: Path) -> Path:
    label = f"{split}_epoch_{args.epoch:03d}"
    output_dir = args.run_dir / "evaluation" / f"{split}_height_bin_error_epoch_{args.epoch:03d}"
    run_command(
        [
            PYTHON,
            rel(SCRIPT_PATH.parent / "plot_height_bin_error.py"),
            "--run-dir",
            rel(args.run_dir),
            "--component-csv",
            rel(scatter_dir / "building_component_predictions.csv"),
            "--output-dir",
            rel(output_dir),
            "--label",
            label,
            "--bin-edges",
            args.height_bin_edges,
        ]
    )
    return output_dir


def infer_full_scene_bases(dataset_dir: Path, visited: set[Path] | None = None) -> dict[str, dict[str, Path | str]]:
    """Infer full-scene raster bases, following derived-dataset links if needed."""
    dataset_dir = dataset_dir.resolve()
    visited = visited or set()
    if dataset_dir in visited:
        return {}
    visited.add(dataset_dir)

    rows = read_csv(dataset_dir / "chips_manifest.csv")
    output: dict[str, dict[str, Path | str]] = {}
    for row in rows:
        city = row["source_city"]
        if city in output:
            continue
        source_root = resolve_path(row["source_dataset_root"])
        full_scene_dir = source_root / "full_scene"
        image_paths = sorted(full_scene_dir.glob("*_IMG.tif"))
        if not image_paths:
            continue
        image_path = image_paths[0]
        base = Path(str(image_path)[: -len("_IMG.tif")])
        output[city] = {
            "base": base,
            "scene_id": row.get("source_scene_id", ""),
        }
    if output:
        return output

    source_roots = sorted({resolve_path(row["source_dataset_root"]) for row in rows if row.get("source_dataset_root")})
    for source_root in source_roots:
        manifest = source_root / "chips_manifest.csv"
        if not manifest.exists():
            continue
        output.update(infer_full_scene_bases(source_root, visited))
        if {"los_angeles", "new_york_city"}.issubset(output):
            break
    return output


def read_to_target(path: Path, ref_profile: dict, bands: list[int] | None = None) -> np.ndarray:
    with rasterio.open(path) as src:
        vrt_kwargs = {
            "crs": ref_profile["crs"],
            "transform": ref_profile["transform"],
            "width": ref_profile["width"],
            "height": ref_profile["height"],
            "resampling": Resampling.nearest,
        }
        with WarpedVRT(src, **vrt_kwargs) as vrt:
            if bands is None:
                data = vrt.read(1).astype("float32")
            else:
                data = vrt.read(bands).astype("float32")
            nodata = vrt.nodata
    if nodata is not None:
        data = np.where(data == nodata, np.nan, data)
    return data


def downsample(arr: np.ndarray, max_dim: int = 1800) -> np.ndarray:
    height, width = arr.shape[-2], arr.shape[-1]
    scale = max(height / max_dim, width / max_dim, 1)
    if scale <= 1:
        return arr
    step = int(np.ceil(scale))
    if arr.ndim == 3:
        return arr[:, ::step, ::step]
    return arr[::step, ::step]


def rgb_stretch(rgb: np.ndarray) -> np.ndarray:
    arr = np.moveaxis(rgb[:3], 0, -1).astype("float32")
    output = np.zeros_like(arr, dtype="float32")
    for band_index in range(3):
        band = arr[..., band_index]
        finite = np.isfinite(band) & (band > 0)
        if finite.any():
            lower, upper = np.nanpercentile(band[finite], [2, 98])
            if upper <= lower:
                upper = lower + 1
            output[..., band_index] = np.clip((band - lower) / (upper - lower), 0, 1)
    return output


def city_title(city: str) -> str:
    return city.replace("_", " ").title()


def create_full_city_panels(args: argparse.Namespace) -> None:
    if args.skip_full_city_panels:
        return

    label = epoch_label(args.epoch)
    mosaic_dir = args.run_dir / f"mosaics_all_{label}_masked_3m"
    if not mosaic_dir.exists():
        raise FileNotFoundError(mosaic_dir)

    full_scene_bases = infer_full_scene_bases(args.dataset_dir)
    if not full_scene_bases:
        print("No full-scene rasters inferred; skipping full-city panels.")
        return

    output_dir = args.run_dir / "evaluation" / f"full_city_three_panel_epoch_{args.epoch:03d}"
    output_dir.mkdir(parents=True, exist_ok=True)

    for city, info in sorted(full_scene_bases.items()):
        base = Path(info["base"])
        image_path = Path(str(base) + "_IMG.tif")
        agl_path = Path(str(base) + "_AGL.tif")
        mask_path = Path(str(base) + "_BLG.tif")
        pred_path = mosaic_dir / f"{city}_predicted_ndsm_epoch_{args.epoch:03d}_3m.tif"
        for path in [image_path, agl_path, mask_path, pred_path]:
            if not path.exists():
                raise FileNotFoundError(path)

        with rasterio.open(agl_path) as ref:
            ref_profile = {
                "crs": ref.crs,
                "transform": ref.transform,
                "width": ref.width,
                "height": ref.height,
            }
            target = ref.read(1).astype("float32")
            if ref.nodata is not None:
                target = np.where(target == ref.nodata, np.nan, target)

        rgb = read_to_target(image_path, ref_profile, bands=[1, 2, 3])
        mask = read_to_target(mask_path, ref_profile) > 0
        pred = read_to_target(pred_path, ref_profile)
        pred_masked = np.where(mask & np.isfinite(pred), pred, np.nan)
        target_masked = np.where(mask & np.isfinite(target), target, np.nan)
        values = np.concatenate(
            [
                pred_masked[np.isfinite(pred_masked)],
                target_masked[np.isfinite(target_masked)],
            ]
        )
        vmax = float(np.nanpercentile(values, 99.5)) if values.size else 50
        vmax = max(vmax, 10.0)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), constrained_layout=True)
        fig.suptitle(
            f"{city_title(city)} | scene {info['scene_id']} | epoch {args.epoch}",
            fontsize=14,
            fontweight="bold",
        )
        axes[0].imshow(rgb_stretch(downsample(rgb)))
        axes[0].set_title("PlanetScope scene")
        image = axes[1].imshow(downsample(pred_masked), cmap="viridis", vmin=0, vmax=vmax)
        axes[1].set_title("Prediction (building mask)")
        axes[2].imshow(downsample(target_masked), cmap="viridis", vmin=0, vmax=vmax)
        axes[2].set_title("Target AGL")
        for ax in axes:
            ax.set_axis_off()
        colorbar = fig.colorbar(image, ax=axes[1:].ravel().tolist(), shrink=0.78, pad=0.012)
        colorbar.set_label("Height (m)")
        output_path = output_dir / f"{city}_three_panel_planetscope_prediction_target_epoch_{args.epoch:03d}.png"
        fig.savefig(output_path, dpi=180)
        plt.close(fig)
        print(f"Full-city panel: {output_path}")


def collect_metrics(args: argparse.Namespace, splits: list[str]) -> Path:
    rows = []
    for split in splits:
        metrics_path = (
            args.run_dir
            / "evaluation"
            / f"{split}_building_scatter_{epoch_label(args.epoch)}"
            / "building_scatter_metrics.csv"
        )
        if not metrics_path.exists():
            continue
        for row in read_csv(metrics_path):
            row = dict(row)
            row["split"] = split
            rows.append(row)
    output_path = args.run_dir / "evaluation" / f"post_training_diagnostics_summary_epoch_{args.epoch:03d}.csv"
    write_csv(output_path, rows)
    return output_path


def main() -> None:
    args = parse_args()
    args.run_dir = args.run_dir.resolve()
    args.dataset_dir = args.dataset_dir.resolve()
    checkpoint = (args.checkpoint or args.run_dir / f"model_epoch_{args.epoch:03d}.pth").resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    if not args.dataset_dir.exists():
        raise FileNotFoundError(args.dataset_dir)

    splits = [split.strip() for split in args.splits.split(",") if split.strip()]
    invalid = sorted(set(splits) - {"train", "val", "test", "all"})
    if invalid:
        raise ValueError(f"Unsupported splits: {invalid}")

    prediction_subdirs = {}
    for split in splits:
        prediction_subdirs[split] = export_split_predictions(args, split, checkpoint)

    if "all" in prediction_subdirs:
        mosaic_all_predictions(args, prediction_subdirs["all"])

    scatter_dirs = {}
    for split in splits:
        scatter_dirs[split] = run_scatter(args, split, prediction_subdirs[split])
        print(f"Three-panel scatter: {create_three_panel_scatter(scatter_dirs[split], split, args.epoch)}")
        print(f"Height-bin diagnostics: {run_height_bins(args, split, scatter_dirs[split])}")

    if "all" in prediction_subdirs:
        create_full_city_panels(args)

    summary_path = collect_metrics(args, splits)
    print(f"Diagnostics summary: {summary_path}")


if __name__ == "__main__":
    main()
