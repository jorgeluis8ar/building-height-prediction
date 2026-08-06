#!/usr/bin/env python3
"""Building-level scatter diagnostics for HTC-DC Net predictions.

The HTC model predicts raster pixels. This diagnostic summarizes those pixels
to building-mask connected components, then compares LiDAR-derived target
height against predicted height.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from skimage.measure import label as connected_components


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3]
DEFAULT_RUN_DIR = (
    REPO_ROOT
    / "data_source/data/ml_models/generated/htc_dc_net/mini_training_runs/"
    / "nyc76_la95_6ch_seed20260709_epoch10_repo_params_guarded"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument(
        "--split",
        choices=["train", "val", "test", "all"],
        default="all",
        help="Dataset split to evaluate when --data-dir points to a full HTC dataset.",
    )
    parser.add_argument("--predictions-subdir", default="predictions")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--height-stat",
        choices=["mean", "median"],
        default="median",
        help="Statistic used to summarize target and predicted raster pixels within each building component.",
    )
    parser.add_argument("--min-component-pixels", type=int, default=3)
    parser.add_argument("--max-plot-points", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260714)
    return parser.parse_args()


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


def read_band(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float32")
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan
        return arr


def summarize(values: np.ndarray, statistic: str) -> float:
    if statistic == "mean":
        return float(np.nanmean(values))
    return float(np.nanmedian(values))


def component_rows_for_chip(
    row: dict,
    data_dir: Path,
    predictions_dir: Path,
    height_stat: str,
    min_component_pixels: int,
) -> list[dict]:
    chip_id = row["chip_id"]
    pred_path = predictions_dir / f"{chip_id}_ndsm_pred.tif"
    target_path = data_dir / "ndsm" / f"{chip_id}_AGL.tif"
    mask_path = data_dir / "mask" / f"{chip_id}_BLG.tif"
    for path in [pred_path, target_path, mask_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    pred = read_band(pred_path)
    target = read_band(target_path)
    mask = read_band(mask_path) > 0
    valid = mask & np.isfinite(pred) & np.isfinite(target) & (target > 0)
    components = connected_components(valid, connectivity=1)

    rows = []
    for component_id in range(1, int(components.max()) + 1):
        component_mask = components == component_id
        pixel_count = int(component_mask.sum())
        if pixel_count < min_component_pixels:
            continue
        target_values = target[component_mask]
        pred_values = pred[component_mask]
        rows.append(
            {
                "source_city": row.get("source_city", ""),
                "chip_id": chip_id,
                "component_id": component_id,
                "component_key": f"{chip_id}_component_{component_id:05d}",
                "component_pixels": pixel_count,
                "lidar_height_m": summarize(target_values, height_stat),
                "predicted_height_m": summarize(pred_values, height_stat),
                "lidar_height_mean_m": float(np.nanmean(target_values)),
                "predicted_height_mean_m": float(np.nanmean(pred_values)),
                "lidar_height_median_m": float(np.nanmedian(target_values)),
                "predicted_height_median_m": float(np.nanmedian(pred_values)),
                "prediction_error_m": summarize(pred_values, height_stat)
                - summarize(target_values, height_stat),
            }
        )
    return rows


def regression_metrics(rows: list[dict], group: str) -> dict:
    x = np.array([float(row["lidar_height_m"]) for row in rows], dtype=float)
    y = np.array([float(row["predicted_height_m"]) for row in rows], dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if x.size < 2:
        return {
            "group": group,
            "buildings": int(x.size),
            "intercept": float("nan"),
            "slope": float("nan"),
            "rmse_m": float("nan"),
            "r2": float("nan"),
            "mae_m": float("nan"),
            "bias_m": float("nan"),
        }

    slope, intercept = np.polyfit(x, y, deg=1)
    fitted = intercept + slope * x
    residual = y - x
    ss_res = float(np.sum((y - fitted) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {
        "group": group,
        "buildings": int(x.size),
        "intercept": float(intercept),
        "slope": float(slope),
        "rmse_m": float(np.sqrt(np.mean(residual**2))),
        "r2": r2,
        "mae_m": float(np.mean(np.abs(residual))),
        "bias_m": float(np.mean(residual)),
        "lidar_min_m": float(np.min(x)),
        "lidar_max_m": float(np.max(x)),
        "pred_min_m": float(np.min(y)),
        "pred_max_m": float(np.max(y)),
    }


def sample_for_plot(rows: list[dict], max_points: int, seed: int) -> list[dict]:
    if len(rows) <= max_points:
        return rows
    rng = np.random.default_rng(seed)
    indexes = rng.choice(len(rows), size=max_points, replace=False)
    return [rows[int(i)] for i in indexes]


def plot_group(
    rows: list[dict],
    metrics: dict,
    title: str,
    output_path: Path,
    max_plot_points: int,
    seed: int,
) -> None:
    plot_rows = sample_for_plot(rows, max_plot_points, seed)
    x = np.array([float(row["lidar_height_m"]) for row in plot_rows], dtype=float)
    y = np.array([float(row["predicted_height_m"]) for row in plot_rows], dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    all_x = np.array([float(row["lidar_height_m"]) for row in rows], dtype=float)
    all_y = np.array([float(row["predicted_height_m"]) for row in rows], dtype=float)
    max_axis = float(np.nanpercentile(np.concatenate([all_x, all_y]), 99.5))
    max_axis = max(max_axis, 10.0)

    fig, ax = plt.subplots(figsize=(7.5, 6.5), constrained_layout=True)
    ax.scatter(x, y, s=10, alpha=0.35, edgecolors="none")
    ax.plot([0, max_axis], [0, max_axis], color="black", linewidth=1, linestyle="--", label="1:1")
    line_x = np.array([0, max_axis])
    line_y = metrics["intercept"] + metrics["slope"] * line_x
    ax.plot(line_x, line_y, color="#b2182b", linewidth=2, label="Best fit")
    ax.set_xlim(0, max_axis)
    ax.set_ylim(0, max_axis)
    ax.set_xlabel("LiDAR-derived building height (m)")
    ax.set_ylabel("Predicted building height (m)")
    ax.set_title(title)
    annotation = (
        f"y = {metrics['intercept']:.2f} + {metrics['slope']:.3f}x\n"
        f"RMSE = {metrics['rmse_m']:.2f} m\n"
        f"R2 = {metrics['r2']:.3f}\n"
        f"N = {metrics['buildings']:,}"
    )
    ax.text(
        0.04,
        0.96,
        annotation,
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox={"facecolor": "white", "edgecolor": "0.7", "alpha": 0.9},
    )
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    data_dir = (args.data_dir or run_dir / "mini_dataset").resolve()
    predictions_dir = (run_dir / args.predictions_subdir).resolve()
    output_dir = (
        args.output_dir
        or run_dir / "building_scatter_diagnostics" / args.predictions_subdir
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = read_csv(data_dir / "chips_manifest.csv")
    if args.split != "all":
        split_path = data_dir / f"{args.split}.txt"
        with split_path.open("r", encoding="utf-8") as f:
            split_ids = {line.strip() for line in f if line.strip()}
        manifest = [row for row in manifest if row["chip_id"] in split_ids]
    rows = []
    for manifest_row in manifest:
        rows.extend(
            component_rows_for_chip(
                row=manifest_row,
                data_dir=data_dir,
                predictions_dir=predictions_dir,
                height_stat=args.height_stat,
                min_component_pixels=args.min_component_pixels,
            )
        )
    if not rows:
        raise RuntimeError("No building components were available for diagnostics.")

    component_csv = output_dir / "building_component_predictions.csv"
    write_csv(component_csv, rows)

    metric_rows = [regression_metrics(rows, "all")]
    for city in sorted({row["source_city"] for row in rows}):
        city_rows = [row for row in rows if row["source_city"] == city]
        metric_rows.append(regression_metrics(city_rows, city))
    metrics_csv = output_dir / "building_scatter_metrics.csv"
    write_csv(metrics_csv, metric_rows)

    metric_by_group = {row["group"]: row for row in metric_rows}
    plot_group(
        rows=rows,
        metrics=metric_by_group["all"],
        title=f"Building Height Scatter | all | {args.predictions_subdir}",
        output_path=output_dir / "building_height_scatter_all.png",
        max_plot_points=args.max_plot_points,
        seed=args.seed,
    )
    for city in sorted({row["source_city"] for row in rows}):
        city_rows = [row for row in rows if row["source_city"] == city]
        plot_group(
            rows=city_rows,
            metrics=metric_by_group[city],
            title=f"Building Height Scatter | {city} | {args.predictions_subdir}",
            output_path=output_dir / f"building_height_scatter_{city}.png",
            max_plot_points=args.max_plot_points,
            seed=args.seed,
        )

    print(f"Component table: {component_csv}")
    print(f"Metrics table: {metrics_csv}")
    for row in metric_rows:
        print(
            f"{row['group']}: N={row['buildings']} "
            f"y={row['intercept']:.3f}+{row['slope']:.3f}x "
            f"RMSE={row['rmse_m']:.3f} R2={row['r2']:.3f}"
        )


if __name__ == "__main__":
    main()
