#!/usr/bin/env python3
"""Compare building-level height predictions across off-nadir HTC models."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3]
RUNS_ROOT = (
    REPO_ROOT
    / "data_source/data/ml_models/generated/htc_dc_net/mini_training_runs"
)
DEFAULT_MODELS = {
    "RGB": "nyc76_la95_offnadir_3ch_lowrise_binweighted_bg005_seed20260720_epoch50_guarded",
    "RGB + mask": "nyc76_la95_offnadir_rgbmask_4ch_lowrise_binweighted_bg005_seed20260721_epoch50_guarded",
    "RGB + NIR": "nyc76_la95_offnadir_rgbnir_4ch_lowrise_binweighted_bg005_seed20260721_epoch50_guarded",
}
SPLITS = ("train", "val", "test")
SPLIT_LABELS = {"train": "Training", "val": "Validation", "test": "Test"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=RUNS_ROOT)
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-points", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--zoom-max-m", type=float, default=50.0)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"No building rows found in {path}")
    return rows


def arrays(rows: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray([float(row["lidar_height_m"]) for row in rows])
    y = np.asarray([float(row["predicted_height_m"]) for row in rows])
    valid = np.isfinite(x) & np.isfinite(y)
    return x[valid], y[valid]


def metrics(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    slope, intercept = np.polyfit(x, y, deg=1)
    fitted = intercept + slope * x
    residual = y - x
    ss_res = float(np.sum((y - fitted) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return {
        "n": int(x.size),
        "intercept": float(intercept),
        "slope": float(slope),
        "rmse_m": float(np.sqrt(np.mean(residual**2))),
        "mae_m": float(np.mean(np.abs(residual))),
        "bias_m": float(np.mean(residual)),
        "r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
    }


def sample_indexes(size: int, max_points: int, seed: int) -> np.ndarray:
    if size <= max_points:
        return np.arange(size)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(size, size=max_points, replace=False))


def write_metrics(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def plot_comparison(
    data: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]],
    metric_lookup: dict[tuple[str, str], dict[str, float]],
    output_path: Path,
    axis_max: float,
    max_points: int,
    seed: int,
    zoomed: bool,
) -> None:
    model_labels = list(DEFAULT_MODELS)
    fig, axes = plt.subplots(
        len(SPLITS),
        len(model_labels),
        figsize=(15.5, 14.5),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    colors = {"RGB": "#3b82a0", "RGB + mask": "#c06c3e", "RGB + NIR": "#438a5e"}

    for row_index, split in enumerate(SPLITS):
        for column_index, model_label in enumerate(model_labels):
            ax = axes[row_index, column_index]
            x, y = data[(split, model_label)]
            idx = sample_indexes(x.size, max_points, seed + row_index)
            ax.scatter(
                x[idx],
                y[idx],
                s=7,
                alpha=0.18,
                color=colors[model_label],
                edgecolors="none",
                rasterized=True,
            )
            ax.plot([0, axis_max], [0, axis_max], "--", color="0.2", linewidth=1.1)
            panel_metrics = metric_lookup[(split, model_label)]
            line_x = np.asarray([0.0, axis_max])
            line_y = panel_metrics["intercept"] + panel_metrics["slope"] * line_x
            ax.plot(line_x, line_y, color="#b2182b", linewidth=1.8)
            ax.set_xlim(0, axis_max)
            ax.set_ylim(0, axis_max)
            ax.grid(alpha=0.18)
            if row_index == 0:
                ax.set_title(model_label, fontsize=13, fontweight="bold")
            if column_index == 0:
                ax.set_ylabel(
                    f"{SPLIT_LABELS[split]}\nPredicted height (m)",
                    fontsize=11,
                )
            if row_index == len(SPLITS) - 1:
                ax.set_xlabel("LiDAR-derived height (m)", fontsize=11)
            annotation = (
                f"y = {panel_metrics['intercept']:.2f} + {panel_metrics['slope']:.3f}x\n"
                f"RMSE = {panel_metrics['rmse_m']:.2f} m\n"
                f"R2 = {panel_metrics['r2']:.3f}\n"
                f"Bias = {panel_metrics['bias_m']:+.2f} m\n"
                f"N = {panel_metrics['n']:,}"
            )
            ax.text(
                0.03,
                0.97,
                annotation,
                transform=ax.transAxes,
                va="top",
                fontsize=9,
                bbox={"facecolor": "white", "edgecolor": "0.75", "alpha": 0.88},
            )

    range_label = f"0-{axis_max:g} m detail" if zoomed else "full distribution"
    fig.suptitle(
        f"Off-Nadir HTC-DC Net Building-Height Comparison ({range_label})",
        fontsize=17,
        fontweight="bold",
    )
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    runs_root = args.runs_root.resolve()
    output_dir = (
        args.output_dir
        or runs_root / f"offnadir_model_scatter_comparison_epoch_{args.epoch:03d}"
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    data: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    metric_lookup: dict[tuple[str, str], dict[str, float]] = {}
    metric_records: list[dict[str, object]] = []
    all_values: list[np.ndarray] = []

    for model_label, run_name in DEFAULT_MODELS.items():
        for split in SPLITS:
            table = (
                runs_root
                / run_name
                / "evaluation"
                / f"{split}_building_scatter_epoch_{args.epoch:03d}"
                / "building_component_predictions.csv"
            )
            x, y = arrays(read_rows(table))
            panel_metrics = metrics(x, y)
            data[(split, model_label)] = (x, y)
            metric_lookup[(split, model_label)] = panel_metrics
            all_values.extend([x, y])
            metric_records.append(
                {
                    "sample": split,
                    "model": model_label,
                    **panel_metrics,
                    "source_table": table.relative_to(REPO_ROOT),
                }
            )

    full_axis_max = float(np.ceil(np.nanmax(np.concatenate(all_values)) / 25.0) * 25.0)
    write_metrics(output_dir / "offnadir_model_scatter_metrics.csv", metric_records)
    plot_comparison(
        data=data,
        metric_lookup=metric_lookup,
        output_path=output_dir / "offnadir_model_scatter_comparison_full.png",
        axis_max=full_axis_max,
        max_points=args.max_points,
        seed=args.seed,
        zoomed=False,
    )
    plot_comparison(
        data=data,
        metric_lookup=metric_lookup,
        output_path=output_dir / "offnadir_model_scatter_comparison_0_50m.png",
        axis_max=args.zoom_max_m,
        max_points=args.max_points,
        seed=args.seed,
        zoomed=True,
    )
    print(f"Wrote comparison diagnostics to {output_dir}")


if __name__ == "__main__":
    main()
