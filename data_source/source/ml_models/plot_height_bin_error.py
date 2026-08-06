#!/usr/bin/env python3
"""Plot building-level prediction errors by LiDAR height bin.

This diagnostic starts from the building-level component table produced by
`plot_building_height_scatter.py`. Each row is one connected building-mask
component inside one chip. We bin buildings by LiDAR-derived height and report
how model error changes across the height distribution.
"""

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
DEFAULT_RUN_DIR = (
    REPO_ROOT
    / "data_source/data/ml_models/generated/htc_dc_net/mini_training_runs/"
    / "nyc76_la95_12ch_lowrise_binweighted_bg005_seed20260715_epoch50_guarded"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument(
        "--component-csv",
        type=Path,
        default=None,
        help="Building component prediction CSV. Defaults to the training scatter diagnostics for --run-dir.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--label", default="train")
    parser.add_argument(
        "--bin-edges",
        default="0,10,20,50",
        help="Comma-separated lower/inner height bin edges in meters. The final bin is open-ended.",
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_edges(value: str) -> list[float]:
    edges = [float(item.strip()) for item in value.split(",") if item.strip()]
    if len(edges) < 2:
        raise ValueError("--bin-edges must contain at least two values.")
    if edges != sorted(edges):
        raise ValueError("--bin-edges must be sorted ascending.")
    return edges


def bin_label(lower: float, upper: float | None) -> str:
    if upper is None:
        return f">{lower:g}m"
    return f"{lower:g}-{upper:g}m"


def assign_bin(height: float, edges: list[float]) -> tuple[str, int] | None:
    if not np.isfinite(height):
        return None
    for index in range(len(edges) - 1):
        lower = edges[index]
        upper = edges[index + 1]
        if lower <= height < upper:
            return bin_label(lower, upper), index
    if height >= edges[-1]:
        return bin_label(edges[-1], None), len(edges) - 1
    return None


def summarize_group(rows: list[dict], group: str, edges: list[float]) -> list[dict]:
    groups: dict[tuple[str, int], list[float]] = {}
    for row in rows:
        height = float(row["lidar_height_m"])
        predicted = float(row["predicted_height_m"])
        if not np.isfinite(height) or not np.isfinite(predicted):
            continue
        assigned = assign_bin(height, edges)
        if assigned is None:
            continue
        label, order = assigned
        groups.setdefault((label, order), []).append(predicted - height)

    output = []
    for (label, order), errors in sorted(groups.items(), key=lambda item: item[0][1]):
        error_arr = np.array(errors, dtype=float)
        output.append(
            {
                "group": group,
                "height_bin": label,
                "bin_order": order,
                "buildings": int(error_arr.size),
                "rmse_m": float(np.sqrt(np.mean(error_arr**2))),
                "mae_m": float(np.mean(np.abs(error_arr))),
                "bias_m": float(np.mean(error_arr)),
                "median_error_m": float(np.median(error_arr)),
                "p25_error_m": float(np.percentile(error_arr, 25)),
                "p75_error_m": float(np.percentile(error_arr, 75)),
            }
        )
    return output


def plot_metric(rows: list[dict], output_path: Path, label: str) -> None:
    city_order = ["all", "los_angeles", "new_york_city"]
    city_labels = {
        "all": "All",
        "los_angeles": "Los Angeles",
        "new_york_city": "New York City",
    }
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True, constrained_layout=True)
    fig.suptitle(f"Building-Level RMSE By LiDAR Height Bin | {label}", fontsize=15)

    ymax = max(float(row["rmse_m"]) for row in rows) * 1.15 if rows else 1
    for ax, group in zip(axes, city_order):
        group_rows = [row for row in rows if row["group"] == group]
        x = [row["height_bin"] for row in group_rows]
        y = [float(row["rmse_m"]) for row in group_rows]
        n = [int(row["buildings"]) for row in group_rows]
        bars = ax.bar(x, y, color="#4c78a8")
        ax.set_title(city_labels[group])
        ax.set_xlabel("LiDAR height bin")
        ax.set_ylim(0, ymax)
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=30)
        for bar, count in zip(bars, n):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"n={count:,}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90,
            )
    axes[0].set_ylabel("RMSE (m)")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_bias(rows: list[dict], output_path: Path, label: str) -> None:
    city_order = ["all", "los_angeles", "new_york_city"]
    city_labels = {
        "all": "All",
        "los_angeles": "Los Angeles",
        "new_york_city": "New York City",
    }
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True, constrained_layout=True)
    fig.suptitle(f"Building-Level Bias By LiDAR Height Bin | {label}", fontsize=15)
    max_abs = max(abs(float(row["bias_m"])) for row in rows) * 1.2 if rows else 1
    max_abs = max(max_abs, 1)

    for ax, group in zip(axes, city_order):
        group_rows = [row for row in rows if row["group"] == group]
        x = [row["height_bin"] for row in group_rows]
        y = [float(row["bias_m"]) for row in group_rows]
        colors = ["#b2182b" if value > 0 else "#2166ac" for value in y]
        ax.bar(x, y, color=colors)
        ax.axhline(0, color="black", linewidth=1)
        ax.set_title(city_labels[group])
        ax.set_xlabel("LiDAR height bin")
        ax.set_ylim(-max_abs, max_abs)
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=30)
    axes[0].set_ylabel("Mean prediction error (m)")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_residual_distribution(
    component_rows: list[dict],
    edges: list[float],
    output_path: Path,
    label: str,
) -> None:
    """Plot residual distributions by LiDAR height bin.

    Residual is predicted height minus LiDAR-derived target height. Positive
    values mean the model over-predicts height; negative values mean it
    under-predicts height.
    """
    bins = [(bin_label(edges[index], edges[index + 1]), index) for index in range(len(edges) - 1)]
    bins.append((bin_label(edges[-1], None), len(edges) - 1))
    errors_by_bin = {name: [] for name, _ in bins}

    for row in component_rows:
        height = float(row["lidar_height_m"])
        predicted = float(row["predicted_height_m"])
        assigned = assign_bin(height, edges)
        if assigned is None or not np.isfinite(predicted):
            continue
        bin_name, _ = assigned
        errors_by_bin[bin_name].append(predicted - height)

    labels = [name for name, _ in bins]
    data = [np.array(errors_by_bin[name], dtype=float) for name in labels]
    data = [values[np.isfinite(values)] for values in data]

    fig, ax = plt.subplots(figsize=(8.5, 6.5), constrained_layout=True)
    ax.boxplot(
        data,
        tick_labels=labels,
        showfliers=True,
        whis=1.5,
        patch_artist=False,
        medianprops={"color": "#ff7f0e", "linewidth": 1.5},
        flierprops={
            "marker": "o",
            "markerfacecolor": "none",
            "markeredgecolor": "black",
            "markersize": 4,
            "linestyle": "none",
        },
    )
    ax.axhline(0, color="0.4", linewidth=1, linestyle="--")
    ax.set_title(
        f"{label} data sample\nResidual = Predicted Height - LiDAR height",
        fontsize=16,
        fontweight="bold",
    )
    ax.set_xlabel("LiDAR height bin")
    ax.set_ylabel("Residual (m)")
    ax.grid(axis="y", alpha=0.2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    component_csv = (
        args.component_csv
        or run_dir
        / "evaluation/train_building_scatter_epoch_050/building_component_predictions.csv"
    ).resolve()
    if not component_csv.exists():
        raise FileNotFoundError(component_csv)
    output_dir = (
        args.output_dir or component_csv.parent / "height_bin_error"
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_rows(component_csv)
    edges = parse_edges(args.bin_edges)
    output_rows = []
    output_rows.extend(summarize_group(rows, "all", edges))
    for city in sorted({row["source_city"] for row in rows}):
        city_rows = [row for row in rows if row["source_city"] == city]
        output_rows.extend(summarize_group(city_rows, city, edges))

    summary_csv = output_dir / f"height_bin_error_summary_{args.label}.csv"
    rmse_png = output_dir / f"height_bin_rmse_{args.label}.png"
    bias_png = output_dir / f"height_bin_bias_{args.label}.png"
    residual_boxplot_png = output_dir / f"height_bin_residual_boxplot_{args.label}.png"
    write_csv(summary_csv, output_rows)
    plot_metric(output_rows, rmse_png, args.label)
    plot_bias(output_rows, bias_png, args.label)
    plot_residual_distribution(rows, edges, residual_boxplot_png, args.label)
    print(f"Summary: {summary_csv}")
    print(f"RMSE plot: {rmse_png}")
    print(f"Bias plot: {bias_png}")
    print(f"Residual boxplot: {residual_boxplot_png}")


if __name__ == "__main__":
    main()
