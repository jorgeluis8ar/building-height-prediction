#!/usr/bin/env python3
"""Compare LiDAR-derived building-height distributions across dataset splits."""

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
    / "nyc76_la95_offnadir_rgbnir_4ch_lowrise_binweighted_bg005_seed20260721_epoch50_guarded"
)
SPLITS = ("train", "val", "test")
SPLIT_LABELS = {"train": "Training", "val": "Validation", "test": "Test"}
SPLIT_COLORS = {"train": "#2878a0", "val": "#dc7f2a", "test": "#4b9560"}
CITIES = ("los_angeles", "new_york_city")
CITY_LABELS = {"los_angeles": "Los Angeles", "new_york_city": "New York City"}
HEIGHT_BIN_EDGES = np.asarray([0.0, 10.0, 20.0, 30.0, 40.0, 50.0, np.inf])
HEIGHT_BIN_LABELS = ("0-10", "10-20", "20-30", "30-40", "40-50", "50+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--display-max-m",
        type=float,
        default=150.0,
        help="Maximum height shown in histogram and ECDF panels.",
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"No building observations found in {path}")
    return rows


def values_for(rows: list[dict[str, str]], city: str | None = None) -> np.ndarray:
    values = np.asarray(
        [
            float(row["lidar_height_m"])
            for row in rows
            if city is None or row["source_city"] == city
        ],
        dtype=float,
    )
    values = values[np.isfinite(values) & (values >= 0)]
    if values.size == 0:
        raise ValueError(f"No valid LiDAR heights for city={city!r}")
    return values


def chip_count(rows: list[dict[str, str]], city: str | None = None) -> int:
    return len(
        {
            row["chip_id"]
            for row in rows
            if city is None or row["source_city"] == city
        }
    )


def bin_shares(values: np.ndarray) -> np.ndarray:
    counts, _ = np.histogram(values, bins=HEIGHT_BIN_EDGES)
    return counts / counts.sum()


def summarize(
    split: str,
    rows: list[dict[str, str]],
    city: str | None,
) -> dict[str, object]:
    values = values_for(rows, city)
    shares = bin_shares(values)
    result: dict[str, object] = {
        "sample": split,
        "group": city or "all",
        "chips": chip_count(rows, city),
        "buildings": int(values.size),
        "mean_m": float(np.mean(values)),
        "std_m": float(np.std(values)),
        "median_m": float(np.median(values)),
        "p75_m": float(np.percentile(values, 75)),
        "p90_m": float(np.percentile(values, 90)),
        "p95_m": float(np.percentile(values, 95)),
        "p99_m": float(np.percentile(values, 99)),
        "max_m": float(np.max(values)),
    }
    for label, share in zip(HEIGHT_BIN_LABELS, shares, strict=True):
        result[f"share_{label.replace('-', '_').replace('+', '_plus')}_pct"] = float(
            100 * share
        )
    return result


def ks_distance(x: np.ndarray, y: np.ndarray) -> float:
    pooled = np.sort(np.concatenate([x, y]))
    x_sorted = np.sort(x)
    y_sorted = np.sort(y)
    x_cdf = np.searchsorted(x_sorted, pooled, side="right") / x_sorted.size
    y_cdf = np.searchsorted(y_sorted, pooled, side="right") / y_sorted.size
    return float(np.max(np.abs(x_cdf - y_cdf)))


def quantile_distance_m(x: np.ndarray, y: np.ndarray) -> float:
    probabilities = np.linspace(0.0, 1.0, 1001)
    return float(np.mean(np.abs(np.quantile(x, probabilities) - np.quantile(y, probabilities))))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_histogram(
    ax: plt.Axes,
    split_rows: dict[str, list[dict[str, str]]],
    city: str | None,
    display_max_m: float,
) -> None:
    bins = np.linspace(0, display_max_m, 76)
    for split in SPLITS:
        values = values_for(split_rows[split], city)
        ax.hist(
            values,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=2,
            color=SPLIT_COLORS[split],
            label=SPLIT_LABELS[split],
        )
    ax.set_xlim(0, display_max_m)
    ax.set_xlabel("LiDAR-derived building height (m)")
    ax.set_ylabel("Density")
    ax.set_title("Normalized distribution")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right", frameon=False)


def plot_ecdf(
    ax: plt.Axes,
    split_rows: dict[str, list[dict[str, str]]],
    city: str | None,
    display_max_m: float,
) -> None:
    for split in SPLITS:
        values = np.sort(values_for(split_rows[split], city))
        cumulative = np.arange(1, values.size + 1) / values.size
        ax.plot(values, cumulative, linewidth=2, color=SPLIT_COLORS[split])
    ax.set_xlim(0, display_max_m)
    ax.set_ylim(0, 1.01)
    ax.set_xlabel("LiDAR-derived building height (m)")
    ax.set_ylabel("Cumulative share")
    ax.set_title("Empirical cumulative distribution")
    ax.grid(alpha=0.2)


def plot_bin_shares(
    ax: plt.Axes,
    split_rows: dict[str, list[dict[str, str]]],
    city: str | None,
) -> None:
    positions = np.arange(len(HEIGHT_BIN_LABELS))
    width = 0.25
    for index, split in enumerate(SPLITS):
        shares = 100 * bin_shares(values_for(split_rows[split], city))
        ax.bar(
            positions + (index - 1) * width,
            shares,
            width=width,
            color=SPLIT_COLORS[split],
            label=SPLIT_LABELS[split],
        )
    ax.set_xticks(positions, [f"{label} m" for label in HEIGHT_BIN_LABELS], rotation=25)
    ax.set_ylabel("Buildings (%)")
    ax.set_title("Height-bin composition")
    ax.grid(axis="y", alpha=0.2)


def create_overall_figure(
    split_rows: dict[str, list[dict[str, str]]],
    output_path: Path,
    display_max_m: float,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2), constrained_layout=True)
    plot_histogram(axes[0], split_rows, None, display_max_m)
    plot_ecdf(axes[1], split_rows, None, display_max_m)
    plot_bin_shares(axes[2], split_rows, None)
    fig.suptitle(
        "LiDAR-Derived Building-Height Distribution by Sample",
        fontsize=16,
        fontweight="bold",
    )
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def create_city_figure(
    split_rows: dict[str, list[dict[str, str]]],
    output_path: Path,
    display_max_m: float,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(17, 10), constrained_layout=True)
    for row_index, city in enumerate(CITIES):
        plot_histogram(axes[row_index, 0], split_rows, city, display_max_m)
        plot_ecdf(axes[row_index, 1], split_rows, city, display_max_m)
        plot_bin_shares(axes[row_index, 2], split_rows, city)
        axes[row_index, 0].text(
            -0.18,
            0.5,
            CITY_LABELS[city],
            transform=axes[row_index, 0].transAxes,
            rotation=90,
            va="center",
            ha="center",
            fontsize=13,
            fontweight="bold",
        )
    fig.suptitle(
        "LiDAR-Derived Height Distributions by Sample and City",
        fontsize=16,
        fontweight="bold",
    )
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    output_dir = (
        args.output_dir
        or run_dir / "evaluation" / f"lidar_height_split_distribution_epoch_{args.epoch:03d}"
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    split_rows = {
        split: read_rows(
            run_dir
            / "evaluation"
            / f"{split}_building_scatter_epoch_{args.epoch:03d}"
            / "building_component_predictions.csv"
        )
        for split in SPLITS
    }

    summary_rows = [
        summarize(split, split_rows[split], city)
        for city in (None, *CITIES)
        for split in SPLITS
    ]
    write_csv(output_dir / "lidar_height_distribution_summary.csv", summary_rows)

    distance_rows: list[dict[str, object]] = []
    for city in (None, *CITIES):
        train_values = values_for(split_rows["train"], city)
        for comparison_split in ("val", "test"):
            comparison_values = values_for(split_rows[comparison_split], city)
            distance_rows.append(
                {
                    "group": city or "all",
                    "reference_sample": "train",
                    "comparison_sample": comparison_split,
                    "ks_distance": ks_distance(train_values, comparison_values),
                    "mean_absolute_quantile_distance_m": quantile_distance_m(
                        train_values, comparison_values
                    ),
                }
            )
    write_csv(output_dir / "lidar_height_distribution_distances.csv", distance_rows)

    create_overall_figure(
        split_rows,
        output_dir / "lidar_height_distribution_train_val_test.png",
        args.display_max_m,
    )
    create_city_figure(
        split_rows,
        output_dir / "lidar_height_distribution_train_val_test_by_city.png",
        args.display_max_m,
    )
    print(f"Wrote LiDAR split-distribution diagnostics to {output_dir}")


if __name__ == "__main__":
    main()
