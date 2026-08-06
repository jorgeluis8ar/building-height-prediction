#!/usr/bin/env python3
"""Evaluate HTC-DC Net epoch prediction summaries and visual QA panels."""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from skimage import io


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3]
DEFAULT_RUN_DIR = (
    REPO_ROOT
    / "data_source/data/ml_models/generated/htc_dc_net/mini_training_runs/"
    / "nyc71_la100_seed20260707_epoch5_repo_params_guarded"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-visual-chips-per-city", type=int, default=4)
    parser.add_argument("--visual-epochs", default="1,5")
    return parser.parse_args()


def epoch_from_path(path: Path) -> int:
    match = re.search(r"predictions_summary_epoch_(\d+)\.csv$", path.name)
    if not match:
        raise ValueError(f"Cannot parse epoch from {path}")
    return int(match.group(1))


def read_csv(path: Path) -> list[dict]:
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


def fnum(value: str) -> float:
    try:
        return float(value)
    except ValueError:
        return float("nan")


def summarize_epochs(run_dir: Path) -> tuple[list[dict], dict[int, list[dict]]]:
    summary_files = sorted(run_dir.glob("predictions_summary_epoch_*.csv"), key=epoch_from_path)
    if not summary_files:
        raise FileNotFoundError(f"No predictions_summary_epoch_*.csv files found in {run_dir}")

    rows_by_epoch: dict[int, list[dict]] = {}
    city_summary = []
    numeric_cols = [
        "mae_m",
        "rmse_m",
        "bias_m",
        "pred_mean_m",
        "pred_std_m",
        "target_mean_m",
        "target_std_m",
    ]

    for path in summary_files:
        epoch = epoch_from_path(path)
        rows = read_csv(path)
        rows_by_epoch[epoch] = rows
        for city in sorted({row["source_city"] for row in rows}):
            city_rows = [row for row in rows if row["source_city"] == city]
            out = {
                "epoch": epoch,
                "source_city": city,
                "chips": len(city_rows),
                "collapsed_chips": sum(row["collapse_flag"] == "True" for row in city_rows),
            }
            out["collapsed_share"] = out["collapsed_chips"] / out["chips"] if out["chips"] else float("nan")
            for col in numeric_cols:
                vals = np.array([fnum(row[col]) for row in city_rows], dtype=float)
                vals = vals[np.isfinite(vals)]
                out[f"mean_{col}"] = float(vals.mean()) if vals.size else float("nan")
                out[f"median_{col}"] = float(np.median(vals)) if vals.size else float("nan")
            city_summary.append(out)

    return city_summary, rows_by_epoch


def plot_epoch_metrics(city_summary: list[dict], output_dir: Path) -> None:
    cities = sorted({row["source_city"] for row in city_summary})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    plots = [
        ("mean_mae_m", "Mean MAE (m)"),
        ("mean_rmse_m", "Mean RMSE (m)"),
        ("mean_bias_m", "Mean bias (m)"),
        ("mean_pred_std_m", "Mean prediction std (m)"),
    ]
    for ax, (col, ylabel) in zip(axes.ravel(), plots):
        for city in cities:
            rows = [row for row in city_summary if row["source_city"] == city]
            rows.sort(key=lambda row: int(row["epoch"]))
            ax.plot([int(row["epoch"]) for row in rows], [float(row[col]) for row in rows], marker="o", label=city)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
    axes[0, 0].legend()
    fig.suptitle("HTC-DC Net Epoch Metrics By City")
    fig.savefig(output_dir / "epoch_metric_trends_by_city.png", dpi=180)
    plt.close(fig)


def plot_collapse(city_summary: list[dict], output_dir: Path) -> None:
    cities = sorted({row["source_city"] for row in city_summary})
    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    for city in cities:
        rows = [row for row in city_summary if row["source_city"] == city]
        rows.sort(key=lambda row: int(row["epoch"]))
        ax.plot(
            [int(row["epoch"]) for row in rows],
            [100 * float(row["collapsed_share"]) for row in rows],
            marker="o",
            label=city,
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Collapsed chips (%)")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title("Prediction Collapse By Epoch")
    fig.savefig(output_dir / "collapse_share_by_epoch.png", dpi=180)
    plt.close(fig)


def choose_visual_chips(rows_by_epoch: dict[int, list[dict]], max_per_city: int) -> list[dict]:
    final_epoch = max(rows_by_epoch)
    rows = rows_by_epoch[final_epoch]
    selected = []
    for city in sorted({row["source_city"] for row in rows}):
        city_rows = [row for row in rows if row["source_city"] == city]
        noncollapsed = [row for row in city_rows if row["collapse_flag"] == "False"]
        candidates = [
            ("highest_target_mean", max(city_rows, key=lambda row: fnum(row["target_mean_m"]))),
            ("highest_mae", max(city_rows, key=lambda row: fnum(row["mae_m"]))),
            (
                "highest_pred_std_noncollapsed",
                max(noncollapsed or city_rows, key=lambda row: fnum(row["pred_std_m"])),
            ),
            (
                "median_mae_noncollapsed",
                median_row(noncollapsed or city_rows, "mae_m"),
            ),
        ]
        seen = set()
        for reason, row in candidates:
            if row["chip_id"] in seen:
                continue
            seen.add(row["chip_id"])
            selected.append(
                {
                    "chip_id": row["chip_id"],
                    "source_city": city,
                    "selection_reason": reason,
                    "final_epoch": final_epoch,
                    "final_mae_m": row["mae_m"],
                    "final_rmse_m": row["rmse_m"],
                    "final_pred_std_m": row["pred_std_m"],
                    "final_target_mean_m": row["target_mean_m"],
                    "final_collapse_flag": row["collapse_flag"],
                }
            )
            if len([item for item in selected if item["source_city"] == city]) >= max_per_city:
                break
    return selected


def median_row(rows: list[dict], col: str) -> dict:
    rows = sorted(rows, key=lambda row: fnum(row[col]))
    return rows[len(rows) // 2]


def row_lookup(rows_by_epoch: dict[int, list[dict]]) -> dict[tuple[int, str], dict]:
    lookup = {}
    for epoch, rows in rows_by_epoch.items():
        for row in rows:
            lookup[(epoch, row["chip_id"])] = row
    return lookup


def load_chip_arrays(run_dir: Path, chip_id: str, pred_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    target_path = run_dir / "mini_dataset" / "ndsm" / f"{chip_id}_AGL.tif"
    mask_path = run_dir / "mini_dataset" / "mask" / f"{chip_id}_BLG.tif"
    pred = io.imread(pred_path).astype(np.float32)
    target = np.nan_to_num(io.imread(target_path).astype(np.float32)).clip(0)
    mask = io.imread(mask_path).astype(np.float32) > 0
    return pred, target, mask


def plot_visual_panel(
    run_dir: Path,
    output_dir: Path,
    selected: list[dict],
    rows_by_epoch: dict[int, list[dict]],
    visual_epochs: list[int],
) -> None:
    lookup = row_lookup(rows_by_epoch)
    visual_dir = output_dir / "visual_qa"
    visual_dir.mkdir(parents=True, exist_ok=True)

    for item in selected:
        chip_id = item["chip_id"]
        for epoch in visual_epochs:
            row = lookup.get((epoch, chip_id))
            if row is None:
                continue
            pred_path = REPO_ROOT / row["prediction_path"]
            pred, target, mask = load_chip_arrays(run_dir, chip_id, pred_path)
            err = np.where(mask & (target > 0), pred - target, np.nan)
            target_vals = target[mask & (target > 0)]
            pred_vals = pred[mask & (target > 0)]
            if target_vals.size and pred_vals.size:
                vmax = float(np.nanpercentile(np.concatenate([target_vals, pred_vals]), 99))
            else:
                vmax = 20.0
            vmax = max(vmax, 10.0)
            err_lim = float(np.nanpercentile(np.abs(err), 95)) if np.isfinite(err).any() else 10.0
            err_lim = max(err_lim, 5.0)

            fig, axes = plt.subplots(1, 4, figsize=(15, 4.2), constrained_layout=True)
            for ax in axes:
                ax.set_xticks([])
                ax.set_yticks([])

            im0 = axes[0].imshow(pred, cmap="viridis", vmin=0, vmax=vmax)
            axes[0].set_title("Prediction")
            fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.02, label="m")

            im1 = axes[1].imshow(target, cmap="viridis", vmin=0, vmax=vmax)
            axes[1].set_title("Target AGL")
            fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.02, label="m")

            axes[2].imshow(mask, cmap="gray")
            axes[2].set_title("Building mask")

            im3 = axes[3].imshow(err, cmap="coolwarm", vmin=-err_lim, vmax=err_lim)
            axes[3].set_title("Prediction - target")
            fig.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.02, label="m")

            fig.suptitle(
                f"{chip_id} | epoch {epoch} | {item['selection_reason']} | "
                f"MAE={fnum(row['mae_m']):.2f} m | RMSE={fnum(row['rmse_m']):.2f} m | "
                f"pred_std={fnum(row['pred_std_m']):.2f} m | collapsed={row['collapse_flag']}",
                fontsize=10,
            )
            out = visual_dir / f"epoch_{epoch:03d}_{chip_id}_{item['selection_reason']}.png"
            fig.savefig(out, dpi=170)
            plt.close(fig)


def parse_epochs(value: str, available_epochs: list[int]) -> list[int]:
    requested = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        requested.append(int(part))
    return [epoch for epoch in requested if epoch in available_epochs]


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    output_dir = (args.output_dir or run_dir / "evaluation").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    city_summary, rows_by_epoch = summarize_epochs(run_dir)
    write_csv(output_dir / "epoch_city_summary.csv", city_summary)
    plot_epoch_metrics(city_summary, output_dir)
    plot_collapse(city_summary, output_dir)

    selected = choose_visual_chips(rows_by_epoch, args.max_visual_chips_per_city)
    write_csv(output_dir / "selected_visual_qa_chips.csv", selected)
    visual_epochs = parse_epochs(args.visual_epochs, sorted(rows_by_epoch))
    plot_visual_panel(run_dir, output_dir, selected, rows_by_epoch, visual_epochs)

    print(f"Evaluation written to: {output_dir}")
    print(f"Epochs evaluated: {sorted(rows_by_epoch)}")
    print(f"Visual QA chips: {len(selected)}")


if __name__ == "__main__":
    main()
