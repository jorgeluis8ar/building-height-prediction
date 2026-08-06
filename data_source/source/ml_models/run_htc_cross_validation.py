#!/usr/bin/env python3
"""Run staged HTC-DC Net cross-validation experiments.

This script is a thin orchestration layer. It intentionally reuses the
project's existing training and checkpoint-evaluation scripts so that CV runs
produce the same artifacts as the single-run experiments.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3]
DEFAULT_DATASET_DIR = REPO_ROOT / "data_source/data/ml_models/generated/htc_dc_net/nyc_la_12ch_v1"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data_source/data/ml_models/generated/htc_dc_net/cross_validation/nyc_la_12ch_v1"
MINI_RUN_ROOT = REPO_ROOT / "data_source/data/ml_models/generated/htc_dc_net/mini_training_runs"
TRAIN_SCRIPT = REPO_ROOT / "data_source/source/ml_models/run_htc_mini_training.py"
EVAL_SCRIPT = REPO_ROOT / "data_source/source/ml_models/evaluate_htc_checkpoint_on_split.py"


BASE_CONFIG = {
    "epochs": 50,
    "lr": 0.00003,
    "batch_size": 8,
    "num_workers": 0,
    "patience": 20,
    "save_predictions_every": 10,
    "save_checkpoints_every": 10,
    "height_loss_weighting": "bins",
    "height_bin_edges": "3,6,10,25,50",
    "height_bin_weights": "4,3,2,1,3,8",
    "background_loss_weight": 0.05,
    "collapse_std_threshold": 0.05,
    "collapse_min_share": 0.8,
    "collapse_patience": 1,
    "seed": 20260718,
}


PARAMETER_GRID = {
    "baseline_current": {},
    "lower_lr": {"lr": 0.00001},
    "higher_lr": {"lr": 0.0001},
    "lower_bg": {"background_loss_weight": 0.02},
    "higher_bg": {"background_loss_weight": 0.10},
    "stronger_lowrise": {"height_bin_weights": "5,4,2,1,3,8"},
    "stronger_highrise": {"height_bin_weights": "4,3,2,1,4,10"},
    "balanced_bins": {"height_bin_weights": "3,3,2,1,3,6"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--folds-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--configs", default="all", help="Comma-separated config names or 'all'.")
    parser.add_argument("--folds", default="all", help="Comma-separated fold names such as fold_01 or 'all'.")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs, useful for smoke tests.")
    parser.add_argument("--save-every", type=int, default=None, help="Override checkpoint/prediction cadence.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def read_ids(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


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


def selected_configs(config_arg: str) -> list[str]:
    if config_arg == "all":
        return list(PARAMETER_GRID.keys())
    names = [name.strip() for name in config_arg.split(",") if name.strip()]
    missing = [name for name in names if name not in PARAMETER_GRID]
    if missing:
        raise RuntimeError(f"Unknown config names: {missing}")
    return names


def selected_folds(folds_dir: Path, folds_arg: str) -> list[str]:
    available = sorted(path.name for path in folds_dir.glob("fold_*") if path.is_dir())
    if not available:
        raise RuntimeError(f"No fold directories found in {folds_dir}")
    if folds_arg == "all":
        return available
    names = [name.strip() for name in folds_arg.split(",") if name.strip()]
    missing = [name for name in names if name not in available]
    if missing:
        raise RuntimeError(f"Unknown fold names: {missing}")
    return names


def merged_config(name: str, args: argparse.Namespace) -> dict:
    config = dict(BASE_CONFIG)
    config.update(PARAMETER_GRID[name])
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.save_every is not None:
        config["save_predictions_every"] = args.save_every
        config["save_checkpoints_every"] = args.save_every
    if args.seed is not None:
        config["seed"] = args.seed
    return config


def train_counts_by_city(fold_dir: Path) -> dict[str, int]:
    train_ids = set(read_ids(fold_dir / "train.txt"))
    rows = read_csv(fold_dir / "chips_manifest.csv")
    counts = defaultdict(int)
    for row in rows:
        if row["chip_id"] in train_ids:
            counts[row["source_city"]] += 1
    return counts


def run_command(command: list[str], cwd: Path) -> None:
    print("Running:", " ".join(command))
    subprocess.run(command, cwd=cwd, check=True)


def summarize_prediction_csv(path: Path, config_name: str, fold_name: str, run_name: str) -> list[dict]:
    rows = read_csv(path)
    groups = [("all", rows)]
    for city in sorted({row["source_city"] for row in rows}):
        groups.append((city, [row for row in rows if row["source_city"] == city]))

    summaries = []
    for group_name, group_rows in groups:
        rmse_values = np.array([float(row["rmse_m"]) for row in group_rows], dtype=float)
        mae_values = np.array([float(row["mae_m"]) for row in group_rows], dtype=float)
        bias_values = np.array([float(row["bias_m"]) for row in group_rows], dtype=float)
        pred_std_values = np.array([float(row["pred_std_m"]) for row in group_rows], dtype=float)
        collapse_flags = [str(row["collapse_flag"]).lower() in {"true", "1"} for row in group_rows]
        summaries.append(
            {
                "config_name": config_name,
                "fold": fold_name,
                "run_name": run_name,
                "group": group_name,
                "validation_chips": len(group_rows),
                "mean_val_rmse_m": float(np.nanmean(rmse_values)),
                "median_val_rmse_m": float(np.nanmedian(rmse_values)),
                "mean_val_mae_m": float(np.nanmean(mae_values)),
                "mean_val_bias_m": float(np.nanmean(bias_values)),
                "mean_prediction_std_m": float(np.nanmean(pred_std_values)),
                "collapsed_chips": int(sum(collapse_flags)),
            }
        )
    return summaries


def aggregate_config_results(fold_rows: list[dict], effective_configs: dict[str, dict]) -> list[dict]:
    config_names = sorted({row["config_name"] for row in fold_rows})
    results = []
    for config_name in config_names:
        rows = [row for row in fold_rows if row["config_name"] == config_name]
        overall = [row for row in rows if row["group"] == "all"]
        la = [row for row in rows if row["group"] == "los_angeles"]
        nyc = [row for row in rows if row["group"] == "new_york_city"]
        result = {
            "config_name": config_name,
            "folds_completed": len(overall),
            "mean_val_rmse_m": float(np.mean([float(row["mean_val_rmse_m"]) for row in overall])),
            "median_val_rmse_m": float(np.median([float(row["median_val_rmse_m"]) for row in overall])),
            "mean_val_mae_m": float(np.mean([float(row["mean_val_mae_m"]) for row in overall])),
            "mean_val_bias_m": float(np.mean([float(row["mean_val_bias_m"]) for row in overall])),
            "mean_prediction_std_m": float(np.mean([float(row["mean_prediction_std_m"]) for row in overall])),
            "collapsed_chips": int(sum(int(row["collapsed_chips"]) for row in overall)),
            "la_mean_val_rmse_m": float(np.mean([float(row["mean_val_rmse_m"]) for row in la])) if la else float("nan"),
            "nyc_mean_val_rmse_m": float(np.mean([float(row["mean_val_rmse_m"]) for row in nyc])) if nyc else float("nan"),
        }
        result.update({key: value for key, value in effective_configs[config_name].items()})
        results.append(result)
    return results


def plot_config_results(rows: list[dict], output_path: Path) -> None:
    ranked = sorted(rows, key=lambda row: float(row["mean_val_rmse_m"]))
    labels = [row["config_name"] for row in ranked]
    values = [float(row["mean_val_rmse_m"]) for row in ranked]
    fig, ax = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
    ax.bar(labels, values)
    ax.set_ylabel("Mean validation-chip RMSE (m)")
    ax.set_title("HTC-DC Net Cross-Validation RMSE By Config")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    folds_dir = (args.folds_dir or dataset_dir / "cv_folds").resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config_names = selected_configs(args.configs)
    fold_names = selected_folds(folds_dir, args.folds)
    effective_configs = {config_name: merged_config(config_name, args) for config_name in config_names}
    all_fold_rows = []

    with (output_dir / "cv_run_request.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "dataset_dir": str(dataset_dir.relative_to(REPO_ROOT)),
                "folds_dir": str(folds_dir.relative_to(REPO_ROOT)),
                "configs": config_names,
                "effective_configs": effective_configs,
                "folds": fold_names,
            },
            f,
            indent=2,
        )

    for config_name in config_names:
        config = effective_configs[config_name]
        for fold_name in fold_names:
            fold_dir = folds_dir / fold_name
            counts = train_counts_by_city(fold_dir)
            run_name = f"cv_12ch_{config_name}_{fold_name}_seed{config['seed']}_epoch{config['epochs']:03d}"
            run_dir = MINI_RUN_ROOT / run_name
            if run_dir.exists() and args.overwrite:
                shutil.rmtree(run_dir)
            if run_dir.exists() and args.skip_existing:
                print(f"Skipping existing run: {run_dir}")
            else:
                command = [
                    sys.executable,
                    str(TRAIN_SCRIPT.relative_to(REPO_ROOT)),
                    "--dataset-dir",
                    str(fold_dir.relative_to(REPO_ROOT)),
                    "--in-channels",
                    "12",
                    "--nyc-chips",
                    str(counts["new_york_city"]),
                    "--la-chips",
                    str(counts["los_angeles"]),
                    "--epochs",
                    str(config["epochs"]),
                    "--seed",
                    str(config["seed"]),
                    "--lr",
                    str(config["lr"]),
                    "--batch-size",
                    str(config["batch_size"]),
                    "--num-workers",
                    str(config["num_workers"]),
                    "--patience",
                    str(config["patience"]),
                    "--save-predictions-every",
                    str(config["save_predictions_every"]),
                    "--save-checkpoints-every",
                    str(config["save_checkpoints_every"]),
                    "--height-loss-weighting",
                    config["height_loss_weighting"],
                    "--height-bin-edges",
                    config["height_bin_edges"],
                    "--height-bin-weights",
                    config["height_bin_weights"],
                    "--background-loss-weight",
                    str(config["background_loss_weight"]),
                    "--collapse-std-threshold",
                    str(config["collapse_std_threshold"]),
                    "--collapse-min-share",
                    str(config["collapse_min_share"]),
                    "--collapse-patience",
                    str(config["collapse_patience"]),
                    "--stop-on-collapse",
                    "--run-name",
                    run_name,
                ]
                run_command(command, REPO_ROOT)

            checkpoint = run_dir / f"model_epoch_{config['epochs']:03d}.pth"
            if not checkpoint.exists():
                checkpoint = run_dir / "model_last.pth"
            if not checkpoint.exists():
                raise FileNotFoundError(checkpoint)

            eval_label = f"epoch_{config['epochs']:03d}"
            eval_command = [
                sys.executable,
                str(EVAL_SCRIPT.relative_to(REPO_ROOT)),
                "--run-dir",
                str(run_dir.relative_to(REPO_ROOT)),
                "--dataset-dir",
                str(fold_dir.relative_to(REPO_ROOT)),
                "--checkpoint",
                str(checkpoint.relative_to(REPO_ROOT)),
                "--split",
                "val",
                "--output-subdir",
                f"cv_val_predictions_{eval_label}",
                "--output-label",
                f"cv_val_{eval_label}",
                "--device",
                args.device,
            ]
            run_command(eval_command, REPO_ROOT)
            summary_path = run_dir / f"val_predictions_summary_cv_val_{eval_label}.csv"
            fold_rows = summarize_prediction_csv(summary_path, config_name, fold_name, run_name)
            all_fold_rows.extend(fold_rows)
            write_csv(output_dir / "cv_results_by_fold.csv", all_fold_rows)

    config_rows = aggregate_config_results(all_fold_rows, effective_configs)
    write_csv(output_dir / "cv_results_by_config.csv", config_rows)
    ranked = sorted(config_rows, key=lambda row: float(row["mean_val_rmse_m"]))
    write_csv(output_dir / "cv_ranked_configs.csv", ranked)
    plot_config_results(ranked, output_dir / "cv_metric_trends.png")
    if ranked:
        best_name = ranked[0]["config_name"]
        best_config = merged_config(best_name, args)
        best_config["config_name"] = best_name
        with (output_dir / "best_config.yaml").open("w", encoding="utf-8") as f:
            yaml.safe_dump(best_config, f, sort_keys=False)

    print(f"CV outputs written to: {output_dir}")
    if ranked:
        print(f"Best config: {ranked[0]['config_name']} RMSE={float(ranked[0]['mean_val_rmse_m']):.4f} m")


if __name__ == "__main__":
    main()
