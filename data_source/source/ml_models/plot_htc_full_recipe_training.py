#!/usr/bin/env python3
"""Plot training loss and validation building RMSE for a full HTC run."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    history = read_csv(run_dir / "training_history.csv")
    validation = read_csv(run_dir / "validation_epoch_metrics.csv")
    epoch_losses: dict[int, list[float]] = {}
    for row in history:
        epoch_losses.setdefault(int(row["epoch"]), []).append(float(row["loss_total"]))
    epochs = sorted(epoch_losses)
    mean_losses = [float(np.mean(epoch_losses[epoch])) for epoch in epochs]
    validation_epochs = [int(row["epoch"]) for row in validation]

    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(epochs, mean_losses, color="#2a6f97", linewidth=2)
    axes[0].set(title="Training loss", xlabel="Epoch", ylabel="Mean total loss")
    axes[0].grid(alpha=0.25)
    series = (
        ("los_angeles_building_rmse_m", "Los Angeles", "#2a6f97"),
        ("new_york_city_building_rmse_m", "New York City", "#d1495b"),
        ("city_balanced_building_rmse_m", "City-balanced", "#3a7d44"),
    )
    for key, label, color in series:
        axes[1].plot(
            validation_epochs,
            [float(row[key]) for row in validation],
            marker="o",
            label=label,
            color=color,
        )
    best_row = min(validation, key=lambda row: float(row["city_balanced_building_rmse_m"]))
    axes[1].axvline(int(best_row["epoch"]), color="#555555", linestyle="--", alpha=0.7)
    axes[1].set(
        title=f"Validation building RMSE (best epoch {best_row['epoch']})",
        xlabel="Epoch",
        ylabel="RMSE (m)",
    )
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    figure.suptitle("Full HTC-DC RGB+NIR EfficientNet-B5 Training")
    figure.tight_layout()
    output = run_dir / "evaluation/training_and_validation_metric_trends.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    print(output)


if __name__ == "__main__":
    main()

