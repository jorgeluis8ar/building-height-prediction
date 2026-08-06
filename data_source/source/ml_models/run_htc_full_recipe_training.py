#!/usr/bin/env python3
"""Train the confirmed four-channel full HTC-DC recipe."""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3]
HTC_REPO_DIR = REPO_ROOT / "data_source/source/ml_models/external/HTC-DC-Net"
OUTPUT_ROOT = REPO_ROOT / "data_source/data/ml_models/generated/htc_dc_net/mini_training_runs"
CONFIRMED_SOURCE_DATASET = REPO_ROOT / "data_source/data/ml_models/generated/htc_dc_net/nyc_la_off_nadir_rgb_nir_v1"
DEFAULT_DATASET = REPO_ROOT / "data_source/data/ml_models/generated/htc_dc_net/nyc_la_off_nadir_rgb_nir_full_recipe_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--device", choices=["cpu", "mps"], default="mps")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--validation-every", type=int, default=5)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument(
        "--height-loss-weighting",
        choices=["none", "bins"],
        default="none",
    )
    parser.add_argument("--height-bin-edges", default="3,6,10,25,50")
    parser.add_argument("--height-bin-weights", default="4,3,2,1,3,8")
    parser.add_argument("--preflight-report-name", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def write_csv(path: Path, rows: list[dict]) -> None:
    import csv

    if not rows:
        raise RuntimeError(f"Refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def resolved_config(dataset_dir: Path, args: argparse.Namespace) -> dict:
    relative_dataset = str(dataset_dir.relative_to(REPO_ROOT))
    height_bin_edges = parse_float_list(args.height_bin_edges)
    height_bin_weights = parse_float_list(args.height_bin_weights)
    if len(height_bin_weights) != len(height_bin_edges) + 1:
        raise ValueError(
            "Height-bin weighting requires one more weight than edge: "
            f"edges={height_bin_edges}, weights={height_bin_weights}"
        )
    return {
        "data_dir": relative_dataset,
        "data_split_dirs": relative_dataset,
        "test_data_split_dirs": [relative_dataset],
        "image_size": 256,
        "in_channels": 4,
        "channel_order": ["red", "green", "blue", "nir"],
        "normalize": True,
        "use_mask": True,
        "test_use_mask": True,
        "augmentation_profile": "spatial_spectral",
        "model": "htcdc",
        "backbone": "efficientnetb5",
        "patch_size": 4,
        "num_classes": 256,
        "fusion_mode": "third",
        "head_tail_cut": True,
        "earlier": True,
        "htc_thres": 1.0,
        "htc_source": "pred",
        "prob_loss": "gaussian",
        "prob_loss_bg": "uniform",
        "height_loss_weighting": args.height_loss_weighting,
        "height_bin_edges": height_bin_edges,
        "height_bin_weights": height_bin_weights,
        "background_loss_weight": 0.0,
        "chamfer_weight": 0.01,
        "optimizer": "AdamW",
        "lr": 0.0001,
        "weight_decay": 0.01,
        "batch_size": 8,
        "max_epochs": args.epochs,
        "num_workers": args.num_workers,
        "device": args.device,
        "seed": args.seed,
        "train_shuffle": True,
        "balanced_batches": False,
        "inference_window_size": 256,
        "inference_stride": 128,
        "save_overlap_mean": True,
        "save_overlap_variance": True,
        "checkpoint_selection_metric": "city_balanced_validation_building_rmse",
        "name": args.run_name,
        "project": "HTCDC_FULL_RECIPE_RGB_NIR",
        "test": False,
    }


def main() -> None:
    args = parse_args()
    os.environ.setdefault("WANDB_MODE", "offline")
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    os.environ.setdefault("TORCH_HOME", "/private/tmp/torch_htc_cache")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    set_seed(args.seed)
    dataset_dir = args.dataset_dir.resolve()
    if dataset_dir != DEFAULT_DATASET.resolve():
        raise RuntimeError(
            "This confirmed run is locked to the train-normalized view of "
            "nyc_la_off_nadir_rgb_nir_v1; "
            f"received {dataset_dir}"
        )
    run_dir = OUTPUT_ROOT / args.run_name
    if run_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Run directory exists: {run_dir}")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    cfg = resolved_config(dataset_dir, args)
    (run_dir / "config_used.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8"
    )
    for split in ("train.txt", "val.txt", "test.txt", "all.txt"):
        (run_dir / split).write_text((dataset_dir / split).read_text(encoding="utf-8"), encoding="utf-8")
    preflight_name = args.preflight_report_name or (
        "preflight_binweighted_report.json"
        if args.height_loss_weighting == "bins"
        else "preflight_report.json"
    )
    preflight_path = dataset_dir / preflight_name
    if not preflight_path.exists():
        raise FileNotFoundError(
            f"Required preflight report is missing: {preflight_path}. Run preflight first."
        )
    shutil.copy2(preflight_path, run_dir / "preflight_report.json")
    source_snapshot = run_dir / "source_snapshot"
    source_snapshot.mkdir()
    source_files = [
        SCRIPT_PATH,
        REPO_ROOT / "data_source/source/ml_models/htc_full_recipe_training.py",
        REPO_ROOT / "data_source/source/ml_models/htc_sliding_window_inference.py",
        REPO_ROOT / "data_source/source/ml_models/preflight_htc_full_recipe.py",
        REPO_ROOT / "data_source/source/ml_models/prepare_htc_full_recipe_dataset.py",
        REPO_ROOT / "data_source/source/ml_models/external/HTC-DC-Net/build.py",
        REPO_ROOT / "data_source/source/ml_models/external/HTC-DC-Net/htcdc.py",
        REPO_ROOT / "data_source/source/ml_models/external/HTC-DC-Net/parts/losses.py",
    ]
    for source_file in source_files:
        destination = source_snapshot / source_file.relative_to(REPO_ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_file, destination)
    try:
        diff = subprocess.run(
            ["git", "diff", "--", "data_source/source/ml_models"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except subprocess.CalledProcessError as error:
        diff = f"git diff failed: {error}"
    (run_dir / "source_code_diff.patch").write_text(diff, encoding="utf-8")
    git_status = subprocess.run(
        ["git", "status", "--short"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    ).stdout
    (run_dir / "git_status_at_start.txt").write_text(git_status, encoding="utf-8")
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_dir": str(dataset_dir.relative_to(REPO_ROOT)),
        "confirmed_source_dataset": str(CONFIRMED_SOURCE_DATASET.relative_to(REPO_ROOT)),
        "dataset_confirmed_by_user": True,
        "split_counts": {"train": 171, "validation": 36, "test": 37},
        "training_city_counts": {"new_york_city": 76, "los_angeles": 95},
        "seed": args.seed,
        "device": args.device,
        "validation_every": args.validation_every,
        "checkpoint_every": args.checkpoint_every,
        "height_loss_weighting": args.height_loss_weighting,
        "height_bin_edges": cfg["height_bin_edges"],
        "height_bin_weights": cfg["height_bin_weights"],
        "preflight_report": preflight_name,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "stop_reason": None,
    }
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    sys.path.insert(0, str(HTC_REPO_DIR))
    from build import get_model_and_optimizer
    from htc_full_recipe_training import run_training

    try:
        result = run_training(
            cfg=cfg,
            dataset_dir=dataset_dir,
            run_dir=run_dir,
            epochs=args.epochs,
            seed=args.seed,
            device=args.device,
            num_workers=args.num_workers,
            validation_every=args.validation_every,
            checkpoint_every=args.checkpoint_every,
            get_model_and_optimizer=get_model_and_optimizer,
            write_csv=write_csv,
        )
        metadata.update(result)
        metadata["stop_reason"] = "maximum_epochs_completed"
    except Exception as error:
        metadata["stop_reason"] = "failed"
        metadata["failure"] = repr(error)
        (run_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        raise
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Run directory: {run_dir}")
    print(f"Best epoch: {result['best_epoch']}")
    print(
        "Best city-balanced validation building RMSE: "
        f"{result['best_city_balanced_building_rmse_m']:.4f} m"
    )


if __name__ == "__main__":
    main()
