#!/usr/bin/env python3
"""Export HTC-DC Net checkpoint predictions for a held-out dataset split."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3]
DEFAULT_DATASET_DIR = REPO_ROOT / "data_source/data/ml_models/generated/htc_dc_net/nyc_la_rgb_v1"
HTC_REPO_DIR = REPO_ROOT / "data_source/source/ml_models/external/HTC-DC-Net"
DEFAULT_RUN_DIR = (
    REPO_ROOT
    / "data_source/data/ml_models/generated/htc_dc_net/mini_training_runs/"
    / "nyc71_la100_seed20260707_epoch5_repo_params_guarded"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="val")
    parser.add_argument("--output-subdir", default=None)
    parser.add_argument("--output-label", default=None)
    parser.add_argument("--collapse-std-threshold", type=float, default=0.05)
    parser.add_argument("--collapse-min-share", type=float, default=0.8)
    parser.add_argument("--mask-predictions", action="store_true")
    parser.add_argument("--prediction-nodata", type=float, default=-9999.0)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def import_project_runner():
    sys.path.insert(0, str(SCRIPT_PATH.parent))
    from run_htc_mini_training import export_predictions, summarize_collapse_check, write_csv  # noqa: PLC0415

    return export_predictions, summarize_collapse_check, write_csv


def import_htc_model_builder():
    sys.path.insert(0, str(HTC_REPO_DIR))
    from build import get_model_and_optimizer  # noqa: PLC0415

    return get_model_and_optimizer


def read_split_ids(dataset_dir: Path, split: str) -> set[str]:
    split_path = dataset_dir / f"{split}.txt"
    with split_path.open("r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def read_manifest_rows(dataset_dir: Path, split: str) -> list[dict]:
    if split == "all":
        split_ids = None
    else:
        split_ids = read_split_ids(dataset_dir, split)

    rows = []
    with (dataset_dir / "chips_manifest.csv").open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if split_ids is None or row["chip_id"] in split_ids:
                rows.append(row)
    rows.sort(key=lambda row: (row["source_city"], row["chip_id"]))
    return rows


def load_model(checkpoint_path: Path, dataset_dir: Path, device: str):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg = dict(checkpoint["cfg"])
    cfg["device"] = device
    cfg["restore"] = False
    cfg["data_dir"] = str(dataset_dir.resolve().relative_to(REPO_ROOT))
    cfg["data_split_dirs"] = cfg["data_dir"]
    cfg["test_data_split_dirs"] = [cfg["data_dir"]]

    get_model_and_optimizer = import_htc_model_builder()
    model, _ = get_model_and_optimizer(cfg)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    return model, cfg


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    dataset_dir = args.dataset_dir.resolve()
    checkpoint = (args.checkpoint or run_dir / "model_epoch_005.pth").resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    output_label = args.output_label or checkpoint.stem.replace("model_", "")
    output_subdir = args.output_subdir or f"{args.split}_predictions_{output_label}"
    summary_path = run_dir / f"{args.split}_predictions_summary_{output_label}.csv"
    collapse_path = run_dir / f"{args.split}_prediction_collapse_check_{output_label}.csv"

    export_predictions, summarize_collapse_check, write_csv = import_project_runner()
    selected = read_manifest_rows(dataset_dir, args.split)
    model, cfg = load_model(checkpoint, dataset_dir, args.device)

    prediction_rows = export_predictions(
        model=model,
        cfg=cfg,
        selected=selected,
        run_dir=run_dir,
        data_dir=dataset_dir,
        output_subdir=output_subdir,
        collapse_std_threshold=args.collapse_std_threshold,
        mask_predictions=args.mask_predictions,
        prediction_nodata=args.prediction_nodata,
    )
    write_csv(summary_path, prediction_rows)
    collapse_check = summarize_collapse_check(
        epoch=output_label,
        prediction_rows=prediction_rows,
        collapse_std_threshold=args.collapse_std_threshold,
        collapse_min_share=args.collapse_min_share,
    )
    write_csv(collapse_path, [collapse_check])

    print(f"Split: {args.split}")
    print(f"Dataset: {dataset_dir}")
    print(f"Chips: {len(selected)}")
    print(f"Checkpoint: {checkpoint}")
    print(f"Predictions: {run_dir / output_subdir}")
    print(f"Summary: {summary_path}")
    print(f"Collapse check: {collapse_path}")


if __name__ == "__main__":
    main()
