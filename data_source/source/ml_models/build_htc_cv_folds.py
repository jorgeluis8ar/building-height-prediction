#!/usr/bin/env python3
"""Create cross-validation folds for an HTC-DC Net dataset.

The folds are written as lightweight HTC dataset folders. Each fold has its
own train/val/test split files and manifest, while image/mask/ndsm folders are
symlinks back to the parent dataset. This avoids copying large GeoTIFF chips.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import rasterio


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3]
DEFAULT_DATASET_DIR = REPO_ROOT / "data_source/data/ml_models/generated/htc_dc_net/nyc_la_12ch_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--output-subdir", default="cv_folds")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_ids(path: Path) -> list[str]:
    """Read a split file into a list of chip ids."""
    with path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def write_ids(path: Path, chip_ids: list[str]) -> None:
    """Write one chip id per line."""
    with path.open("w", encoding="utf-8") as f:
        for chip_id in chip_ids:
            f.write(f"{chip_id}\n")


def read_manifest(dataset_dir: Path) -> dict[str, dict]:
    """Read chips_manifest.csv and index rows by chip id."""
    rows = {}
    with (dataset_dir / "chips_manifest.csv").open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows[row["chip_id"]] = row
    return rows


def target_height_summary(dataset_dir: Path, chip_id: str) -> dict:
    """Summarize positive AGL pixels for stratification."""
    ndsm_path = dataset_dir / "ndsm" / f"{chip_id}_AGL.tif"
    if not ndsm_path.exists():
        raise FileNotFoundError(ndsm_path)
    with rasterio.open(ndsm_path) as src:
        arr = src.read(1, masked=True).astype("float32")
    values = arr.compressed()
    values = values[np.isfinite(values) & (values > 0)]
    if values.size == 0:
        return {
            "target_mean_m": math.nan,
            "target_p90_m": math.nan,
            "target_p95_m": math.nan,
            "target_max_m": math.nan,
            "height_bin": "empty",
        }
    p95 = float(np.percentile(values, 95))
    if p95 < 6:
        height_bin = "p95_00_06m"
    elif p95 < 10:
        height_bin = "p95_06_10m"
    elif p95 < 25:
        height_bin = "p95_10_25m"
    elif p95 < 50:
        height_bin = "p95_25_50m"
    else:
        height_bin = "p95_50m_plus"
    return {
        "target_mean_m": float(np.mean(values)),
        "target_p90_m": float(np.percentile(values, 90)),
        "target_p95_m": p95,
        "target_max_m": float(np.max(values)),
        "height_bin": height_bin,
    }


def assign_folds(rows: list[dict], folds: int, seed: int) -> dict[str, int]:
    """Assign chips to folds while balancing city and height-bin groups."""
    rng = np.random.default_rng(seed)
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["source_city"], row["height_bin"])].append(row)

    assignment = {}
    for _, group_rows in sorted(groups.items()):
        indexes = np.arange(len(group_rows))
        rng.shuffle(indexes)
        for position, row_index in enumerate(indexes):
            assignment[group_rows[int(row_index)]["chip_id"]] = position % folds
    return assignment


def safe_link_or_copy(src: Path, dst: Path) -> None:
    """Create a symlink, falling back to a copy if symlinks are unavailable."""
    if dst.exists() or dst.is_symlink():
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    try:
        dst.symlink_to(src, target_is_directory=src.is_dir())
    except OSError:
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)


def write_manifest(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"No rows available for {path}")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    if not dataset_dir.exists():
        raise FileNotFoundError(dataset_dir)

    output_dir = dataset_dir / args.output_subdir
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_dir} exists. Use --overwrite to rebuild folds.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    manifest = read_manifest(dataset_dir)
    train_ids = read_ids(dataset_dir / "train.txt")
    val_ids = read_ids(dataset_dir / "val.txt")
    test_ids = read_ids(dataset_dir / "test.txt")
    cv_pool = train_ids + val_ids
    if set(cv_pool) & set(test_ids):
        raise RuntimeError("The CV pool overlaps the test split. Refusing to continue.")

    annotated = []
    for chip_id in cv_pool:
        if chip_id not in manifest:
            raise RuntimeError(f"{chip_id} is missing from chips_manifest.csv")
        row = manifest[chip_id].copy()
        row.update(target_height_summary(dataset_dir, chip_id))
        annotated.append(row)

    assignment = assign_folds(annotated, folds=args.folds, seed=args.seed)
    fold_manifest_rows = []
    for fold_index in range(args.folds):
        fold_name = f"fold_{fold_index + 1:02d}"
        fold_dir = output_dir / fold_name
        fold_dir.mkdir(parents=True)
        for name in ["image", "mask", "ndsm"]:
            safe_link_or_copy(dataset_dir / name, fold_dir / name)
        for stats_name in ["image_stats.pickle", "ndsm_stats.pickle"]:
            safe_link_or_copy(dataset_dir / stats_name, fold_dir / stats_name)
        safe_link_or_copy(dataset_dir / "stats", fold_dir / "stats")

        val_fold_ids = sorted([chip_id for chip_id, fold in assignment.items() if fold == fold_index])
        train_fold_ids = sorted([chip_id for chip_id in cv_pool if assignment[chip_id] != fold_index])
        write_ids(fold_dir / "train.txt", train_fold_ids)
        write_ids(fold_dir / "val.txt", val_fold_ids)
        write_ids(fold_dir / "test.txt", test_ids)
        write_ids(fold_dir / "all.txt", sorted(cv_pool + test_ids))

        fold_rows = []
        for chip_id in sorted(cv_pool + test_ids):
            row = manifest[chip_id].copy()
            if chip_id in val_fold_ids:
                row["split"] = "val"
            elif chip_id in train_fold_ids:
                row["split"] = "train"
            else:
                row["split"] = "test"
            fold_rows.append(row)
        write_manifest(fold_dir / "chips_manifest.csv", fold_rows)

        train_set = set(train_fold_ids)
        val_set = set(val_fold_ids)
        if train_set & val_set:
            raise RuntimeError(f"{fold_name} has overlapping train/validation chips.")
        city_counts = defaultdict(lambda: {"train": 0, "val": 0})
        for row in fold_rows:
            if row["split"] in {"train", "val"}:
                city_counts[row["source_city"]][row["split"]] += 1
        for city, counts in city_counts.items():
            if counts["train"] == 0 or counts["val"] == 0:
                raise RuntimeError(f"{fold_name} is missing {city} train or validation chips.")
        for chip_id in val_fold_ids:
            pool_row = next(row for row in annotated if row["chip_id"] == chip_id)
            fold_manifest_rows.append(
                {
                    "fold": fold_name,
                    "chip_id": chip_id,
                    "source_city": pool_row["source_city"],
                    "height_bin": pool_row["height_bin"],
                    "target_mean_m": pool_row["target_mean_m"],
                    "target_p90_m": pool_row["target_p90_m"],
                    "target_p95_m": pool_row["target_p95_m"],
                    "target_max_m": pool_row["target_max_m"],
                }
            )

    validation_coverage = [row["chip_id"] for row in fold_manifest_rows]
    if sorted(validation_coverage) != sorted(cv_pool):
        raise RuntimeError("Validation folds do not cover the CV pool exactly once.")

    write_manifest(output_dir / "fold_manifest.csv", fold_manifest_rows)
    summary = {
        "dataset_dir": str(dataset_dir.relative_to(REPO_ROOT)),
        "output_dir": str(output_dir.relative_to(REPO_ROOT)),
        "folds": args.folds,
        "seed": args.seed,
        "cv_pool_chips": len(cv_pool),
        "test_chips_untouched": len(test_ids),
    }
    with (output_dir / "fold_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"CV folds written to: {output_dir}")
    print(f"CV pool chips: {len(cv_pool)}")
    print(f"Test chips untouched: {len(test_ids)}")
    print(f"Fold manifest: {output_dir / 'fold_manifest.csv'}")


if __name__ == "__main__":
    main()
