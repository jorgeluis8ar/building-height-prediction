"""
Combine HTC-DC Net Chip Datasets

Creates one model-ready HTC-DC Net dataset from city-level chip datasets.

Inputs:
    - data_source/data/height_labels/generated/new_york_city/lidar_ndsm/htc_dc_net/
    - data_source/data/height_labels/generated/los_angeles/lidar_ndsm/htc_dc_net/

Output:
    - data_source/data/ml_models/generated/htc_dc_net/nyc_la_rgb_v1/
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import os
from pathlib import Path
import pickle
import random
import shutil
import sys
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
HEIGHT_LABELS_VENV = (
    PROJECT_ROOT / "data_source/source/height_labels/venv_height_labels"
)
VENV_PYTHON = HEIGHT_LABELS_VENV / (
    "Scripts/python.exe" if os.name == "nt" else "bin/python"
)
VENV_MARKER = "ML_MODELS_HEIGHT_LABELS_VENV_ACTIVE"

DATASET_SOURCES = [
    {
        "city": "new_york_city",
        "scene_id": "20200122_154449_92_1061",
        "dataset_root": (
            PROJECT_ROOT
            / "data_source/data/height_labels/generated/new_york_city/lidar_ndsm/"
            / "htc_dc_net/20200122_154449_92_1061"
        ),
    },
    {
        "city": "los_angeles",
        "scene_id": "20231203_182937_07_2488",
        "dataset_root": (
            PROJECT_ROOT
            / "data_source/data/height_labels/generated/los_angeles/lidar_ndsm/"
            / "htc_dc_net/20231203_182937_07_2488"
        ),
    },
]

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "data_source/data/ml_models/generated/htc_dc_net/nyc_la_rgb_v1"
)
DEFAULT_SEED = 20260706


def relaunch_inside_venv() -> None:
    """Restart this script inside the environment that has rasterio/numpy."""
    if os.environ.get(VENV_MARKER) == "1":
        return

    current_python = Path(sys.executable).absolute()
    expected_python = VENV_PYTHON.absolute()
    if current_python == expected_python or Path(sys.prefix).absolute() == HEIGHT_LABELS_VENV.absolute():
        os.environ[VENV_MARKER] = "1"
        return

    if not VENV_PYTHON.exists():
        print(f"ERROR: Missing expected Python executable: {VENV_PYTHON}")
        sys.exit(1)

    environment = os.environ.copy()
    environment[VENV_MARKER] = "1"
    os.execve(
        str(VENV_PYTHON),
        [str(VENV_PYTHON), str(Path(__file__).resolve())] + sys.argv[1:],
        environment,
    )


def parse_args() -> argparse.Namespace:
    """Read command-line options."""
    parser = argparse.ArgumentParser(
        description="Combine NYC and LA HTC-DC Net chip datasets."
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Combined HTC dataset output directory.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Deterministic split seed. Default: {DEFAULT_SEED}.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the existing combined dataset directory.",
    )
    return parser.parse_args()


def relative_path(path: Path) -> str:
    """Return a project-relative path string."""
    return path.resolve().relative_to(PROJECT_ROOT).as_posix()


def ensure_clean_output(output_dir: Path, overwrite: bool) -> None:
    """Create an empty output directory."""
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output exists: {output_dir}. Pass --overwrite.")
        shutil.rmtree(output_dir)
    for name in ["image", "mask", "ndsm", "stats"]:
        (output_dir / name).mkdir(parents=True, exist_ok=True)


def read_manifest(source: dict[str, Any]) -> list[dict[str, Any]]:
    """Read and validate one source chip manifest."""
    manifest_path = source["dataset_root"] / "chips_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")

    rows = []
    with manifest_path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            chip_id = row["chip_id"]
            image_path = PROJECT_ROOT / row["image_path"]
            mask_path = PROJECT_ROOT / row["mask_path"]
            agl_path = PROJECT_ROOT / row["agl_path"]
            for path in [image_path, mask_path, agl_path]:
                if not path.exists():
                    raise FileNotFoundError(f"Missing chip file: {path}")

            rows.append(
                {
                    "chip_id": chip_id,
                    "source_city": source["city"],
                    "source_scene_id": source["scene_id"],
                    "source_dataset_root": relative_path(source["dataset_root"]),
                    "row_off": row["row_off"],
                    "col_off": row["col_off"],
                    "positive_agl_pixels": row["positive_agl_pixels"],
                    "building_mask_pixels": row["building_mask_pixels"],
                    "source_image_path": relative_path(image_path),
                    "source_mask_path": relative_path(mask_path),
                    "source_agl_path": relative_path(agl_path),
                    "_image_path_abs": image_path,
                    "_mask_path_abs": mask_path,
                    "_agl_path_abs": agl_path,
                }
            )
    return rows


def copy_chip_files(rows: list[dict[str, Any]], output_dir: Path) -> list[dict[str, Any]]:
    """Copy all chip triplets into the combined dataset folders."""
    output_rows = []
    seen = set()
    for row in rows:
        chip_id = row["chip_id"]
        if chip_id in seen:
            raise RuntimeError(f"Duplicate chip_id found: {chip_id}")
        seen.add(chip_id)

        image_out = output_dir / "image" / f"{chip_id}_IMG.tif"
        mask_out = output_dir / "mask" / f"{chip_id}_BLG.tif"
        agl_out = output_dir / "ndsm" / f"{chip_id}_AGL.tif"

        shutil.copy2(row["_image_path_abs"], image_out)
        shutil.copy2(row["_mask_path_abs"], mask_out)
        shutil.copy2(row["_agl_path_abs"], agl_out)

        clean_row = {key: value for key, value in row.items() if not key.startswith("_")}
        clean_row.update(
            {
                "image_path": relative_path(image_out),
                "mask_path": relative_path(mask_out),
                "agl_path": relative_path(agl_out),
            }
        )
        output_rows.append(clean_row)
    return output_rows


def split_chip_ids(chip_ids: list[str], seed: int) -> dict[str, list[str]]:
    """Create deterministic 70/15/15 train/val/test split files."""
    shuffled = chip_ids[:]
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    train_end = int(len(shuffled) * 0.70)
    val_end = int(len(shuffled) * 0.85)
    return {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
        "all": shuffled,
    }


def write_split_files(output_dir: Path, splits: dict[str, list[str]]) -> None:
    """Write split files with one chip ID per line."""
    for name, chip_ids in splits.items():
        (output_dir / f"{name}.txt").write_text(
            "".join(f"{chip_id}\n" for chip_id in chip_ids),
            encoding="utf-8",
        )


def write_manifest(output_dir: Path, rows: list[dict[str, Any]], splits: dict[str, list[str]]) -> None:
    """Write combined chip manifest."""
    split_lookup = {
        chip_id: split
        for split, chip_ids in splits.items()
        if split != "all"
        for chip_id in chip_ids
    }
    output_rows = []
    for row in rows:
        row = row.copy()
        row["split"] = split_lookup[row["chip_id"]]
        output_rows.append(row)

    fieldnames = [
        "chip_id",
        "split",
        "source_city",
        "source_scene_id",
        "source_dataset_root",
        "row_off",
        "col_off",
        "positive_agl_pixels",
        "building_mask_pixels",
        "source_image_path",
        "source_mask_path",
        "source_agl_path",
        "image_path",
        "mask_path",
        "agl_path",
    ]
    with (output_dir / "chips_manifest.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)


def compute_stats(rows: list[dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    """Compute combined image and positive-AGL statistics."""
    import rasterio

    channel_sum = None
    channel_sum_sq = None
    channel_count = 0
    agl_sum = 0.0
    agl_sum_sq = 0.0
    agl_count = 0

    for row in rows:
        with rasterio.open(PROJECT_ROOT / row["image_path"]) as image:
            data = image.read(masked=True).astype("float64")
            valid = ~np.ma.getmaskarray(data)
            if channel_sum is None:
                channel_sum = np.zeros(data.shape[0], dtype="float64")
                channel_sum_sq = np.zeros(data.shape[0], dtype="float64")
            for band in range(data.shape[0]):
                vals = data[band][valid[band]].compressed()
                channel_sum[band] += vals.sum()
                channel_sum_sq[band] += np.square(vals).sum()
            channel_count += int(valid[0].sum())

        with rasterio.open(PROJECT_ROOT / row["agl_path"]) as agl:
            target = agl.read(1, masked=True).astype("float64")
            valid_agl = (target > 0) & ~np.ma.getmaskarray(target)
            vals = target[valid_agl].compressed()
            agl_sum += vals.sum()
            agl_sum_sq += np.square(vals).sum()
            agl_count += len(vals)

    if channel_sum is None or channel_count == 0:
        raise RuntimeError("No valid image pixels were available for stats.")
    if agl_count == 0:
        raise RuntimeError("No positive AGL pixels were available for stats.")

    image_mean = channel_sum / channel_count
    image_var = np.maximum(channel_sum_sq / channel_count - np.square(image_mean), 0)
    image_std = np.sqrt(image_var)
    agl_mean = agl_sum / agl_count
    agl_std = np.sqrt(max(agl_sum_sq / agl_count - agl_mean**2, 0))

    image_stats = {
        "image_mean": image_mean.tolist(),
        "image_std": image_std.tolist(),
        "image_pixel_count": int(channel_count),
        "channel_order": ["red", "green", "blue"],
    }
    ndsm_stats = {
        "ndsm_positive_mean": float(agl_mean),
        "ndsm_positive_std": float(agl_std),
        "ndsm_positive_pixel_count": int(agl_count),
        "target": "building_only_agl_m",
    }
    combined_stats = image_stats | ndsm_stats

    stats_dir = output_dir / "stats"
    with (stats_dir / "image_stats.pickle").open("wb") as file:
        pickle.dump(image_stats, file)
    with (stats_dir / "ndsm_stats.pickle").open("wb") as file:
        pickle.dump(ndsm_stats, file)
    with (stats_dir / "combined_stats.pickle").open("wb") as file:
        pickle.dump(combined_stats, file)
    return combined_stats


def write_readme(
    output_dir: Path,
    rows: list[dict[str, Any]],
    splits: dict[str, list[str]],
    stats: dict[str, Any],
    seed: int,
) -> None:
    """Write a small dataset README."""
    counts_by_city: dict[str, int] = {}
    for row in rows:
        counts_by_city[row["source_city"]] = counts_by_city.get(row["source_city"], 0) + 1

    city_lines = "\n".join(
        f"- `{city}`: {count} chips" for city, count in sorted(counts_by_city.items())
    )
    text = f"""# NYC + LA HTC-DC Net RGB Dataset v1

Created: {datetime.now(timezone.utc).isoformat()}

This dataset combines the true New York City and Los Angeles HTC-DC Net chip
datasets. It excludes the NYC/New Jersey Sandy LiDAR diagnostic variant.

## Contents

```text
image/   *_IMG.tif  3-band Planet RGB chips
mask/    *_BLG.tif  building-mask chips
ndsm/    *_AGL.tif  building-only LiDAR nDSM target chips
stats/   normalization statistics
```

## Counts

Total chips: {len(rows)}

{city_lines}

Split seed: `{seed}`

| Split | Chips |
|---|---:|
| Train | {len(splits["train"])} |
| Validation | {len(splits["val"])} |
| Test | {len(splits["test"])} |
| All | {len(splits["all"])} |

## Statistics

Image channel order is RGB.

```text
image_mean = {stats["image_mean"]}
image_std = {stats["image_std"]}
ndsm_positive_mean = {stats["ndsm_positive_mean"]}
ndsm_positive_std = {stats["ndsm_positive_std"]}
```
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def write_summary(
    output_dir: Path,
    rows: list[dict[str, Any]],
    splits: dict[str, list[str]],
    stats: dict[str, Any],
    seed: int,
) -> None:
    """Write one-row CSV dataset summary."""
    counts_by_city: dict[str, int] = {}
    for row in rows:
        counts_by_city[row["source_city"]] = counts_by_city.get(row["source_city"], 0) + 1
    summary = {
        "dataset": "nyc_la_rgb_v1",
        "output_dir": relative_path(output_dir),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "total_chips": len(rows),
        "train_chips": len(splits["train"]),
        "val_chips": len(splits["val"]),
        "test_chips": len(splits["test"]),
        "new_york_city_chips": counts_by_city.get("new_york_city", 0),
        "los_angeles_chips": counts_by_city.get("los_angeles", 0),
        "image_mean": stats["image_mean"],
        "image_std": stats["image_std"],
        "ndsm_positive_mean": stats["ndsm_positive_mean"],
        "ndsm_positive_std": stats["ndsm_positive_std"],
    }
    with (output_dir / "dataset_summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)


def main() -> None:
    """Create the combined NYC+LA HTC dataset."""
    relaunch_inside_venv()
    args = parse_args()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    ensure_clean_output(output_dir, args.overwrite)

    rows = []
    for source in DATASET_SOURCES:
        rows.extend(read_manifest(source))
    copied_rows = copy_chip_files(rows, output_dir)
    splits = split_chip_ids([row["chip_id"] for row in copied_rows], args.seed)
    write_split_files(output_dir, splits)
    write_manifest(output_dir, copied_rows, splits)
    stats = compute_stats(copied_rows, output_dir)
    write_readme(output_dir, copied_rows, splits, stats, args.seed)
    write_summary(output_dir, copied_rows, splits, stats, args.seed)

    print(f"Wrote combined HTC dataset: {relative_path(output_dir)}")
    print(f"Total chips: {len(copied_rows)}")
    print(
        "Splits: "
        f"train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}"
    )


if __name__ == "__main__":
    main()
