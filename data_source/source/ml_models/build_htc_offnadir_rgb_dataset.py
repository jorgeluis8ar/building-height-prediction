#!/usr/bin/env python3
"""Combine rebuilt LA and NYC off-nadir RGB chips using reference splits.

This script reproduces the three-channel dataset used by the selected model.
The large raster chips are rebuilt locally, while a lightweight reference
dataset supplies the exact train/validation/test assignment from the original
experiment. No new random split is created.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import pickle
import shutil
from typing import Any

import numpy as np
import rasterio
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
SPLIT_NAMES = ("train", "val", "test")
ALL_SPLIT_NAMES = (*SPLIT_NAMES, "all")
EXPECTED_CHANNELS = 3
EXPECTED_CHIP_SIZE = 256
EXPECTED_RESOLUTION_M = 3.0
NODATA_AGL = -9999.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--la-source",
        type=Path,
        required=True,
        help="LA city-level HTC directory containing image/mask/ndsm and chips_manifest.csv.",
    )
    parser.add_argument(
        "--nyc-source",
        type=Path,
        required=True,
        help="NYC city-level HTC directory containing image/mask/ndsm and chips_manifest.csv.",
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        required=True,
        help="Lightweight original dataset metadata containing split files and chips_manifest.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Model-ready combined dataset directory to create.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace output-dir when it already exists.",
    )
    parser.add_argument(
        "--copy-mode",
        choices=("copy", "hardlink"),
        default="copy",
        help="Use normal copies by default; hardlink is useful for local testing on one drive.",
    )
    return parser.parse_args()


def resolve_project_path(path: Path) -> Path:
    """Resolve a CLI path from the repository root and reject outside paths."""
    resolved = path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()
    try:
        resolved.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise ValueError(f"Path must remain inside the repository: {resolved}") from exc
    return resolved


def project_relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT).as_posix()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required CSV does not exist: {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise RuntimeError(f"CSV contains no records: {path}")
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_split_files(reference_dir: Path) -> dict[str, list[str]]:
    """Read and verify the exact split assignment from the reference dataset."""
    splits: dict[str, list[str]] = {}
    for split in ALL_SPLIT_NAMES:
        path = reference_dir / f"{split}.txt"
        if not path.is_file():
            raise FileNotFoundError(f"Reference split file is missing: {path}")
        values = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        if not values:
            raise RuntimeError(f"Reference split file is empty: {path}")
        if len(values) != len(set(values)):
            raise RuntimeError(f"Reference split contains duplicate chip IDs: {path}")
        splits[split] = values

    split_sets = {name: set(splits[name]) for name in SPLIT_NAMES}
    for index, first in enumerate(SPLIT_NAMES):
        for second in SPLIT_NAMES[index + 1 :]:
            overlap = split_sets[first] & split_sets[second]
            if overlap:
                example = sorted(overlap)[0]
                raise RuntimeError(f"Reference splits {first} and {second} overlap; example: {example}")

    covered = set().union(*(split_sets[name] for name in SPLIT_NAMES))
    if covered != set(splits["all"]):
        missing = set(splits["all"]) - covered
        extra = covered - set(splits["all"])
        raise RuntimeError(
            "Reference train/val/test files do not exactly cover all.txt: "
            f"missing={len(missing)}, extra={len(extra)}"
        )
    return splits


def load_reference(reference_dir: Path, splits: dict[str, list[str]]) -> dict[str, dict[str, str]]:
    rows = read_csv(reference_dir / "chips_manifest.csv")
    required = {"chip_id", "source_city", "source_scene_id", "row_off", "col_off"}
    missing_columns = required - set(rows[0])
    if missing_columns:
        raise RuntimeError(f"Reference manifest lacks columns: {sorted(missing_columns)}")

    by_id: dict[str, dict[str, str]] = {}
    for row in rows:
        chip_id = row["chip_id"]
        if chip_id in by_id:
            raise RuntimeError(f"Duplicate chip ID in reference manifest: {chip_id}")
        by_id[chip_id] = row

    expected = set(splits["all"])
    if set(by_id) != expected:
        raise RuntimeError(
            "Reference chips_manifest.csv and all.txt differ: "
            f"manifest_only={len(set(by_id) - expected)}, all_only={len(expected - set(by_id))}"
        )
    return by_id


def load_source(source_dir: Path, city: str) -> dict[str, dict[str, Any]]:
    """Load one rebuilt city manifest and reconstruct local chip paths."""
    rows = read_csv(source_dir / "chips_manifest.csv")
    required = {"chip_id", "row_off", "col_off", "positive_agl_pixels", "building_mask_pixels"}
    missing_columns = required - set(rows[0])
    if missing_columns:
        raise RuntimeError(f"Source manifest {source_dir} lacks columns: {sorted(missing_columns)}")

    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        chip_id = row["chip_id"]
        if chip_id in by_id:
            raise RuntimeError(f"Duplicate chip ID in source manifest: {chip_id}")
        paths = {
            "image": source_dir / "image" / f"{chip_id}_IMG.tif",
            "mask": source_dir / "mask" / f"{chip_id}_BLG.tif",
            "agl": source_dir / "ndsm" / f"{chip_id}_AGL.tif",
        }
        missing = [str(path) for path in paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Source chip {chip_id} is incomplete: {missing}")
        by_id[chip_id] = {"city": city, "row": row, **paths}
    return by_id


def prepare_output(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {output_dir}. Pass --overwrite to replace it.")
        shutil.rmtree(output_dir)
    for name in ("image", "mask", "ndsm", "stats"):
        (output_dir / name).mkdir(parents=True, exist_ok=True)


def transfer_file(source: Path, destination: Path, mode: str) -> None:
    if mode == "copy":
        shutil.copy2(source, destination)
        return
    try:
        os.link(source, destination)
    except OSError as exc:
        raise OSError(
            f"Could not hardlink {source} to {destination}. Use --copy-mode copy when paths are on different drives."
        ) from exc


def raster_signature(dataset: rasterio.io.DatasetReader) -> tuple[Any, ...]:
    return (
        dataset.crs,
        dataset.transform,
        dataset.width,
        dataset.height,
        dataset.bounds,
        dataset.res,
    )


def validate_chip(image_path: Path, mask_path: Path, agl_path: Path, chip_id: str) -> dict[str, Any]:
    """Fail immediately when one RGB/mask/AGL triplet is unusable or misaligned."""
    with rasterio.open(image_path) as image, rasterio.open(mask_path) as mask, rasterio.open(agl_path) as agl:
        if image.count != EXPECTED_CHANNELS:
            raise RuntimeError(f"{chip_id}: image has {image.count} bands; expected {EXPECTED_CHANNELS} RGB bands")
        if mask.count != 1 or agl.count != 1:
            raise RuntimeError(f"{chip_id}: mask and AGL must each contain one band")
        if raster_signature(image) != raster_signature(mask) or raster_signature(image) != raster_signature(agl):
            raise RuntimeError(f"{chip_id}: image, mask, and AGL grids are not identical")
        if image.width != EXPECTED_CHIP_SIZE or image.height != EXPECTED_CHIP_SIZE:
            raise RuntimeError(f"{chip_id}: expected 256x256 pixels, found {image.width}x{image.height}")
        if not np.allclose(np.abs(image.res), (EXPECTED_RESOLUTION_M, EXPECTED_RESOLUTION_M)):
            raise RuntimeError(f"{chip_id}: expected 3 m resolution, found {image.res}")

        image_data = image.read(masked=True)
        image_values = image_data.compressed()
        if image_values.size == 0 or not np.isfinite(image_values).any() or float(np.nanmax(image_values)) <= 0:
            raise RuntimeError(f"{chip_id}: RGB image is empty or nonpositive")

        mask_data = mask.read(1, masked=True)
        mask_values = np.asarray(mask_data.compressed())
        if mask_values.size == 0 or not np.isin(mask_values, [0, 1]).all():
            raise RuntimeError(f"{chip_id}: building mask is empty or contains values other than 0 and 1")
        building_pixels = int((mask_values > 0).sum())
        if building_pixels == 0:
            raise RuntimeError(f"{chip_id}: building mask contains no building pixels")

        agl_data = agl.read(1, masked=True)
        agl_values = np.asarray(agl_data.compressed(), dtype="float64")
        positive_agl_pixels = int((agl_values > 0).sum())
        if positive_agl_pixels == 0:
            raise RuntimeError(f"{chip_id}: AGL target contains no positive heights")

        return {
            "crs": str(image.crs),
            "width": image.width,
            "height": image.height,
            "resolution_x": abs(float(image.res[0])),
            "resolution_y": abs(float(image.res[1])),
            "building_mask_pixels_checked": building_pixels,
            "positive_agl_pixels_checked": positive_agl_pixels,
        }


def compute_stats(rows: list[dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    """Compute the same all-chip RGB and positive-AGL statistics as the original base dataset."""
    channel_sum = np.zeros(EXPECTED_CHANNELS, dtype="float64")
    channel_sum_sq = np.zeros(EXPECTED_CHANNELS, dtype="float64")
    channel_count = np.zeros(EXPECTED_CHANNELS, dtype="int64")
    agl_sum = 0.0
    agl_sum_sq = 0.0
    agl_count = 0

    for row in rows:
        with rasterio.open(PROJECT_ROOT / row["image_path"]) as image:
            data = image.read(masked=True).astype("float64")
            for band in range(EXPECTED_CHANNELS):
                values = np.asarray(data[band].compressed(), dtype="float64")
                values = values[np.isfinite(values)]
                channel_sum[band] += values.sum()
                channel_sum_sq[band] += np.square(values).sum()
                channel_count[band] += values.size

        with rasterio.open(PROJECT_ROOT / row["agl_path"]) as agl:
            data = agl.read(1, masked=True).astype("float64")
            values = np.asarray(data.compressed(), dtype="float64")
            values = values[np.isfinite(values) & (values > 0) & (values != NODATA_AGL)]
            agl_sum += values.sum()
            agl_sum_sq += np.square(values).sum()
            agl_count += values.size

    if np.any(channel_count == 0) or agl_count == 0:
        raise RuntimeError("Cannot calculate normalization statistics because valid pixels are missing")

    image_mean = channel_sum / channel_count
    image_std = np.sqrt(np.maximum(channel_sum_sq / channel_count - np.square(image_mean), 0))
    if np.any(image_std <= 0):
        raise RuntimeError(f"At least one RGB channel has zero standard deviation: {image_std.tolist()}")
    agl_mean = agl_sum / agl_count
    agl_std = float(np.sqrt(max(agl_sum_sq / agl_count - agl_mean**2, 0)))

    image_stats = {
        "image_mean": image_mean.tolist(),
        "image_std": image_std.tolist(),
        "image_pixel_count_by_channel": channel_count.tolist(),
        "channel_order": ["red", "green", "blue"],
    }
    ndsm_stats = {
        "ndsm_positive_mean": float(agl_mean),
        "ndsm_positive_std": agl_std,
        "ndsm_positive_pixel_count": int(agl_count),
        "target": "building_only_agl_m",
    }
    stats_dir = output_dir / "stats"
    with (stats_dir / "image_stats.pickle").open("wb") as file:
        pickle.dump(image_stats, file)
    with (stats_dir / "ndsm_stats.pickle").open("wb") as file:
        pickle.dump(ndsm_stats, file)
    with (stats_dir / "combined_stats.pickle").open("wb") as file:
        pickle.dump(image_stats | ndsm_stats, file)

    # The upstream HTC loader expects these additional Torch-serialized files.
    torch.save([image_stats["image_mean"], image_stats["image_std"]], output_dir / "image_stats.pickle")
    ndsm_max = float(agl_mean + 6 * agl_std)
    torch.save(
        [float(agl_mean), agl_std, 0.0, ndsm_max, torch.zeros(int(max(1, round(ndsm_max))) + 1)],
        output_dir / "ndsm_stats.pickle",
    )
    return image_stats | ndsm_stats


def main() -> None:
    args = parse_args()
    la_source = resolve_project_path(args.la_source)
    nyc_source = resolve_project_path(args.nyc_source)
    reference_dir = resolve_project_path(args.reference_dir)
    output_dir = resolve_project_path(args.output_dir)

    for name, path in {
        "LA source": la_source,
        "NYC source": nyc_source,
        "reference directory": reference_dir,
    }.items():
        if not path.is_dir():
            raise FileNotFoundError(f"{name} does not exist: {path}")

    splits = read_split_files(reference_dir)
    reference = load_reference(reference_dir, splits)
    source_by_id = load_source(la_source, "los_angeles")
    duplicate_ids = set(source_by_id) & set(load_source(nyc_source, "new_york_city"))
    if duplicate_ids:
        raise RuntimeError(f"LA and NYC source manifests overlap; example: {sorted(duplicate_ids)[0]}")
    source_by_id.update(load_source(nyc_source, "new_york_city"))

    missing_source = set(splits["all"]) - set(source_by_id)
    if missing_source:
        raise RuntimeError(
            f"{len(missing_source)} reference chips are absent from rebuilt sources; example: {sorted(missing_source)[0]}"
        )
    ignored_source_ids = sorted(set(source_by_id) - set(splits["all"]))

    prepare_output(output_dir, args.overwrite)
    split_lookup = {
        chip_id: split
        for split in SPLIT_NAMES
        for chip_id in splits[split]
    }
    manifest_rows: list[dict[str, Any]] = []
    alignment_rows: list[dict[str, Any]] = []

    for chip_id in splits["all"]:
        source = source_by_id[chip_id]
        reference_row = reference[chip_id]
        expected_city = reference_row["source_city"]
        if source["city"] != expected_city:
            raise RuntimeError(
                f"{chip_id}: reference city {expected_city} differs from rebuilt source city {source['city']}"
            )
        for coordinate in ("row_off", "col_off"):
            if str(source["row"][coordinate]) != str(reference_row[coordinate]):
                raise RuntimeError(
                    f"{chip_id}: rebuilt {coordinate}={source['row'][coordinate]} differs from reference "
                    f"{reference_row[coordinate]}"
                )

        image_out = output_dir / "image" / f"{chip_id}_IMG.tif"
        mask_out = output_dir / "mask" / f"{chip_id}_BLG.tif"
        agl_out = output_dir / "ndsm" / f"{chip_id}_AGL.tif"
        transfer_file(source["image"], image_out, args.copy_mode)
        transfer_file(source["mask"], mask_out, args.copy_mode)
        transfer_file(source["agl"], agl_out, args.copy_mode)
        checked = validate_chip(image_out, mask_out, agl_out, chip_id)
        alignment_rows.append({"chip_id": chip_id, "status": "passed", **checked})

        manifest_rows.append(
            {
                "chip_id": chip_id,
                "split": split_lookup[chip_id],
                "source_city": source["city"],
                "source_scene_id": reference_row["source_scene_id"],
                "source_dataset_root": project_relative(la_source if source["city"] == "los_angeles" else nyc_source),
                "row_off": source["row"]["row_off"],
                "col_off": source["row"]["col_off"],
                "positive_agl_pixels": source["row"]["positive_agl_pixels"],
                "building_mask_pixels": source["row"]["building_mask_pixels"],
                "source_image_path": project_relative(source["image"]),
                "source_mask_path": project_relative(source["mask"]),
                "source_agl_path": project_relative(source["agl"]),
                "image_path": project_relative(image_out),
                "mask_path": project_relative(mask_out),
                "agl_path": project_relative(agl_out),
            }
        )

    for split in ALL_SPLIT_NAMES:
        (output_dir / f"{split}.txt").write_text(
            "".join(f"{chip_id}\n" for chip_id in splits[split]), encoding="utf-8"
        )

    manifest_fields = [
        "chip_id", "split", "source_city", "source_scene_id", "source_dataset_root",
        "row_off", "col_off", "positive_agl_pixels", "building_mask_pixels",
        "source_image_path", "source_mask_path", "source_agl_path",
        "image_path", "mask_path", "agl_path",
    ]
    write_csv(output_dir / "chips_manifest.csv", manifest_rows, manifest_fields)
    write_csv(output_dir / "alignment_validation_summary.csv", alignment_rows, list(alignment_rows[0]))
    stats = compute_stats(manifest_rows, output_dir)

    split_city_counts = {
        split: {
            city: sum(
                1
                for chip_id in splits[split]
                if reference[chip_id]["source_city"] == city
            )
            for city in ("los_angeles", "new_york_city")
        }
        for split in SPLIT_NAMES
    }
    summary = {
        "status": "built_and_validated",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "output_dataset": project_relative(output_dir),
        "reference_dataset": project_relative(reference_dir),
        "copy_mode": args.copy_mode,
        "total_chips": len(splits["all"]),
        "split_counts": {split: len(splits[split]) for split in ALL_SPLIT_NAMES},
        "split_city_counts": split_city_counts,
        "ignored_rebuilt_source_chips": ignored_source_ids,
        "alignment_checks_passed": len(alignment_rows),
        "image_stats": stats,
    }
    (output_dir / "dataset_setup_status.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (output_dir / "README.md").write_text(
        "# Windows-Rebuilt NYC + LA Off-Nadir RGB Dataset\n\n"
        "This dataset was rebuilt from city-level aligned chips while preserving the exact "
        "train/validation/test assignment supplied by the migration reference metadata.\n\n"
        f"- Total chips: {len(splits['all'])}\n"
        f"- Train: {len(splits['train'])}\n"
        f"- Validation: {len(splits['val'])}\n"
        f"- Test: {len(splits['test'])}\n"
        f"- Ignored rebuilt source chips not present in the reference: {len(ignored_source_ids)}\n"
        "- Alignment status: passed\n",
        encoding="utf-8",
    )

    print(f"SUCCESS: created {project_relative(output_dir)}")
    print(f"Chips: all={len(splits['all'])}, train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")
    print(f"City counts by split: {split_city_counts}")
    print(f"Ignored rebuilt source chips not in reference: {len(ignored_source_ids)}")
    print("Alignment validation: passed")


if __name__ == "__main__":
    main()
