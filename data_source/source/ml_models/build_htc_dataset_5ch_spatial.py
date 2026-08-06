#!/usr/bin/env python3
"""Build a five-channel, spatially stratified off-nadir HTC dataset.

The output image chips contain red, green, blue, NIR, and the binary building
footprint mask.  Train, validation, and test assignments are made at the 2x2
chip-block level while balancing city-specific building-height distributions.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import pickle
import random
import shutil
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.patches import Rectangle
from skimage.measure import label as connected_components
import torch


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3]
DATA_ROOT = REPO_ROOT / "data_source/data/ml_models/generated/htc_dc_net"
DEFAULT_NIR_DATASET = DATA_ROOT / "nyc_la_off_nadir_rgb_nir_v1"
DEFAULT_MASK_DATASET = DATA_ROOT / "nyc_la_off_nadir_rgb_mask_v1"
DEFAULT_OUTPUT = DATA_ROOT / "nyc_la_off_nadir_rgb_nir_mask_spatial_v1"
SPLITS = ("train", "val", "test")
SPLIT_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}
HEIGHT_BIN_EDGES = np.asarray([0, 10, 20, 30, 40, 50, np.inf], dtype=float)
HEIGHT_BIN_LABELS = ("0_10", "10_20", "20_30", "30_40", "40_50", "50_plus")
CHANNEL_ORDER = ("red", "green", "blue", "nir", "building_footprint_mask")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nir-dataset-dir", type=Path, default=DEFAULT_NIR_DATASET)
    parser.add_argument("--mask-dataset-dir", type=Path, default=DEFAULT_MASK_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--block-chips", type=int, default=2)
    parser.add_argument("--split-search-iterations", type=int, default=50000)
    parser.add_argument("--min-spectral-building-coverage", type=float, default=0.99)
    parser.add_argument("--min-positive-target-coverage", type=float, default=0.95)
    parser.add_argument("--min-component-pixels", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_repo_path(path: Path) -> Path:
    resolved = path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()
    if not resolved.is_relative_to(REPO_ROOT):
        raise ValueError(f"Path is outside the repository: {resolved}")
    return resolved


def relative(path: Path) -> str:
    return resolve_repo_path(path).relative_to(REPO_ROOT).as_posix()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError(f"No rows found in {path}")
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows and fieldnames is None:
        raise RuntimeError(f"Cannot infer columns for empty CSV: {path}")
    columns = fieldnames or list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def grid_signature(src: rasterio.io.DatasetReader) -> tuple[Any, ...]:
    return (src.crs, src.transform, src.width, src.height, src.bounds)


def component_heights(target: np.ndarray, mask: np.ndarray, minimum_pixels: int) -> np.ndarray:
    valid = mask & np.isfinite(target) & (target > 0)
    labels = connected_components(valid, connectivity=1)
    heights = []
    for component_id in range(1, int(labels.max()) + 1):
        component = labels == component_id
        if int(component.sum()) >= minimum_pixels:
            heights.append(float(np.median(target[component])))
    return np.asarray(heights, dtype=float)


def height_category(p90: float, maximum: float) -> str:
    if p90 >= 30 or maximum >= 50:
        return "highrise"
    if p90 >= 10:
        return "midrise"
    return "lowrise"


def inspect_chip(
    nir_row: dict[str, str],
    mask_row: dict[str, str],
    nir_dataset: Path,
    mask_dataset: Path,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[float]]:
    chip_id = nir_row["chip_id"]
    nir_image_path = nir_dataset / "image" / f"{chip_id}_IMG.tif"
    mask_image_path = mask_dataset / "image" / f"{chip_id}_IMG.tif"
    mask_path = nir_dataset / "mask" / f"{chip_id}_BLG.tif"
    agl_path = nir_dataset / "ndsm" / f"{chip_id}_AGL.tif"
    for path in (nir_image_path, mask_image_path, mask_path, agl_path):
        if not path.exists():
            raise FileNotFoundError(path)

    with (
        rasterio.open(nir_image_path) as nir_image,
        rasterio.open(mask_image_path) as mask_image,
        rasterio.open(mask_path) as mask_src,
        rasterio.open(agl_path) as agl_src,
    ):
        if nir_image.count != 4 or mask_image.count != 4:
            raise RuntimeError(f"Expected two four-band source images for {chip_id}")
        signatures = {
            "nir_image": grid_signature(nir_image),
            "mask_image": grid_signature(mask_image),
            "mask": grid_signature(mask_src),
            "agl": grid_signature(agl_src),
        }
        if len(set(signatures.values())) != 1:
            raise RuntimeError(f"Grid mismatch for {chip_id}: {signatures}")
        if nir_image.width != 256 or nir_image.height != 256:
            raise RuntimeError(f"Expected 256x256 chip for {chip_id}")
        if not np.isclose(abs(nir_image.transform.a), 3.0) or not np.isclose(
            abs(nir_image.transform.e), 3.0
        ):
            raise RuntimeError(f"Expected 3 m resolution for {chip_id}: {nir_image.transform}")

        spectral = nir_image.read().astype("float32")
        spectral_valid = np.all(nir_image.read_masks() > 0, axis=0)
        feature_mask = mask_image.read(4).astype("float32")
        building_mask = mask_src.read(1).astype("float32") > 0
        target = agl_src.read(1).astype("float32")
        target_valid = np.isfinite(target) & (target > 0)

        binary_feature = (feature_mask > 0).astype("uint8")
        if not np.array_equal(binary_feature, building_mask.astype("uint8")):
            raise RuntimeError(f"Source feature mask differs from _BLG.tif for {chip_id}")
        if not np.isin(np.unique(binary_feature), [0, 1]).all():
            raise RuntimeError(f"Mask is not binary for {chip_id}")
        building_pixels = int(building_mask.sum())
        if building_pixels == 0:
            raise RuntimeError(f"No building pixels in {chip_id}")

        spectral_coverage = float((spectral_valid & building_mask).sum() / building_pixels)
        target_coverage = float((target_valid & building_mask).sum() / building_pixels)
        heights = component_heights(target, building_mask, args.min_component_pixels)

    exclusion_reasons = []
    if spectral_coverage < args.min_spectral_building_coverage:
        exclusion_reasons.append("insufficient_spectral_building_coverage")
    if target_coverage < args.min_positive_target_coverage:
        exclusion_reasons.append("insufficient_positive_target_coverage")
    if heights.size == 0:
        exclusion_reasons.append("no_eligible_building_components")

    row_off = int(nir_row["row_off"])
    col_off = int(nir_row["col_off"])
    block_pixels = args.block_chips * 256
    bin_counts, _ = np.histogram(heights, bins=HEIGHT_BIN_EDGES) if heights.size else (
        np.zeros(len(HEIGHT_BIN_LABELS), dtype=int),
        HEIGHT_BIN_EDGES,
    )
    building_count = int(heights.size)
    output = {
        "chip_id": chip_id,
        "source_city": nir_row["source_city"],
        "source_scene_id": nir_row["source_scene_id"],
        "row_off": row_off,
        "col_off": col_off,
        "block_row": row_off // block_pixels,
        "block_col": col_off // block_pixels,
        "spatial_block_id": (
            f"{nir_row['source_city']}_block_"
            f"r{row_off // block_pixels:03d}_c{col_off // block_pixels:03d}"
        ),
        "building_mask_pixels": building_pixels,
        "positive_agl_pixels": int((target_valid & building_mask).sum()),
        "spectral_building_coverage": spectral_coverage,
        "positive_target_coverage": target_coverage,
        "building_components": building_count,
        "target_median_m": float(np.median(heights)) if heights.size else float("nan"),
        "target_p90_m": float(np.percentile(heights, 90)) if heights.size else float("nan"),
        "target_max_m": float(np.max(heights)) if heights.size else float("nan"),
        "height_category": (
            height_category(float(np.percentile(heights, 90)), float(np.max(heights)))
            if heights.size
            else "empty"
        ),
        "eligible": not exclusion_reasons,
        "exclusion_reason": ";".join(exclusion_reasons),
        "source_nir_image_path": relative(nir_image_path),
        "source_mask_image_path": relative(mask_image_path),
        "source_mask_path": relative(mask_path),
        "source_agl_path": relative(agl_path),
    }
    for label, count in zip(HEIGHT_BIN_LABELS, bin_counts, strict=True):
        output[f"height_bin_{label}_count"] = int(count)
        output[f"height_bin_{label}_share"] = float(count / building_count) if building_count else 0.0
    return output, heights.tolist()


def block_rows(chip_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in chip_rows:
        grouped.setdefault(row["spatial_block_id"], []).append(row)
    blocks = []
    for block_id, rows in sorted(grouped.items()):
        counts = np.asarray(
            [sum(int(row[f"height_bin_{label}_count"]) for row in rows) for label in HEIGHT_BIN_LABELS],
            dtype=float,
        )
        blocks.append(
            {
                "spatial_block_id": block_id,
                "source_city": rows[0]["source_city"],
                "block_row": rows[0]["block_row"],
                "block_col": rows[0]["block_col"],
                "chips": len(rows),
                "buildings": int(counts.sum()),
                "median_sum": float(sum(row["target_median_m"] for row in rows)),
                "p90_sum": float(sum(row["target_p90_m"] for row in rows)),
                "logmax_sum": float(sum(np.log1p(row["target_max_m"]) for row in rows)),
                "category_lowrise": int(any(row["height_category"] == "lowrise" for row in rows)),
                "category_midrise": int(any(row["height_category"] == "midrise" for row in rows)),
                "category_highrise": int(any(row["height_category"] == "highrise" for row in rows)),
                **{f"height_bin_{label}_count": int(count) for label, count in zip(HEIGHT_BIN_LABELS, counts, strict=True)},
            }
        )
    return blocks


def split_score(assignments: dict[str, str], blocks: list[dict[str, Any]]) -> float:
    score = 0.0
    for city in sorted({row["source_city"] for row in blocks}):
        city_blocks = [row for row in blocks if row["source_city"] == city]
        total_chips = sum(row["chips"] for row in city_blocks)
        total_buildings = sum(row["buildings"] for row in city_blocks)
        global_bins = np.asarray(
            [sum(row[f"height_bin_{label}_count"] for row in city_blocks) for label in HEIGHT_BIN_LABELS],
            dtype=float,
        )
        global_bin_share = global_bins / max(global_bins.sum(), 1)
        global_feature = np.asarray(
            [
                sum(row["median_sum"] for row in city_blocks) / total_chips,
                sum(row["p90_sum"] for row in city_blocks) / total_chips,
                sum(row["logmax_sum"] for row in city_blocks) / total_chips,
            ]
        )
        feature_scale = np.maximum(np.abs(global_feature), 1.0)
        for split in SPLITS:
            selected = [row for row in city_blocks if assignments[row["spatial_block_id"]] == split]
            if not selected:
                return float("inf")
            chips = sum(row["chips"] for row in selected)
            buildings = sum(row["buildings"] for row in selected)
            bins = np.asarray(
                [sum(row[f"height_bin_{label}_count"] for row in selected) for label in HEIGHT_BIN_LABELS],
                dtype=float,
            )
            bin_share = bins / max(bins.sum(), 1)
            features = np.asarray(
                [
                    sum(row["median_sum"] for row in selected) / chips,
                    sum(row["p90_sum"] for row in selected) / chips,
                    sum(row["logmax_sum"] for row in selected) / chips,
                ]
            )
            target = SPLIT_FRACTIONS[split]
            score += 20 * (chips / total_chips - target) ** 2
            score += 8 * (buildings / max(total_buildings, 1) - target) ** 2
            score += float(np.mean((bin_share - global_bin_share) ** 2 / (global_bin_share + 0.02)))
            score += 0.5 * float(np.mean(((features - global_feature) / feature_scale) ** 2))
    return score


def assign_spatial_blocks(
    blocks: list[dict[str, Any]], seed: int, iterations: int
) -> tuple[dict[str, str], float]:
    rng = random.Random(seed)
    city_blocks = {
        city: sorted([row for row in blocks if row["source_city"] == city], key=lambda row: row["spatial_block_id"])
        for city in sorted({row["source_city"] for row in blocks})
    }
    split_counts: dict[str, dict[str, int]] = {}
    for city, rows in city_blocks.items():
        count = len(rows)
        val_count = max(4, int(round(count * SPLIT_FRACTIONS["val"])))
        test_count = max(4, int(round(count * SPLIT_FRACTIONS["test"])))
        if val_count + test_count >= count:
            raise RuntimeError(f"Not enough {city} blocks for required validation/test blocks")
        split_counts[city] = {"train": count - val_count - test_count, "val": val_count, "test": test_count}

    best_assignment: dict[str, str] | None = None
    best_score = float("inf")
    for _ in range(iterations):
        candidate: dict[str, str] = {}
        feasible = True
        for city, rows in city_blocks.items():
            shuffled = rows[:]
            rng.shuffle(shuffled)
            cursor = 0
            for split in SPLITS:
                take = split_counts[city][split]
                for row in shuffled[cursor : cursor + take]:
                    candidate[row["spatial_block_id"]] = split
                cursor += take
            for split in SPLITS:
                selected = [row for row in rows if candidate[row["spatial_block_id"]] == split]
                for category in ("lowrise", "midrise", "highrise"):
                    available = sum(row[f"category_{category}"] for row in rows)
                    if available >= len(SPLITS) and not any(row[f"category_{category}"] for row in selected):
                        feasible = False
        if not feasible:
            continue
        score = split_score(candidate, blocks)
        if score < best_score:
            best_score = score
            best_assignment = candidate.copy()
    if best_assignment is None:
        raise RuntimeError("No feasible spatial split assignment was found")
    return best_assignment, best_score


def create_output_chip(row: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    chip_id = row["chip_id"]
    image_out = output_dir / "image" / f"{chip_id}_IMG.tif"
    mask_out = output_dir / "mask" / f"{chip_id}_BLG.tif"
    agl_out = output_dir / "ndsm" / f"{chip_id}_AGL.tif"
    with rasterio.open(REPO_ROOT / row["source_nir_image_path"]) as nir_src:
        spectral = nir_src.read().astype("uint16")
        profile = nir_src.profile.copy()
    with rasterio.open(REPO_ROOT / row["source_mask_path"]) as mask_src:
        mask = (mask_src.read(1) > 0).astype("uint16")
    stacked = np.concatenate([spectral, mask[None, :, :]], axis=0)
    profile.update(count=5, dtype="uint16", nodata=None, compress="deflate", tiled=True)
    profile.pop("photometric", None)
    with rasterio.open(image_out, "w", **profile) as dst:
        dst.write(stacked)
        for band, description in enumerate(CHANNEL_ORDER, start=1):
            dst.set_band_description(band, description)
        dst.update_tags(
            variant="rgb_nir_mask_spatial",
            channel_order=";".join(CHANNEL_ORDER),
            split=row["split"],
            spatial_block_id=row["spatial_block_id"],
        )
    shutil.copy2(REPO_ROOT / row["source_mask_path"], mask_out)
    shutil.copy2(REPO_ROOT / row["source_agl_path"], agl_out)
    output = row.copy()
    output.update(
        {
            "variant": "rgb_nir_mask_spatial",
            "input_channels": 5,
            "channel_order": ";".join(CHANNEL_ORDER),
            "image_path": relative(image_out),
            "mask_path": relative(mask_out),
            "agl_path": relative(agl_out),
        }
    )
    return output


def compute_training_stats(rows: list[dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    training_rows = [row for row in rows if row["split"] == "train"]
    sums = np.zeros(5, dtype="float64")
    sums_sq = np.zeros(5, dtype="float64")
    counts = np.zeros(5, dtype="int64")
    target_values = []
    for row in training_rows:
        with rasterio.open(REPO_ROOT / row["image_path"]) as src:
            image = src.read().astype("float64")
        for band in range(5):
            valid = np.isfinite(image[band])
            if band < 4:
                valid &= image[band] > 0
            values = image[band][valid]
            sums[band] += values.sum()
            sums_sq[band] += np.square(values).sum()
            counts[band] += values.size
        with rasterio.open(REPO_ROOT / row["agl_path"]) as src:
            target = src.read(1, masked=True).astype("float64")
        values = target.compressed()
        target_values.append(values[np.isfinite(values) & (values > 0)])
    if np.any(counts == 0):
        raise RuntimeError(f"Empty normalization channel: counts={counts.tolist()}")
    means = sums / counts
    stds = np.sqrt(np.maximum(sums_sq / counts - means**2, 0))
    stds[stds == 0] = 1.0
    targets = np.concatenate(target_values)
    target_mean = float(np.mean(targets))
    target_std = float(np.std(targets))
    target_max_for_stats = float(target_mean + 6 * target_std)
    count_tensor = torch.zeros(int(max(1, round(target_max_for_stats))) + 1)
    image_stats = {
        "image_mean": means.tolist(),
        "image_std": stds.tolist(),
        "image_pixel_count_by_channel": counts.tolist(),
        "channel_order": list(CHANNEL_ORDER),
        "stats_split": "train",
        "training_chips": len(training_rows),
    }
    ndsm_stats = {
        "ndsm_positive_mean": target_mean,
        "ndsm_positive_std": target_std,
        "ndsm_positive_pixel_count": int(targets.size),
        "stats_split": "train",
    }
    with (output_dir / "stats/image_stats.pickle").open("wb") as stream:
        pickle.dump(image_stats, stream)
    with (output_dir / "stats/ndsm_stats.pickle").open("wb") as stream:
        pickle.dump(ndsm_stats, stream)
    torch.save([image_stats["image_mean"], image_stats["image_std"]], output_dir / "image_stats.pickle")
    torch.save(
        [target_mean, target_std, 0.0, target_max_for_stats, count_tensor],
        output_dir / "ndsm_stats.pickle",
    )
    return image_stats | ndsm_stats


def validate_output_chip(row: dict[str, Any]) -> dict[str, Any]:
    image_path = REPO_ROOT / row["image_path"]
    mask_path = REPO_ROOT / row["mask_path"]
    agl_path = REPO_ROOT / row["agl_path"]
    with rasterio.open(image_path) as image, rasterio.open(mask_path) as mask, rasterio.open(agl_path) as agl:
        if image.count != 5 or image.descriptions != CHANNEL_ORDER:
            raise RuntimeError(f"Invalid five-band contract for {image_path}: {image.descriptions}")
        if grid_signature(image) != grid_signature(mask) or grid_signature(image) != grid_signature(agl):
            raise RuntimeError(f"Output grid mismatch for {row['chip_id']}")
        feature_mask = image.read(5)
        separate_mask = (mask.read(1) > 0).astype(feature_mask.dtype)
        if not np.array_equal(feature_mask, separate_mask):
            raise RuntimeError(f"Band 5 differs from building mask for {row['chip_id']}")
        if not np.isin(np.unique(feature_mask), [0, 1]).all():
            raise RuntimeError(f"Non-binary feature mask for {row['chip_id']}")
        nir = image.read(4)
        if not np.isfinite(nir).any() or float(np.max(nir)) <= 0:
            raise RuntimeError(f"Empty NIR channel for {row['chip_id']}")
        return {
            "chip_id": row["chip_id"],
            "split": row["split"],
            "source_city": row["source_city"],
            "spatial_block_id": row["spatial_block_id"],
            "input_channels": image.count,
            "width": image.width,
            "height": image.height,
            "resolution_x_m": abs(image.transform.a),
            "resolution_y_m": abs(image.transform.e),
            "crs": str(image.crs),
            "grid_match": True,
            "mask_band_match": True,
            "mask_binary": True,
            "nir_nonempty": True,
            "spectral_building_coverage": row["spectral_building_coverage"],
            "positive_target_coverage": row["positive_target_coverage"],
            "validation_status": "passed",
        }


def split_summary(rows: list[dict[str, Any]], heights_by_chip: dict[str, list[float]]) -> list[dict[str, Any]]:
    summaries = []
    for city in ("all", "los_angeles", "new_york_city"):
        for split in SPLITS:
            selected = [row for row in rows if row["split"] == split and (city == "all" or row["source_city"] == city)]
            heights = np.asarray(
                [height for row in selected for height in heights_by_chip[row["chip_id"]]], dtype=float
            )
            counts, _ = np.histogram(heights, bins=HEIGHT_BIN_EDGES)
            result = {
                "group": city,
                "split": split,
                "chips": len(selected),
                "spatial_blocks": len({row["spatial_block_id"] for row in selected}),
                "buildings": int(heights.size),
                "mean_m": float(np.mean(heights)),
                "median_m": float(np.median(heights)),
                "p90_m": float(np.percentile(heights, 90)),
                "p95_m": float(np.percentile(heights, 95)),
                "p99_m": float(np.percentile(heights, 99)),
                "max_m": float(np.max(heights)),
            }
            for label, count in zip(HEIGHT_BIN_LABELS, counts, strict=True):
                result[f"share_{label}_pct"] = float(100 * count / heights.size)
            summaries.append(result)
    return summaries


def plot_height_distributions(
    rows: list[dict[str, Any]], heights_by_chip: dict[str, list[float]], output_path: Path
) -> None:
    colors = {"train": "#2878a0", "val": "#dc7f2a", "test": "#4b9560"}
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for row_index, city in enumerate(("los_angeles", "new_york_city")):
        for split in SPLITS:
            heights = np.asarray(
                [height for row in rows if row["split"] == split and row["source_city"] == city for height in heights_by_chip[row["chip_id"]]],
                dtype=float,
            )
            axes[row_index, 0].hist(
                heights,
                bins=np.linspace(0, 150, 76),
                density=True,
                histtype="step",
                linewidth=2,
                color=colors[split],
                label=split,
            )
            counts, _ = np.histogram(heights, bins=HEIGHT_BIN_EDGES)
            positions = np.arange(len(HEIGHT_BIN_LABELS))
            offset = {"train": -0.25, "val": 0.0, "test": 0.25}[split]
            axes[row_index, 1].bar(
                positions + offset,
                100 * counts / counts.sum(),
                width=0.25,
                color=colors[split],
                label=split,
            )
        axes[row_index, 0].set_title(f"{city}: normalized distribution")
        axes[row_index, 0].set_xlim(0, 150)
        axes[row_index, 0].set_xlabel("LiDAR-derived building height (m)")
        axes[row_index, 0].set_ylabel("Density")
        axes[row_index, 0].legend(frameon=False)
        axes[row_index, 1].set_title(f"{city}: height-bin composition")
        axes[row_index, 1].set_xticks(
            np.arange(len(HEIGHT_BIN_LABELS)),
            [label.replace("_", "-").replace("-plus", "+") + " m" for label in HEIGHT_BIN_LABELS],
            rotation=25,
        )
        axes[row_index, 1].set_ylabel("Buildings (%)")
        axes[row_index, 1].legend(frameon=False)
    fig.suptitle("Spatially Stratified LiDAR Height Distributions", fontsize=16, fontweight="bold")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_spatial_blocks(blocks: list[dict[str, Any]], output_path: Path) -> None:
    colors = {"train": "#2878a0", "val": "#dc7f2a", "test": "#4b9560"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), constrained_layout=True)
    for ax, city in zip(axes, ("los_angeles", "new_york_city"), strict=True):
        city_rows = [row for row in blocks if row["source_city"] == city]
        for row in city_rows:
            ax.add_patch(
                Rectangle(
                    (row["block_col"], -row["block_row"] - 1),
                    1,
                    1,
                    facecolor=colors[row["split"]],
                    edgecolor="white",
                    linewidth=1,
                )
            )
            ax.text(
                row["block_col"] + 0.5,
                -row["block_row"] - 0.5,
                str(row["chips"]),
                ha="center",
                va="center",
                fontsize=7,
            )
        ax.autoscale_view()
        ax.set_aspect("equal")
        ax.set_title(city.replace("_", " ").title())
        ax.set_xlabel("2x2 block column")
        ax.set_ylabel("2x2 block row")
    handles = [Rectangle((0, 0), 1, 1, color=colors[split]) for split in SPLITS]
    fig.legend(handles, SPLITS, loc="upper center", ncol=3, frameon=False)
    fig.suptitle("Spatial Block Assignment", fontsize=16, fontweight="bold")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def write_readme(
    output_dir: Path,
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    exclusions: list[dict[str, Any]],
    stats: dict[str, Any],
    score: float,
) -> None:
    counts = {split: sum(row["split"] == split for row in rows) for split in SPLITS}
    city_counts = {
        city: sum(row["source_city"] == city for row in rows)
        for city in ("los_angeles", "new_york_city")
    }
    text = f"""# NYC + LA Off-Nadir RGB+NIR+Mask Spatial Dataset

Created: {datetime.now(timezone.utc).isoformat()}

This HTC-DC Net dataset has five channels in the fixed order:

```text
{';'.join(CHANNEL_ORDER)}
```

Chips are grouped into `{args.block_chips}x{args.block_chips}` spatial blocks and assigned as whole blocks to train, validation, or test. The deterministic assignment uses seed `{args.seed}` and balances city-specific chip counts, building counts, medians, P90s, maxima, and height-bin shares.

| Split | Chips |
|---|---:|
| Train | {counts['train']} |
| Validation | {counts['val']} |
| Test | {counts['test']} |
| Total | {len(rows)} |

| City | Chips |
|---|---:|
| Los Angeles | {city_counts['los_angeles']} |
| New York City | {city_counts['new_york_city']} |

Excluded chips: `{len(exclusions)}`  
Split objective score: `{score:.8f}`

Normalization statistics use training chips only:

```text
image_mean = {stats['image_mean']}
image_std = {stats['image_std']}
```

Alignment requirements are recorded per chip in `alignment_validation_summary.csv`.
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    nir_dataset = resolve_repo_path(args.nir_dataset_dir)
    mask_dataset = resolve_repo_path(args.mask_dataset_dir)
    output_dir = resolve_repo_path(args.output_dir)
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_dir} exists; use --overwrite to rebuild it")
        shutil.rmtree(output_dir)

    nir_rows = {row["chip_id"]: row for row in read_csv(nir_dataset / "chips_manifest.csv")}
    mask_rows = {row["chip_id"]: row for row in read_csv(mask_dataset / "chips_manifest.csv")}
    if set(nir_rows) != set(mask_rows):
        raise RuntimeError("RGB+NIR and RGB+mask datasets do not contain identical chip IDs")

    inspected = []
    heights_by_chip: dict[str, list[float]] = {}
    for chip_id in sorted(nir_rows):
        row, heights = inspect_chip(nir_rows[chip_id], mask_rows[chip_id], nir_dataset, mask_dataset, args)
        inspected.append(row)
        heights_by_chip[chip_id] = heights
    eligible = [row for row in inspected if row["eligible"]]
    exclusions = [row for row in inspected if not row["eligible"]]
    if not eligible:
        raise RuntimeError("All chips were excluded")

    blocks = block_rows(eligible)
    assignments, objective_score = assign_spatial_blocks(
        blocks, seed=args.seed, iterations=args.split_search_iterations
    )
    for row in eligible:
        row["split"] = assignments[row["spatial_block_id"]]
    for row in blocks:
        row["split"] = assignments[row["spatial_block_id"]]
        row["split_objective_score"] = objective_score

    for name in ("image", "mask", "ndsm", "stats"):
        (output_dir / name).mkdir(parents=True, exist_ok=True)
    output_rows = [create_output_chip(row, output_dir) for row in eligible]
    for split in SPLITS:
        ids = sorted(row["chip_id"] for row in output_rows if row["split"] == split)
        (output_dir / f"{split}.txt").write_text("".join(f"{chip_id}\n" for chip_id in ids), encoding="utf-8")
    all_ids = sorted(row["chip_id"] for row in output_rows)
    (output_dir / "all.txt").write_text("".join(f"{chip_id}\n" for chip_id in all_ids), encoding="utf-8")

    stats = compute_training_stats(output_rows, output_dir)
    validation_rows = [validate_output_chip(row) for row in output_rows]
    summary_rows = split_summary(output_rows, heights_by_chip)
    write_csv(output_dir / "chips_manifest.csv", output_rows)
    write_csv(output_dir / "spatial_block_assignments.csv", blocks)
    write_csv(output_dir / "split_height_summary.csv", summary_rows)
    write_csv(
        output_dir / "excluded_chips.csv",
        exclusions,
        fieldnames=list(inspected[0]),
    )
    write_csv(output_dir / "alignment_validation_summary.csv", validation_rows)
    plot_height_distributions(
        output_rows,
        heights_by_chip,
        output_dir / "split_height_distributions.png",
    )
    plot_spatial_blocks(blocks, output_dir / "spatial_split_map.png")
    write_readme(output_dir, args, output_rows, exclusions, stats, objective_score)
    (output_dir / "dataset_build_metadata.json").write_text(
        json.dumps(
            {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "seed": args.seed,
                "block_chips": args.block_chips,
                "split_fractions": SPLIT_FRACTIONS,
                "split_search_iterations": args.split_search_iterations,
                "split_objective_score": objective_score,
                "eligible_chips": len(output_rows),
                "excluded_chips": len(exclusions),
                "normalization_stats_split": "train",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote five-channel dataset: {relative(output_dir)}")
    print(f"Eligible chips: {len(output_rows)}; excluded chips: {len(exclusions)}")
    print(
        "Splits: "
        + ", ".join(f"{split}={sum(row['split'] == split for row in output_rows)}" for split in SPLITS)
    )
    print(f"Spatial split objective: {objective_score:.8f}")


if __name__ == "__main__":
    main()
