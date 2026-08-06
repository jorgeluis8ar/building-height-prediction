#!/usr/bin/env python3
"""Build a multi-scene NYC+LA HTC-DC Net dataset.

The script starts from the existing 3-channel HTC dataset and stacks RGB
channels from additional PlanetScope scenes onto the same chip grid.  The
default configuration creates the first 12-channel dataset:

    4 PlanetScope scenes x 3 RGB bands = 12 input channels

The output keeps the same HTC folder contract used by the previous RGB and
6-channel datasets:

    image/   *_IMG.tif
    mask/    *_BLG.tif
    ndsm/    *_AGL.tif
    stats/   normalization metadata
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
import pickle
import random
import shutil
from typing import Any

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_BASE_DATASET_DIR = (
    PROJECT_ROOT / "data_source/data/ml_models/generated/htc_dc_net/nyc_la_rgb_v1"
)
DEFAULT_SCENE_REVIEW = (
    PROJECT_ROOT
    / "data_source/data/planet_imagery/generated/intermediate_sun_elevation_scene_review.csv"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "data_source/data/ml_models/generated/htc_dc_net/nyc_la_12ch_v1"
)
DEFAULT_SPLIT_TEMPLATE_DIR = (
    PROJECT_ROOT / "data_source/data/ml_models/generated/htc_dc_net/nyc_la_6ch_v1"
)
PLANET_SOURCE_DIR = PROJECT_ROOT / "data_source/data/planet_imagery/source"
DEFAULT_SEED = 20260716


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dataset-dir", type=Path, default=DEFAULT_BASE_DATASET_DIR)
    parser.add_argument("--scene-review", type=Path, default=DEFAULT_SCENE_REVIEW)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--split-template-dir",
        type=Path,
        default=DEFAULT_SPLIT_TEMPLATE_DIR,
        help=(
            "Optional existing HTC dataset whose train/val/test/all split files "
            "should be reused. Defaults to nyc_la_6ch_v1 for comparability."
        ),
    )
    parser.add_argument("--scene-count", type=int, default=4)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_project_path(path: Path) -> Path:
    resolved = path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()
    if not resolved.is_relative_to(PROJECT_ROOT):
        raise ValueError(f"Path is outside the project repository: {resolved}")
    return resolved


def rel(path: Path) -> str:
    return resolve_project_path(path).relative_to(PROJECT_ROOT).as_posix()


def ensure_clean_output(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output exists: {output_dir}. Pass --overwrite.")
        shutil.rmtree(output_dir)
    for name in ["image", "mask", "ndsm", "stats"]:
        (output_dir / name).mkdir(parents=True, exist_ok=True)


def read_manifest(base_dataset_dir: Path) -> list[dict[str, Any]]:
    manifest_path = base_dataset_dir / "chips_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    with manifest_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise RuntimeError(f"No rows found in {manifest_path}")
    return rows


def read_scene_review(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing scene review CSV: {path}. Run select_intermediate_sun_elevation_scenes.py first."
        )
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise RuntimeError(f"No rows found in {path}")
    required = {"city_slug", "id", "scene_role", "acquired_date", "sun_elevation"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"Scene review CSV is missing columns: {sorted(missing)}")
    return rows


def source_raster_for_scene(city_slug: str, scene_id: str) -> Path:
    """Find the downloaded clipped PlanetScope SR raster for one city/scene."""
    city_dir = PLANET_SOURCE_DIR / city_slug
    matches = sorted(
        path
        for path in city_dir.glob(f"**/{scene_id}_3B_AnalyticMS_SR*_clip.tif")
        if "udm" not in path.name.lower()
    )
    if not matches:
        raise FileNotFoundError(
            f"Missing downloaded Planet SR clip for {city_slug} scene {scene_id}. "
            "Wait for the Planet order to succeed, then run download_ordered_planet_scenes.py."
        )
    if len(matches) > 1:
        raise RuntimeError(f"Found multiple Planet SR clips for {city_slug} scene {scene_id}: {matches}")
    return matches[0]


def rgb_band_indexes(path: Path) -> list[int]:
    """Return source band indexes for RGB in Planet 4-band or 8-band SR imagery."""
    with rasterio.open(path) as src:
        descriptions = [desc.lower() if desc else "" for desc in src.descriptions]
        if {"red", "green", "blue"}.issubset(set(descriptions)):
            return [
                descriptions.index("red") + 1,
                descriptions.index("green") + 1,
                descriptions.index("blue") + 1,
            ]
        if src.count == 4:
            return [3, 2, 1]
        if src.count == 8:
            return [6, 4, 2]
    raise RuntimeError(f"Cannot infer RGB bands for {path}")


def scene_plan_for_city(
    city_slug: str,
    base_scene_id: str,
    scene_rows: list[dict[str, Any]],
    scene_count: int,
) -> list[dict[str, Any]]:
    """Build the fixed channel plan for one city.

    The base RGB chip scene stays first because it defines the chip grid and
    keeps compatibility with the existing dataset.  The other existing scene
    comes second, followed by the newly selected intermediate scenes in date
    order.
    """
    city_rows = [row for row in scene_rows if row["city_slug"] == city_slug]
    if not city_rows:
        raise RuntimeError(f"No scene-review rows found for {city_slug}")
    by_id = {row["id"]: row for row in city_rows}
    if base_scene_id not in by_id:
        raise RuntimeError(
            f"Base scene {base_scene_id} for {city_slug} is missing from the scene review CSV."
        )

    base = by_id[base_scene_id].copy()
    base["channel_source"] = "base_chip"
    other_existing = sorted(
        [
            row.copy()
            for row in city_rows
            if row["scene_role"] == "existing_6ch_scene" and row["id"] != base_scene_id
        ],
        key=lambda row: row["acquired_date"],
    )
    new_intermediate = sorted(
        [row.copy() for row in city_rows if row["scene_role"] == "new_intermediate_scene"],
        key=lambda row: row["acquired_date"],
    )
    plan = [base] + other_existing + new_intermediate
    if len(plan) != scene_count:
        raise RuntimeError(
            f"Expected {scene_count} scenes for {city_slug}; got {len(plan)}. "
            "Check the scene review CSV and the requested --scene-count."
        )
    for order, row in enumerate(plan, start=1):
        row["model_scene_order"] = order
        row["channel_prefix"] = f"scene_{order:02d}_{row['id']}"
        if row.get("channel_source") != "base_chip":
            row["channel_source"] = "planet_source_raster"
            row["source_raster_path"] = rel(source_raster_for_scene(city_slug, row["id"]))
            row["rgb_source_band_indexes"] = ",".join(map(str, rgb_band_indexes(PROJECT_ROOT / row["source_raster_path"])))
        else:
            row["source_raster_path"] = ""
            row["rgb_source_band_indexes"] = "1,2,3"
    return plan


def read_reprojected_rgb(
    source_path: Path,
    rgb_indexes: list[int],
    template_profile: dict[str, Any],
) -> np.ndarray:
    """Read RGB from a full-scene raster and align it to one chip grid."""
    output = np.zeros((3, template_profile["height"], template_profile["width"]), dtype="float32")
    with rasterio.open(source_path) as src:
        for out_idx, src_idx in enumerate(rgb_indexes):
            reproject(
                source=rasterio.band(src, src_idx),
                destination=output[out_idx],
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=src.nodata,
                dst_transform=template_profile["transform"],
                dst_crs=template_profile["crs"],
                dst_nodata=0,
                resampling=Resampling.bilinear,
            )
    return output


def write_multiscene_chip(
    row: dict[str, Any],
    city_plan: list[dict[str, Any]],
    base_dataset_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    chip_id = row["chip_id"]
    image_in = PROJECT_ROOT / row["image_path"]
    mask_in = PROJECT_ROOT / row["mask_path"]
    agl_in = PROJECT_ROOT / row["agl_path"]
    for path in [image_in, mask_in, agl_in]:
        if not path.exists():
            raise FileNotFoundError(path)

    image_out = output_dir / "image" / f"{chip_id}_IMG.tif"
    mask_out = output_dir / "mask" / f"{chip_id}_BLG.tif"
    agl_out = output_dir / "ndsm" / f"{chip_id}_AGL.tif"

    stacked_parts = []
    with rasterio.open(image_in) as src:
        primary_rgb = src.read().astype("float32")
        if primary_rgb.shape[0] != 3:
            raise RuntimeError(f"Expected 3 bands in base chip {image_in}; found {primary_rgb.shape[0]}")
        profile = src.profile.copy()
        template_profile = {
            "height": src.height,
            "width": src.width,
            "transform": src.transform,
            "crs": src.crs,
        }
    stacked_parts.append(primary_rgb)

    for scene in city_plan[1:]:
        source_path = PROJECT_ROOT / scene["source_raster_path"]
        rgb_indexes = [int(value) for value in scene["rgb_source_band_indexes"].split(",")]
        stacked_parts.append(read_reprojected_rgb(source_path, rgb_indexes, template_profile))

    stacked = np.concatenate(stacked_parts, axis=0)
    expected_bands = len(city_plan) * 3
    if stacked.shape[0] != expected_bands:
        raise RuntimeError(f"Expected {expected_bands} bands for {chip_id}; got {stacked.shape[0]}")

    descriptions = []
    for scene in city_plan:
        descriptions.extend(
            [
                f"{scene['channel_prefix']}_red",
                f"{scene['channel_prefix']}_green",
                f"{scene['channel_prefix']}_blue",
            ]
        )

    profile.update(
        driver="GTiff",
        count=expected_bands,
        dtype="uint16",
        nodata=0,
        compress="deflate",
        tiled=True,
        blockxsize=256,
        blockysize=256,
        BIGTIFF="IF_SAFER",
    )
    profile.pop("photometric", None)
    with rasterio.open(image_out, "w", **profile) as dst:
        dst.write(np.clip(np.rint(stacked), 0, np.iinfo(np.uint16).max).astype("uint16"))
        for band_idx, description in enumerate(descriptions, start=1):
            dst.set_band_description(band_idx, description)
        dst.update_tags(
            scene_ids=",".join(scene["id"] for scene in city_plan),
            scene_acquired_dates=",".join(scene["acquired_date"] for scene in city_plan),
            scene_sun_elevations=",".join(str(scene["sun_elevation"]) for scene in city_plan),
            channel_order=";".join(descriptions),
        )

    shutil.copy2(mask_in, mask_out)
    shutil.copy2(agl_in, agl_out)

    out_row = row.copy()
    out_row.update(
        {
            "source_dataset_root": rel(base_dataset_dir),
            "scene_count": len(city_plan),
            "input_channels": expected_bands,
            "scene_ids": ",".join(scene["id"] for scene in city_plan),
            "scene_acquired_dates": ",".join(scene["acquired_date"] for scene in city_plan),
            "scene_sun_elevations": ",".join(str(scene["sun_elevation"]) for scene in city_plan),
            "channel_order": ";".join(descriptions),
            "source_scene_raster_paths": ";".join(scene.get("source_raster_path", "") for scene in city_plan),
            "image_path": rel(image_out),
            "mask_path": rel(mask_out),
            "agl_path": rel(agl_out),
        }
    )
    return out_row


def split_chip_ids(chip_ids: list[str], seed: int) -> dict[str, list[str]]:
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


def read_template_splits(template_dir: Path, chip_ids: set[str]) -> dict[str, list[str]]:
    """Reuse split files from an existing dataset when all chip IDs match."""
    splits: dict[str, list[str]] = {}
    for split in ["train", "val", "test", "all"]:
        path = template_dir / f"{split}.txt"
        if not path.exists():
            raise FileNotFoundError(path)
        values = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not values:
            raise RuntimeError(f"Template split is empty: {path}")
        unknown = sorted(set(values) - chip_ids)
        if unknown:
            raise RuntimeError(
                f"Template split {path} contains chip IDs not present in the new dataset: {unknown[:5]}"
            )
        splits[split] = values

    combined = set(splits["train"]) | set(splits["val"]) | set(splits["test"])
    if combined != chip_ids:
        missing = sorted(chip_ids - combined)
        extra = sorted(combined - chip_ids)
        raise RuntimeError(
            "Template train/val/test split does not exactly cover the new dataset. "
            f"Missing={missing[:5]}, extra={extra[:5]}"
        )
    if set(splits["all"]) != chip_ids:
        raise RuntimeError("Template all.txt does not match the new dataset chip IDs.")
    return splits


def write_split_files(output_dir: Path, splits: dict[str, list[str]]) -> None:
    for split, chip_ids in splits.items():
        (output_dir / f"{split}.txt").write_text(
            "".join(f"{chip_id}\n" for chip_id in chip_ids),
            encoding="utf-8",
        )


def compute_stats(rows: list[dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    channel_sum = None
    channel_sum_sq = None
    channel_count = None
    agl_sum = 0.0
    agl_sum_sq = 0.0
    agl_count = 0
    channel_order = None

    for row in rows:
        with rasterio.open(PROJECT_ROOT / row["image_path"]) as image:
            data = image.read(masked=True).astype("float64")
            mask = np.ma.getmaskarray(data)
            if channel_sum is None:
                channel_sum = np.zeros(data.shape[0], dtype="float64")
                channel_sum_sq = np.zeros(data.shape[0], dtype="float64")
                channel_count = np.zeros(data.shape[0], dtype="int64")
                channel_order = list(image.descriptions)
            for band in range(data.shape[0]):
                valid = (~mask[band]) & (np.ma.filled(data[band], 0) > 0)
                vals = np.asarray(data[band][valid], dtype="float64")
                channel_sum[band] += vals.sum()
                channel_sum_sq[band] += np.square(vals).sum()
                channel_count[band] += vals.size

        with rasterio.open(PROJECT_ROOT / row["agl_path"]) as agl:
            target = agl.read(1, masked=True).astype("float64")
            valid_agl = (target > 0) & ~np.ma.getmaskarray(target)
            vals = np.asarray(target[valid_agl], dtype="float64")
            agl_sum += vals.sum()
            agl_sum_sq += np.square(vals).sum()
            agl_count += vals.size

    if channel_sum is None or channel_count is None or np.any(channel_count == 0):
        raise RuntimeError("At least one image channel has no valid pixels for normalization stats.")
    if agl_count == 0:
        raise RuntimeError("No positive AGL pixels were available for stats.")

    image_mean = channel_sum / channel_count
    image_std = np.sqrt(np.maximum(channel_sum_sq / channel_count - image_mean**2, 0))
    image_std = np.where(image_std == 0, 1.0, image_std)
    agl_mean = agl_sum / agl_count
    agl_std = np.sqrt(max(agl_sum_sq / agl_count - agl_mean**2, 0))
    ndsm_max = float(agl_mean + 6 * agl_std)
    count = torch.zeros(int(max(1, round(ndsm_max))) + 1)

    image_stats = {
        "image_mean": image_mean.tolist(),
        "image_std": image_std.tolist(),
        "image_pixel_count_by_channel": channel_count.tolist(),
        "channel_order": channel_order,
    }
    ndsm_stats = {
        "ndsm_positive_mean": float(agl_mean),
        "ndsm_positive_std": float(agl_std),
        "ndsm_positive_pixel_count": int(agl_count),
        "target": "building_only_agl_m",
    }
    with (output_dir / "stats/image_stats.pickle").open("wb") as file:
        pickle.dump(image_stats, file)
    with (output_dir / "stats/ndsm_stats.pickle").open("wb") as file:
        pickle.dump(ndsm_stats, file)
    with (output_dir / "stats/combined_stats.pickle").open("wb") as file:
        pickle.dump(image_stats | ndsm_stats, file)

    torch.save([image_stats["image_mean"], image_stats["image_std"]], output_dir / "image_stats.pickle")
    torch.save(
        [
            ndsm_stats["ndsm_positive_mean"],
            ndsm_stats["ndsm_positive_std"],
            0.0,
            ndsm_max,
            count,
        ],
        output_dir / "ndsm_stats.pickle",
    )
    return image_stats | ndsm_stats


def write_manifest(output_dir: Path, rows: list[dict[str, Any]], splits: dict[str, list[str]]) -> None:
    split_lookup = {
        chip_id: split
        for split, chip_ids in splits.items()
        if split != "all"
        for chip_id in chip_ids
    }
    fieldnames = [
        "chip_id",
        "split",
        "source_city",
        "source_scene_id",
        "scene_count",
        "input_channels",
        "scene_ids",
        "scene_acquired_dates",
        "scene_sun_elevations",
        "channel_order",
        "source_dataset_root",
        "row_off",
        "col_off",
        "positive_agl_pixels",
        "building_mask_pixels",
        "source_image_path",
        "source_scene_raster_paths",
        "source_mask_path",
        "source_agl_path",
        "image_path",
        "mask_path",
        "agl_path",
    ]
    clean_rows = []
    for row in rows:
        out = {key: row.get(key, "") for key in fieldnames}
        out["split"] = split_lookup[row["chip_id"]]
        clean_rows.append(out)
    with (output_dir / "chips_manifest.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(clean_rows)


def write_scene_plan(output_dir: Path, scene_plans: dict[str, list[dict[str, Any]]]) -> None:
    rows = []
    for city_slug, plan in scene_plans.items():
        for scene in plan:
            rows.append(
                {
                    "city_slug": city_slug,
                    "model_scene_order": scene["model_scene_order"],
                    "scene_id": scene["id"],
                    "scene_role": scene["scene_role"],
                    "acquired_date": scene["acquired_date"],
                    "sun_elevation": scene["sun_elevation"],
                    "channel_source": scene["channel_source"],
                    "source_raster_path": scene.get("source_raster_path", ""),
                    "rgb_source_band_indexes": scene["rgb_source_band_indexes"],
                }
            )
    with (output_dir / "scene_channel_plan.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_summary_files(
    output_dir: Path,
    rows: list[dict[str, Any]],
    splits: dict[str, list[str]],
    stats: dict[str, Any],
    seed: int,
    scene_count: int,
) -> None:
    counts_by_city: dict[str, int] = {}
    for row in rows:
        counts_by_city[row["source_city"]] = counts_by_city.get(row["source_city"], 0) + 1

    input_channels = scene_count * 3
    readme = f"""# NYC + LA HTC-DC Net {input_channels}-Channel Dataset v1

Created: {datetime.now(timezone.utc).isoformat()}

This dataset stacks RGB imagery from `{scene_count}` PlanetScope scenes per city
onto the existing HTC-DC Net chip grid.  It is designed for the sun-elevation
diversity experiment.

```text
image/   *_IMG.tif  {input_channels}-band PlanetScope chips
mask/    *_BLG.tif  building-mask chips
ndsm/    *_AGL.tif  building-only LiDAR nDSM target chips
stats/   normalization statistics
```

The fixed channel order is recorded in `scene_channel_plan.csv`,
`chips_manifest.csv`, and each image chip's GeoTIFF band descriptions.

Split seed: `{seed}`

| Split | Chips |
|---|---:|
| Train | {len(splits["train"])} |
| Validation | {len(splits["val"])} |
| Test | {len(splits["test"])} |
| All | {len(splits["all"])} |

City counts:

```text
new_york_city = {counts_by_city.get("new_york_city", 0)}
los_angeles = {counts_by_city.get("los_angeles", 0)}
```

Image stats:

```text
image_mean = {stats["image_mean"]}
image_std = {stats["image_std"]}
ndsm_positive_mean = {stats["ndsm_positive_mean"]}
ndsm_positive_std = {stats["ndsm_positive_std"]}
```
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")

    summary = {
        "dataset": output_dir.name,
        "output_dir": rel(output_dir),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "scene_count": scene_count,
        "input_channels": input_channels,
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
    args = parse_args()
    base_dataset_dir = resolve_project_path(args.base_dataset_dir)
    output_dir = resolve_project_path(args.output_dir)
    scene_review = resolve_project_path(args.scene_review)
    split_template_dir = resolve_project_path(args.split_template_dir) if args.split_template_dir else None

    if args.scene_count < 1:
        raise ValueError("--scene-count must be positive.")
    if args.scene_count * 3 not in {3, 6, 12}:
        print(
            f"WARNING: requested {args.scene_count * 3} channels. "
            "The training runner supports arbitrary channel counts, but 3/6/12 are the tested cases."
        )

    ensure_clean_output(output_dir, args.overwrite)
    source_rows = read_manifest(base_dataset_dir)
    scene_rows = read_scene_review(scene_review)
    base_scene_by_city = {}
    for row in source_rows:
        base_scene_by_city.setdefault(row["source_city"], row["source_scene_id"])
    scene_plans = {
        city_slug: scene_plan_for_city(city_slug, base_scene_id, scene_rows, args.scene_count)
        for city_slug, base_scene_id in sorted(base_scene_by_city.items())
    }

    output_rows = [
        write_multiscene_chip(
            row=row,
            city_plan=scene_plans[row["source_city"]],
            base_dataset_dir=base_dataset_dir,
            output_dir=output_dir,
        )
        for row in source_rows
    ]
    chip_ids = [row["chip_id"] for row in output_rows]
    if split_template_dir:
        splits = read_template_splits(split_template_dir, set(chip_ids))
    else:
        splits = split_chip_ids(chip_ids, args.seed)
    write_split_files(output_dir, splits)
    write_manifest(output_dir, output_rows, splits)
    write_scene_plan(output_dir, scene_plans)
    stats = compute_stats(output_rows, output_dir)
    write_summary_files(output_dir, output_rows, splits, stats, args.seed, args.scene_count)

    print(f"Wrote multi-scene HTC dataset: {rel(output_dir)}")
    print(f"Input channels: {args.scene_count * 3}")
    print(f"Total chips: {len(output_rows)}")
    print(
        "Splits: "
        f"train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}"
    )


if __name__ == "__main__":
    main()
