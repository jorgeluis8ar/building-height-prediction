#!/usr/bin/env python3
"""Build single-scene HTC-DC Net input variants from an existing RGB dataset.

The current off-nadir baseline uses three RGB channels.  This script creates
independent 4-channel variants on the same chip grid:

* `rgb_mask`: red, green, blue, building footprint mask
* `rgb_nir`: red, green, blue, near-infrared

The output keeps the HTC-DC Net dataset contract:

```text
image/   *_IMG.tif
mask/    *_BLG.tif
ndsm/    *_AGL.tif
stats/   normalization metadata
```
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
import pickle
import shutil
from typing import Any

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
PLANET_SOURCE_DIR = PROJECT_ROOT / "data_source/data/planet_imagery/source"
DEFAULT_BASE_DATASET_DIR = (
    PROJECT_ROOT / "data_source/data/ml_models/generated/htc_dc_net/nyc_la_off_nadir_rgb_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dataset-dir", type=Path, default=DEFAULT_BASE_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variant", choices=["rgb_mask", "rgb_nir"], required=True)
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


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise RuntimeError(f"No rows found in {path}")
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def copy_split_files(base_dataset_dir: Path, output_dir: Path) -> dict[str, list[str]]:
    splits: dict[str, list[str]] = {}
    for split in ["train", "val", "test", "all"]:
        source = base_dataset_dir / f"{split}.txt"
        if not source.exists():
            raise FileNotFoundError(source)
        values = [line.strip() for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not values:
            raise RuntimeError(f"Split file is empty: {source}")
        splits[split] = values
        (output_dir / f"{split}.txt").write_text("".join(f"{value}\n" for value in values), encoding="utf-8")
    combined = set(splits["train"]) | set(splits["val"]) | set(splits["test"])
    if combined != set(splits["all"]):
        raise RuntimeError("train/val/test split files do not exactly cover all.txt.")
    return splits


def planet_sr_raster_for_scene(city_slug: str, scene_id: str) -> Path:
    city_dir = PLANET_SOURCE_DIR / city_slug
    matches = sorted(
        path
        for path in city_dir.glob(f"**/{scene_id}_3B_AnalyticMS_SR*_clip.tif")
        if "udm" not in path.name.lower()
    )
    if not matches:
        raise FileNotFoundError(f"Missing PlanetScope SR clip for {city_slug} scene {scene_id}")
    if len(matches) > 1:
        raise RuntimeError(f"Found multiple PlanetScope SR clips for {city_slug} scene {scene_id}: {matches}")
    return matches[0]


def band_index(path: Path, requested: str) -> int:
    with rasterio.open(path) as src:
        descriptions = [description.lower() if description else "" for description in src.descriptions]
        if requested in descriptions:
            return descriptions.index(requested) + 1
        if requested == "nir":
            if src.count == 4:
                return 4
            if src.count == 8:
                return 8
    raise RuntimeError(f"Cannot infer {requested} band in {path}; descriptions={descriptions}")


def read_reprojected_band(source_path: Path, source_band: int, template_profile: dict[str, Any]) -> np.ndarray:
    output = np.zeros((template_profile["height"], template_profile["width"]), dtype="float32")
    with rasterio.open(source_path) as src:
        reproject(
            source=rasterio.band(src, source_band),
            destination=output,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=template_profile["transform"],
            dst_crs=template_profile["crs"],
            dst_nodata=0,
            resampling=Resampling.bilinear,
        )
    return output


def write_variant_chip(
    row: dict[str, str],
    base_dataset_dir: Path,
    output_dir: Path,
    variant: str,
) -> dict[str, Any]:
    chip_id = row["chip_id"]
    image_in = base_dataset_dir / "image" / f"{chip_id}_IMG.tif"
    mask_in = base_dataset_dir / "mask" / f"{chip_id}_BLG.tif"
    agl_in = base_dataset_dir / "ndsm" / f"{chip_id}_AGL.tif"
    for path in [image_in, mask_in, agl_in]:
        if not path.exists():
            raise FileNotFoundError(path)

    image_out = output_dir / "image" / f"{chip_id}_IMG.tif"
    mask_out = output_dir / "mask" / f"{chip_id}_BLG.tif"
    agl_out = output_dir / "ndsm" / f"{chip_id}_AGL.tif"

    with rasterio.open(image_in) as src:
        rgb = src.read().astype("float32")
        if rgb.shape[0] != 3:
            raise RuntimeError(f"Expected 3 RGB bands in {image_in}; found {rgb.shape[0]}")
        profile = src.profile.copy()
        template_profile = {
            "height": src.height,
            "width": src.width,
            "transform": src.transform,
            "crs": src.crs,
        }

    source_raster_path = ""
    source_band_indexes = "1,2,3"
    if variant == "rgb_mask":
        with rasterio.open(mask_in) as mask_src:
            mask = mask_src.read(1).astype("float32")
            if (
                mask_src.crs != template_profile["crs"]
                or mask_src.transform != template_profile["transform"]
                or mask_src.width != template_profile["width"]
                or mask_src.height != template_profile["height"]
            ):
                raise RuntimeError(f"Mask chip grid does not match RGB chip grid for {chip_id}")
        extra = (mask > 0).astype("float32")[None, :, :]
        descriptions = ["red", "green", "blue", "building_footprint_mask"]
        output_dtype = "float32"
        nodata = None
        stacked = np.concatenate([rgb, extra], axis=0).astype("float32")
    else:
        source_raster = planet_sr_raster_for_scene(row["source_city"], row["source_scene_id"])
        nir_index = band_index(source_raster, "nir")
        nir = read_reprojected_band(source_raster, nir_index, template_profile)[None, :, :]
        descriptions = ["red", "green", "blue", "nir"]
        output_dtype = "uint16"
        nodata = 0
        stacked = np.concatenate([rgb, nir], axis=0)
        source_raster_path = rel(source_raster)
        source_band_indexes = f"1,2,3,{nir_index}"

    if stacked.shape[0] != 4:
        raise RuntimeError(f"Expected 4 bands for {chip_id}; got {stacked.shape[0]}")

    profile.update(
        driver="GTiff",
        count=4,
        dtype=output_dtype,
        nodata=nodata,
        compress="deflate",
        tiled=True,
        blockxsize=256,
        blockysize=256,
        BIGTIFF="IF_SAFER",
    )
    profile.pop("photometric", None)
    with rasterio.open(image_out, "w", **profile) as dst:
        if output_dtype == "uint16":
            dst.write(np.clip(np.rint(stacked), 0, np.iinfo(np.uint16).max).astype("uint16"))
        else:
            dst.write(stacked.astype("float32"))
        for band_idx, description in enumerate(descriptions, start=1):
            dst.set_band_description(band_idx, description)
        dst.update_tags(
            variant=variant,
            channel_order=";".join(descriptions),
            source_planet_raster=source_raster_path,
            source_band_indexes=source_band_indexes,
        )

    shutil.copy2(mask_in, mask_out)
    shutil.copy2(agl_in, agl_out)

    out_row = row.copy()
    out_row.update(
        {
            "source_dataset_root": rel(base_dataset_dir),
            "variant": variant,
            "input_channels": 4,
            "channel_order": ";".join(descriptions),
            "source_scene_raster_paths": source_raster_path,
            "source_band_indexes": source_band_indexes,
            "image_path": rel(image_out),
            "mask_path": rel(mask_out),
            "agl_path": rel(agl_out),
        }
    )
    return out_row


def compute_stats(rows: list[dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    channel_sum = None
    channel_sum_sq = None
    channel_count = None
    channel_order = None
    agl_sum = 0.0
    agl_sum_sq = 0.0
    agl_count = 0

    for row in rows:
        with rasterio.open(PROJECT_ROOT / row["image_path"]) as image:
            data = image.read(masked=True).astype("float64")
            mask = np.ma.getmaskarray(data)
            descriptions = [description or f"band_{idx}" for idx, description in enumerate(image.descriptions, start=1)]
            if channel_sum is None:
                channel_sum = np.zeros(data.shape[0], dtype="float64")
                channel_sum_sq = np.zeros(data.shape[0], dtype="float64")
                channel_count = np.zeros(data.shape[0], dtype="int64")
                channel_order = descriptions
            for band in range(data.shape[0]):
                filled = np.ma.filled(data[band], np.nan)
                if descriptions[band] == "building_footprint_mask":
                    valid = np.isfinite(filled) & ~mask[band]
                else:
                    valid = np.isfinite(filled) & ~mask[band] & (filled > 0)
                vals = np.asarray(filled[valid], dtype="float64")
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
        raise RuntimeError("No positive AGL pixels were available for target stats.")

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


def validate_outputs(rows: list[dict[str, Any]], output_dir: Path, variant: str) -> dict[str, Any]:
    city_counts: dict[str, int] = {}
    for row in rows:
        chip_id = row["chip_id"]
        city_counts[row["source_city"]] = city_counts.get(row["source_city"], 0) + 1
        image_path = output_dir / "image" / f"{chip_id}_IMG.tif"
        mask_path = output_dir / "mask" / f"{chip_id}_BLG.tif"
        agl_path = output_dir / "ndsm" / f"{chip_id}_AGL.tif"
        with rasterio.open(image_path) as image, rasterio.open(mask_path) as mask, rasterio.open(agl_path) as agl:
            if image.count != 4:
                raise RuntimeError(f"{image_path} has {image.count} bands, expected 4.")
            for other_path, other in [(mask_path, mask), (agl_path, agl)]:
                if (
                    other.crs != image.crs
                    or other.transform != image.transform
                    or other.width != image.width
                    or other.height != image.height
                ):
                    raise RuntimeError(f"{other_path} grid does not match {image_path}")
            image_data = image.read(masked=True)
            for band_index in range(image.count):
                band = image_data[band_index]
                if variant == "rgb_mask" and band_index == 3:
                    values = np.ma.filled(band, np.nan)
                    if not np.isin(values[np.isfinite(values)], [0, 1]).all():
                        raise RuntimeError(f"Mask feature band contains values other than 0/1 in {image_path}")
                else:
                    values = band.compressed()
                    if values.size == 0 or float(np.max(values)) <= 0:
                        raise RuntimeError(f"Image band {band_index + 1} is empty in {image_path}")

    return {
        "variant": variant,
        "input_channels": 4,
        "total_chips": len(rows),
        "new_york_city_chips": city_counts.get("new_york_city", 0),
        "los_angeles_chips": city_counts.get("los_angeles", 0),
        "validation_status": "passed",
    }


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
        "variant",
        "input_channels",
        "channel_order",
        "source_dataset_root",
        "row_off",
        "col_off",
        "positive_agl_pixels",
        "building_mask_pixels",
        "source_image_path",
        "source_scene_raster_paths",
        "source_band_indexes",
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
    write_csv(output_dir / "chips_manifest.csv", clean_rows)


def write_readme(
    output_dir: Path,
    base_dataset_dir: Path,
    variant: str,
    splits: dict[str, list[str]],
    stats: dict[str, Any],
    validation: dict[str, Any],
) -> None:
    description = {
        "rgb_mask": "RGB plus the building-footprint mask as a fourth input channel.",
        "rgb_nir": "RGB plus the PlanetScope near-infrared band as a fourth input channel.",
    }[variant]
    readme = f"""# NYC + LA Off-Nadir HTC-DC Net 4-Channel Dataset: {variant}

Created: {datetime.now(timezone.utc).isoformat()}

This dataset is an independent 4-channel variant of:

```text
{rel(base_dataset_dir)}
```

Variant definition:

```text
{description}
```

Folder contract:

```text
image/   *_IMG.tif  4-band model inputs
mask/    *_BLG.tif  building-mask chips used by the loss/diagnostics
ndsm/    *_AGL.tif  LiDAR nDSM target chips
stats/   normalization metadata
```

Channel order:

```text
{stats["channel_order"]}
```

Splits are copied from the base off-nadir RGB dataset.

| Split | Chips |
|---|---:|
| Train | {len(splits["train"])} |
| Validation | {len(splits["val"])} |
| Test | {len(splits["test"])} |
| All | {len(splits["all"])} |

City counts:

```text
new_york_city = {validation["new_york_city_chips"]}
los_angeles = {validation["los_angeles_chips"]}
```

Image stats:

```text
image_mean = {stats["image_mean"]}
image_std = {stats["image_std"]}
ndsm_positive_mean = {stats["ndsm_positive_mean"]}
ndsm_positive_std = {stats["ndsm_positive_std"]}
```

Validation:

```text
{validation}
```
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    args = parse_args()
    base_dataset_dir = resolve_project_path(args.base_dataset_dir)
    output_dir = resolve_project_path(args.output_dir)
    ensure_clean_output(output_dir, args.overwrite)

    source_rows = read_csv(base_dataset_dir / "chips_manifest.csv")
    splits = copy_split_files(base_dataset_dir, output_dir)
    output_rows = [
        write_variant_chip(row=row, base_dataset_dir=base_dataset_dir, output_dir=output_dir, variant=args.variant)
        for row in source_rows
    ]
    write_manifest(output_dir, output_rows, splits)
    stats = compute_stats(output_rows, output_dir)
    validation = validate_outputs(output_rows, output_dir, args.variant)
    write_csv(output_dir / "alignment_validation_summary.csv", [validation])
    write_readme(output_dir, base_dataset_dir, args.variant, splits, stats, validation)

    print(f"Wrote HTC dataset variant: {rel(output_dir)}")
    print(f"Variant: {args.variant}")
    print(f"Input channels: 4")
    print(f"Total chips: {len(output_rows)}")
    print(
        "Splits: "
        f"train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}"
    )


if __name__ == "__main__":
    main()
