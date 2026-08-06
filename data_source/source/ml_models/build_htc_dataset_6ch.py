#!/usr/bin/env python3
"""Build a NYC+LA HTC-DC Net dataset with two PlanetScope RGB scenes.

The output keeps the HTC-DC Net folder contract, but writes 6-band image chips:

    bands 1-3: primary/winter RGB scene
    bands 4-6: secondary/summer RGB scene aligned to the same chip grid
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
RGB_DATASET_DIR = (
    PROJECT_ROOT / "data_source/data/ml_models/generated/htc_dc_net/nyc_la_rgb_v1"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "data_source/data/ml_models/generated/htc_dc_net/nyc_la_6ch_v1"
)
DEFAULT_SEED = 20260709

SECONDARY_SCENES = {
    "los_angeles": {
        "season": "summer",
        "scene_id": "20230705_174134_45_245c",
        "path": (
            PROJECT_ROOT
            / "data_source/data/planet_imagery/source/los_angeles/"
            / "summer_jun_jul_20230705_174134_45_245c/"
            / "c074f656-c35f-40f1-86e0-5c1c87bd3de3/PSScene/"
            / "20230705_174134_45_245c_3B_AnalyticMS_SR_8b_clip.tif"
        ),
    },
    "new_york_city": {
        "season": "summer",
        "scene_id": "20200614_155201_71_105e",
        "path": (
            PROJECT_ROOT
            / "data_source/data/planet_imagery/source/new_york_city/"
            / "summer_jun_jul_20200614_155201_71_105e/"
            / "bfd6d704-820c-4457-a089-f835d36f8383/PSScene/"
            / "20200614_155201_71_105e_3B_AnalyticMS_SR_clip.tif"
        ),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb-dataset-dir", type=Path, default=RGB_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def rel(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT).as_posix()


def ensure_clean_output(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output exists: {output_dir}. Pass --overwrite.")
        shutil.rmtree(output_dir)
    for name in ["image", "mask", "ndsm", "stats"]:
        (output_dir / name).mkdir(parents=True, exist_ok=True)


def rgb_band_indexes(path: Path) -> list[int]:
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


def read_manifest(rgb_dataset_dir: Path) -> list[dict[str, Any]]:
    manifest_path = rgb_dataset_dir / "chips_manifest.csv"
    with manifest_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise RuntimeError(f"No rows found in {manifest_path}")
    return rows


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


def write_split_files(output_dir: Path, splits: dict[str, list[str]]) -> None:
    for split, chip_ids in splits.items():
        (output_dir / f"{split}.txt").write_text(
            "".join(f"{chip_id}\n" for chip_id in chip_ids),
            encoding="utf-8",
        )


def read_secondary_rgb(
    secondary_path: Path,
    rgb_indexes: list[int],
    template_profile: dict,
) -> np.ndarray:
    secondary = np.zeros((3, template_profile["height"], template_profile["width"]), dtype="float32")
    with rasterio.open(secondary_path) as src:
        for out_idx, src_idx in enumerate(rgb_indexes):
            reproject(
                source=rasterio.band(src, src_idx),
                destination=secondary[out_idx],
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=src.nodata,
                dst_transform=template_profile["transform"],
                dst_crs=template_profile["crs"],
                dst_nodata=0,
                resampling=Resampling.bilinear,
            )
    return secondary


def write_6ch_chip(row: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    city = row["source_city"]
    secondary = SECONDARY_SCENES[city]
    secondary_path = secondary["path"]
    if not secondary_path.exists():
        raise FileNotFoundError(secondary_path)
    secondary_rgb_indexes = rgb_band_indexes(secondary_path)

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

    with rasterio.open(image_in) as src:
        primary_rgb = src.read().astype("float32")
        profile = src.profile.copy()
        template_profile = {
            "height": src.height,
            "width": src.width,
            "transform": src.transform,
            "crs": src.crs,
        }
    secondary_rgb = read_secondary_rgb(secondary_path, secondary_rgb_indexes, template_profile)
    stacked = np.concatenate([primary_rgb, secondary_rgb], axis=0)

    profile.update(
        driver="GTiff",
        count=6,
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
        descriptions = [
            "winter_red",
            "winter_green",
            "winter_blue",
            "summer_red",
            "summer_green",
            "summer_blue",
        ]
        for band_idx, description in enumerate(descriptions, start=1):
            dst.set_band_description(band_idx, description)
        dst.update_tags(
            primary_scene_id=row["source_scene_id"],
            secondary_scene_id=secondary["scene_id"],
            secondary_season=secondary["season"],
            secondary_rgb_source_band_indexes=",".join(map(str, secondary_rgb_indexes)),
        )

    shutil.copy2(mask_in, mask_out)
    shutil.copy2(agl_in, agl_out)

    out_row = row.copy()
    out_row.update(
        {
            "source_primary_scene_id": row["source_scene_id"],
            "source_secondary_scene_id": secondary["scene_id"],
            "source_secondary_season": secondary["season"],
            "source_secondary_image_path": rel(secondary_path),
            "secondary_rgb_source_band_indexes": ",".join(map(str, secondary_rgb_indexes)),
            "image_path": rel(image_out),
            "mask_path": rel(mask_out),
            "agl_path": rel(agl_out),
        }
    )
    return out_row


def compute_stats(rows: list[dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    channel_sum = None
    channel_sum_sq = None
    channel_count = 0
    agl_sum = 0.0
    agl_sum_sq = 0.0
    agl_count = 0

    for row in rows:
        with rasterio.open(PROJECT_ROOT / row["image_path"]) as image:
            data = image.read(masked=True).astype("float64")
            valid = np.ma.filled(~np.ma.getmaskarray(data) & (data > 0), False)
            if channel_sum is None:
                channel_sum = np.zeros(data.shape[0], dtype="float64")
                channel_sum_sq = np.zeros(data.shape[0], dtype="float64")
            for band in range(data.shape[0]):
                vals = data[band][valid[band]].compressed()
                channel_sum[band] += vals.sum()
                channel_sum_sq[band] += np.square(vals).sum()
            channel_count += int(np.sum(valid[0]))

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
    image_std = np.sqrt(np.maximum(channel_sum_sq / channel_count - image_mean**2, 0))
    agl_mean = agl_sum / agl_count
    agl_std = np.sqrt(max(agl_sum_sq / agl_count - agl_mean**2, 0))
    ndsm_max = float(agl_mean + 6 * agl_std)
    count = torch.zeros(int(max(1, round(ndsm_max))) + 1)

    image_stats = {
        "image_mean": image_mean.tolist(),
        "image_std": image_std.tolist(),
        "image_pixel_count": int(channel_count),
        "channel_order": [
            "winter_red",
            "winter_green",
            "winter_blue",
            "summer_red",
            "summer_green",
            "summer_blue",
        ],
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
        "source_primary_scene_id",
        "source_secondary_scene_id",
        "source_secondary_season",
        "source_dataset_root",
        "row_off",
        "col_off",
        "positive_agl_pixels",
        "building_mask_pixels",
        "source_image_path",
        "source_secondary_image_path",
        "source_mask_path",
        "source_agl_path",
        "secondary_rgb_source_band_indexes",
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


def write_summary_files(
    output_dir: Path,
    rows: list[dict[str, Any]],
    splits: dict[str, list[str]],
    stats: dict[str, Any],
    seed: int,
) -> None:
    counts_by_city: dict[str, int] = {}
    for row in rows:
        counts_by_city[row["source_city"]] = counts_by_city.get(row["source_city"], 0) + 1

    readme = f"""# NYC + LA HTC-DC Net 6-Channel Dataset v1

Created: {datetime.now(timezone.utc).isoformat()}

This dataset is a two-scene extension of `nyc_la_rgb_v1`. It keeps the same
HTC-DC Net folder contract while writing 6-band image chips.

```text
image/   *_IMG.tif  6-band PlanetScope chips
mask/    *_BLG.tif  building-mask chips
ndsm/    *_AGL.tif  building-only LiDAR nDSM target chips
stats/   normalization statistics
```

Band order:

```text
1 winter_red
2 winter_green
3 winter_blue
4 summer_red
5 summer_green
6 summer_blue
```

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
        "dataset": "nyc_la_6ch_v1",
        "output_dir": rel(output_dir),
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
    args = parse_args()
    rgb_dataset_dir = args.rgb_dataset_dir.resolve()
    output_dir = args.output_dir.resolve()
    ensure_clean_output(output_dir, args.overwrite)

    source_rows = read_manifest(rgb_dataset_dir)
    output_rows = [write_6ch_chip(row, output_dir) for row in source_rows]
    splits = split_chip_ids([row["chip_id"] for row in output_rows], args.seed)
    write_split_files(output_dir, splits)
    write_manifest(output_dir, output_rows, splits)
    stats = compute_stats(output_rows, output_dir)
    write_summary_files(output_dir, output_rows, splits, stats, args.seed)

    print(f"Wrote 6-channel HTC dataset: {rel(output_dir)}")
    print(f"Total chips: {len(output_rows)}")
    print(
        "Splits: "
        f"train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}"
    )


if __name__ == "__main__":
    main()
