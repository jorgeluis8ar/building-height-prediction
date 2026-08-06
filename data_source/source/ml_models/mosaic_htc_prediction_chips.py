#!/usr/bin/env python3
"""Mosaic HTC-DC Net prediction chips into city-level GeoTIFF rasters."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import rasterio
from rasterio.merge import merge
from rasterio.warp import reproject
from rasterio.enums import Resampling


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3]
DEFAULT_RUN_DIR = (
    REPO_ROOT
    / "data_source/data/ml_models/generated/htc_dc_net/mini_training_runs/"
    / "nyc71_la100_seed20260707_epoch5_repo_params_guarded"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--summary-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--predictions-subdir", default="all_predictions_epoch_005")
    parser.add_argument(
        "--epoch",
        type=int,
        default=5,
        help="Epoch number used in output mosaic filenames.",
    )
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument(
        "--resolution-meters",
        type=float,
        default=None,
        help="Output mosaic pixel size in map units. Use 3 for 3 x 3 meter rasters in UTM CRS.",
    )
    parser.add_argument(
        "--output-suffix",
        default="",
        help="Suffix inserted before .tif in city mosaic filenames, for example _3m.",
    )
    parser.add_argument(
        "--mask-predictions",
        action="store_true",
        help="Apply each chip building mask before mosaicking predictions.",
    )
    parser.add_argument("--nodata", type=float, default=-9999.0)
    return parser.parse_args()


def read_rows(summary_csv: Path) -> list[dict]:
    with summary_csv.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def city_slug(city: str) -> str:
    return city.lower().replace(" ", "_")


def write_masked_prediction(
    prediction_path: Path,
    mask_path: Path,
    output_path: Path,
    nodata: float,
) -> Path:
    with rasterio.open(prediction_path) as pred_src:
        pred = pred_src.read(1).astype("float32")
        if pred_src.nodata is not None:
            pred[pred == pred_src.nodata] = nodata
        with rasterio.open(mask_path) as mask_src:
            mask_arr = mask_src.read(1).astype("float32")
            if (
                mask_src.crs != pred_src.crs
                or mask_src.transform != pred_src.transform
                or mask_src.width != pred_src.width
                or mask_src.height != pred_src.height
            ):
                aligned = pred.copy()
                aligned.fill(0)
                reproject(
                    source=rasterio.band(mask_src, 1),
                    destination=aligned,
                    src_transform=mask_src.transform,
                    src_crs=mask_src.crs,
                    src_nodata=mask_src.nodata,
                    dst_transform=pred_src.transform,
                    dst_crs=pred_src.crs,
                    dst_nodata=0,
                    resampling=Resampling.nearest,
                )
                mask_arr = aligned
        masked = pred.copy()
        masked[mask_arr <= 0] = nodata
        profile = pred_src.profile.copy()
        profile.update(nodata=nodata, dtype="float32", compress="deflate", predictor=2)
        profile.pop("photometric", None)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(masked.astype("float32"), 1)
    return output_path


def mosaic_city(
    city: str,
    paths: list[Path],
    output_dir: Path,
    nodata: float,
    resolution_meters: float | None,
    output_suffix: str,
    epoch: int,
) -> Path:
    datasets = [rasterio.open(path) for path in paths]
    try:
        merge_kwargs = {"nodata": nodata}
        if resolution_meters is not None:
            merge_kwargs["res"] = (resolution_meters, resolution_meters)
        mosaic, transform = merge(datasets, **merge_kwargs)
        profile = datasets[0].profile.copy()
        profile.update(
            driver="GTiff",
            height=mosaic.shape[1],
            width=mosaic.shape[2],
            transform=transform,
            count=1,
            dtype="float32",
            nodata=nodata,
            compress="deflate",
            predictor=2,
            tiled=True,
            blockxsize=256,
            blockysize=256,
        )
        profile.pop("photometric", None)
        out_path = output_dir / f"{city_slug(city)}_predicted_ndsm_epoch_{epoch:03d}{output_suffix}.tif"
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(mosaic[0].astype("float32"), 1)
        return out_path
    finally:
        for dataset in datasets:
            dataset.close()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    summary_csv = (args.summary_csv or run_dir / "all_predictions_summary_epoch_005.csv").resolve()
    output_dir = (args.output_dir or run_dir / "mosaics_epoch_005").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = (args.data_dir or run_dir / "mini_dataset").resolve()
    masked_dir = output_dir / "masked_prediction_chips"

    rows = read_rows(summary_csv)
    paths_by_city: dict[str, list[Path]] = defaultdict(list)
    for row in rows:
        prediction_path = REPO_ROOT / row["prediction_path"]
        if not prediction_path.exists():
            fallback = run_dir / args.predictions_subdir / f"{row['chip_id']}_ndsm_pred.tif"
            prediction_path = fallback
        if not prediction_path.exists():
            raise FileNotFoundError(prediction_path)
        if args.mask_predictions:
            mask_path = data_dir / "mask" / f"{row['chip_id']}_BLG.tif"
            if not mask_path.exists():
                raise FileNotFoundError(mask_path)
            prediction_path = write_masked_prediction(
                prediction_path=prediction_path,
                mask_path=mask_path,
                output_path=masked_dir / f"{row['chip_id']}_ndsm_pred_masked.tif",
                nodata=args.nodata,
            )
        paths_by_city[row["source_city"]].append(prediction_path)

    manifest_rows = []
    for city, paths in sorted(paths_by_city.items()):
        out_path = mosaic_city(
            city=city,
            paths=sorted(paths),
            output_dir=output_dir,
            nodata=args.nodata,
            resolution_meters=args.resolution_meters,
            output_suffix=args.output_suffix,
            epoch=args.epoch,
        )
        with rasterio.open(out_path) as src:
            manifest_rows.append(
                {
                    "source_city": city,
                    "chip_count": len(paths),
                    "mosaic_path": str(out_path.relative_to(REPO_ROOT)),
                    "crs": str(src.crs),
                    "width": src.width,
                    "height": src.height,
                    "pixel_width": src.res[0],
                    "pixel_height": src.res[1],
                    "requested_resolution_meters": args.resolution_meters,
                    "nodata": src.nodata,
                    "masked_to_buildings": args.mask_predictions,
                    "bounds_left": src.bounds.left,
                    "bounds_bottom": src.bounds.bottom,
                    "bounds_right": src.bounds.right,
                    "bounds_top": src.bounds.top,
                }
            )

    manifest_path = output_dir / "mosaic_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"Mosaics written to: {output_dir}")
    print(f"Manifest: {manifest_path}")
    for row in manifest_rows:
        print(f"{row['source_city']}: {row['chip_count']} chips -> {row['mosaic_path']}")


if __name__ == "__main__":
    main()
