"""
Build Planet-Aligned LiDAR nDSM And HTC-DC Net Inputs

Environment: data_source/source/height_labels/venv_height_labels

Description:
    Builds LiDAR-derived DSM, DTM, and nDSM rasters on the exact grid of a
    downloaded PlanetScope scene. It can also export HTC-DC-Net-style files:

        image/<chip_id>_IMG.tif   3-band RGB Planet chip
        mask/<chip_id>_BLG.tif    building footprint mask chip
        ndsm/<chip_id>_AGL.tif    above-ground LiDAR nDSM target chip

    The New Jersey Sandy LiDAR tiles are stored in the NYC LiDAR folder because
    they intersect the NYC AOI. Use
    `--city new_york_city --lidar-project NJ_New_Jersey_SANDY_LiDAR_15` to
    produce the New Jersey LiDAR variant.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import logging
import os
from pathlib import Path
import pickle
import random
import re
import sys
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
VENV_DIR = SCRIPT_DIR / "venv_height_labels"
VENV_PYTHON = VENV_DIR / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
VENV_MARKER = "HEIGHT_LABELS_VENV_ACTIVE"

NODATA_VALUE = -9999.0
GROUND_CLASSES = {2}
EXCLUDED_CLASSES = {7, 9, 17, 18}

HEIGHT_GENERATED_DIR = PROJECT_ROOT / "data_source/data/height_labels/generated"
PLANET_SOURCE_DIR = PROJECT_ROOT / "data_source/data/planet_imagery/source"
FOOTPRINT_GENERATED_DIR = PROJECT_ROOT / "data_source/data/building_footprints/generated"

DEFAULT_LIDAR_PROJECT = {
    "new_york_city": "NY_New_York_CMGP_SANDY_LiDAR_15",
    "los_angeles": "CA_LosAngeles_B23",
}

DEFAULT_TEMPLATE_SCENE_ID = {
    "new_york_city": "20200122_154449_92_1061",
    "los_angeles": "20231203_182937_07_2488",
}

DEFAULT_FOOTPRINT_PATH = {
    "new_york_city": (
        FOOTPRINT_GENERATED_DIR
        / "new_york_city"
        / "new_york_city_building_footprints_merged_5km.gpkg"
    ),
    "los_angeles": (
        FOOTPRINT_GENERATED_DIR
        / "los_angeles"
        / "los_angeles_building_footprints_merged_5km.gpkg"
    ),
}


def relaunch_inside_venv() -> None:
    """Restart this script inside the task-specific virtual environment."""
    if os.environ.get(VENV_MARKER) == "1":
        return

    current_python = Path(sys.executable).absolute()
    expected_python = VENV_PYTHON.absolute()
    if current_python == expected_python or Path(sys.prefix).absolute() == VENV_DIR.absolute():
        os.environ[VENV_MARKER] = "1"
        return

    if not VENV_PYTHON.exists():
        print("ERROR: Missing height_labels virtual environment.")
        print(f"Expected Python executable: {VENV_PYTHON}")
        print("Create it with:")
        print("  python3 -m venv data_source/source/height_labels/venv_height_labels")
        print(
            "  data_source/source/height_labels/venv_height_labels/bin/python "
            "-m pip install -r data_source/source/height_labels/requirements.txt"
        )
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
        description="Build Planet-aligned LiDAR nDSM rasters and HTC-ready files."
    )
    parser.add_argument(
        "--city",
        choices=sorted(DEFAULT_LIDAR_PROJECT),
        default="new_york_city",
        help="City folder to process.",
    )
    parser.add_argument(
        "--lidar-project",
        default=None,
        help=(
            "USGS 3DEP project_directory to use. Defaults to the city project. "
            "For New Jersey Sandy tiles use NJ_New_Jersey_SANDY_LiDAR_15."
        ),
    )
    parser.add_argument(
        "--template-scene-id",
        default=None,
        help="Planet scene ID whose grid should be used as the output template.",
    )
    parser.add_argument(
        "--output-label",
        default=None,
        help=(
            "Output folder/file label. Defaults to the city slug, except the "
            "New Jersey Sandy project defaults to new_york_city_new_jersey_lidar."
        ),
    )
    parser.add_argument(
        "--footprints",
        default=None,
        help="Optional footprint GeoPackage path. Defaults to the merged city footprints.",
    )
    parser.add_argument(
        "--ground-fill-distance-pixels",
        type=float,
        default=250.0,
        help=(
            "Maximum pixel distance used to interpolate missing DTM cells from "
            "observed ground cells. Default: 250 pixels, or 750 m on a 3 m grid."
        ),
    )
    parser.add_argument(
        "--all-touched-building-mask",
        action="store_true",
        help="Rasterize the building mask into every pixel touched by a footprint.",
    )
    parser.add_argument(
        "--no-htc-files",
        action="store_true",
        help="Only write the diagnostic multiband nDSM raster and summary.",
    )
    parser.add_argument(
        "--no-chips",
        action="store_true",
        help="Write full-scene HTC files but skip 256x256 chip creation.",
    )
    parser.add_argument(
        "--chip-size",
        type=int,
        default=256,
        help="HTC chip size in pixels. Default: 256.",
    )
    parser.add_argument(
        "--chip-stride",
        type=int,
        default=256,
        help="HTC chip stride in pixels. Default: 256.",
    )
    parser.add_argument(
        "--min-positive-agl-pixels",
        type=int,
        default=25,
        help="Minimum positive AGL pixels required to keep a chip. Default: 25.",
    )
    parser.add_argument(
        "--min-building-agl-m",
        type=float,
        default=2.4,
        help=(
            "Minimum AGL height, in meters, assigned to finite nonpositive nDSM "
            "pixels inside the building mask. Default: 2.4."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260702,
        help="Seed for deterministic train/val/test chip splits.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs.",
    )
    return parser.parse_args()


def slugify(value: str) -> str:
    """Return a filesystem-safe label."""
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()
    return cleaned or "output"


def output_label_for(city: str, lidar_project: str, output_label: str | None) -> str:
    """Choose an output label for this run."""
    if output_label:
        return slugify(output_label)
    if city == "new_york_city" and lidar_project.startswith("NJ_New_Jersey"):
        return "new_york_city_new_jersey_lidar"
    return city


def configure_logging(output_label: str) -> Path:
    """Create a run log."""
    log_dir = HEIGHT_GENERATED_DIR / output_label / "lidar_ndsm"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"build_lidar_ndsm_raster_{timestamp}.log"
    for handler in logging.getLogger().handlers[:]:
        logging.getLogger().removeHandler(handler)
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    logging.getLogger().addHandler(console)
    return log_path


def relative_path(path: Path) -> str:
    """Return a project-relative path for logs and summaries."""
    return path.resolve().relative_to(PROJECT_ROOT).as_posix()


def scene_id_from_planet_tif(path: Path) -> str:
    """Extract the Planet scene ID from the TIFF filename."""
    return path.name.split("_3B_", maxsplit=1)[0]


def find_planet_template(city: str, scene_id: str) -> Path:
    """Find one downloaded Planet TIFF to use as the output grid template."""
    matches = [
        path
        for path in sorted((PLANET_SOURCE_DIR / city).rglob("*_3B_AnalyticMS_SR*_clip.tif"))
        if scene_id_from_planet_tif(path) == scene_id
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one Planet template for {city} scene {scene_id}, "
            f"found {len(matches)}."
        )
    return matches[0]


def load_manifest_tiles(city: str, lidar_project: str) -> list[Path]:
    """Load manifest-approved local LiDAR tiles for a city/project pair."""
    import pandas as pd

    manifest_path = HEIGHT_GENERATED_DIR / "usgs_3dep_tile_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing LiDAR manifest: {manifest_path}")

    manifest = pd.read_csv(manifest_path)
    tiles = manifest[
        (manifest["city_slug"] == city)
        & (manifest["project_directory"] == lidar_project)
        & (manifest["download_status"].isin(["existing", "recovered"]))
    ].copy()
    if tiles.empty:
        raise RuntimeError(f"No local manifest tiles found for {city} / {lidar_project}.")

    paths = []
    for _, row in tiles.iterrows():
        path = PROJECT_ROOT / row["local_path"]
        if not path.exists():
            raise FileNotFoundError(f"Manifest LiDAR tile is missing: {path}")
        if int(path.stat().st_size) != int(row["expected_bytes"]):
            raise RuntimeError(
                f"LiDAR tile byte size mismatch for {path}: "
                f"expected {row['expected_bytes']}, got {path.stat().st_size}"
            )
        paths.append(path)
    return paths


def point_crs_from_header(header: Any) -> Any:
    """Read CRS from a LAS/LAZ header, falling back to common source CRSs."""
    import pyproj

    crs = header.parse_crs()
    if crs is not None:
        return crs

    # NYC Sandy LiDAR tiles have used NAD83(2011) / UTM zone 18N in our prior
    # runs. LA 2023 tiles usually carry CRS metadata, so this is mainly a NYC
    # safety fallback.
    return pyproj.CRS.from_epsg(6347)


def point_grid_indices(
    x: np.ndarray,
    y: np.ndarray,
    transform: Any,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert projected point coordinates to template row/column indices."""
    col = np.floor((x - transform.c) / transform.a).astype(np.int64)
    row = np.floor((y - transform.f) / transform.e).astype(np.int64)
    valid = (col >= 0) & (col < width) & (row >= 0) & (row < height)
    return row[valid], col[valid], valid


def has_dimension(points: Any, name: str) -> bool:
    """Check whether a LAS point record has a dimension."""
    return name in points.point_format.dimension_names


def accumulate_lidar_surfaces(
    tile_paths: list[Path],
    template_crs: Any,
    transform: Any,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Create raw DSM and observed-ground DTM arrays from LAZ point tiles."""
    import laspy
    import pyproj

    dsm = np.full((height, width), NODATA_VALUE, dtype=np.float32)
    ground_sum = np.zeros((height, width), dtype=np.float64)
    ground_count = np.zeros((height, width), dtype=np.uint32)

    stats: dict[str, Any] = {
        "tile_count": len(tile_paths),
        "total_points": 0,
        "template_points": 0,
        "surface_points": 0,
        "ground_points": 0,
        "excluded_points": 0,
    }

    transformer = None
    transformer_key = None

    for tile_index, tile_path in enumerate(tile_paths, start=1):
        logging.info("Reading tile %s/%s: %s", tile_index, len(tile_paths), relative_path(tile_path))
        laz = laspy.read(tile_path)
        point_crs = point_crs_from_header(laz.header)
        key = (point_crs.to_string(), template_crs.to_string())
        if transformer is None or key != transformer_key:
            transformer = pyproj.Transformer.from_crs(point_crs, template_crs, always_xy=True)
            transformer_key = key

        x = np.asarray(laz.x)
        y = np.asarray(laz.y)
        z = np.asarray(laz.z, dtype=np.float32)
        classification = np.asarray(laz.classification)

        x_proj, y_proj = transformer.transform(x, y)
        row, col, in_template = point_grid_indices(
            np.asarray(x_proj),
            np.asarray(y_proj),
            transform,
            width,
            height,
        )

        if not in_template.any():
            stats["total_points"] += len(z)
            continue

        z = z[in_template]
        classification = classification[in_template]
        flat = row * width + col

        if has_dimension(laz.points, "withheld"):
            withheld = np.asarray(laz.withheld)[in_template].astype(bool)
        else:
            withheld = np.zeros(len(z), dtype=bool)

        excluded = np.isin(classification, list(EXCLUDED_CLASSES)) | withheld
        ground = np.isin(classification, list(GROUND_CLASSES)) & ~excluded
        surface = (~ground) & ~excluded

        if ground.any():
            np.add.at(ground_sum.ravel(), flat[ground], z[ground])
            np.add.at(ground_count.ravel(), flat[ground], 1)

        if surface.any():
            np.maximum.at(dsm.ravel(), flat[surface], z[surface])

        stats["total_points"] += len(laz.points)
        stats["template_points"] += int(in_template.sum())
        stats["surface_points"] += int(surface.sum())
        stats["ground_points"] += int(ground.sum())
        stats["excluded_points"] += int(excluded.sum())

    dtm_observed = np.full((height, width), np.nan, dtype=np.float32)
    observed = ground_count > 0
    dtm_observed[observed] = (ground_sum[observed] / ground_count[observed]).astype(np.float32)

    dsm[dsm == NODATA_VALUE] = np.nan
    stats["dsm_observed_pixels"] = int(np.isfinite(dsm).sum())
    stats["dtm_observed_pixels"] = int(observed.sum())
    return dsm, dtm_observed, ground_count, stats


def fill_dtm(dtm_observed: np.ndarray, max_distance_pixels: float) -> np.ndarray:
    """Fill DTM gaps from observed ground cells using rasterio's inverse-distance fill."""
    from rasterio.fill import fillnodata

    valid_mask = np.isfinite(dtm_observed)
    if not valid_mask.any():
        raise RuntimeError("No observed ground cells were available for DTM filling.")

    image = np.where(valid_mask, dtm_observed, 0).astype(np.float32)
    filled = fillnodata(
        image,
        mask=valid_mask.astype(np.uint8),
        max_search_distance=max_distance_pixels,
        smoothing_iterations=1,
    )
    filled[~np.isfinite(filled)] = np.nan
    return filled.astype(np.float32)


def rasterize_building_mask(
    template_path: Path,
    footprint_path: Path,
    all_touched: bool,
) -> np.ndarray:
    """Rasterize the merged footprint layer to the Planet grid."""
    import geopandas as gpd
    import rasterio
    from rasterio.features import rasterize

    if not footprint_path.exists():
        raise FileNotFoundError(f"Missing footprint layer: {footprint_path}")

    with rasterio.open(template_path) as template:
        footprints = gpd.read_file(footprint_path).to_crs(template.crs)
        shapes = [
            (geometry, 1)
            for geometry in footprints.geometry
            if geometry is not None and not geometry.is_empty
        ]
        if not shapes:
            raise RuntimeError(f"No valid footprint geometries found in {footprint_path}")
        mask = rasterize(
            shapes,
            out_shape=(template.height, template.width),
            transform=template.transform,
            fill=0,
            dtype="uint8",
            all_touched=all_touched,
        )
    return mask


def diagnostic_bands(
    dsm: np.ndarray,
    dtm_observed: np.ndarray,
    dtm_filled: np.ndarray,
    ndsm: np.ndarray,
    building_mask: np.ndarray,
    ndsm_buildings_only: np.ndarray,
) -> list[tuple[str, np.ndarray]]:
    """Return the multiband diagnostic raster band set."""
    return [
        ("dsm_m", dsm),
        ("dtm_ground_observed_m", dtm_observed),
        ("dtm_ground_filled_m", dtm_filled),
        ("ndsm_m", ndsm),
        ("building_mask", building_mask.astype(np.float32)),
        ("ndsm_buildings_only_m", ndsm_buildings_only),
    ]


def enforce_minimum_building_agl(
    ndsm: np.ndarray,
    building_mask: np.ndarray,
    minimum_height_m: float,
) -> tuple[np.ndarray, dict[str, int]]:
    """Create building-only AGL with a minimum positive height inside buildings."""
    building_pixels = building_mask > 0
    finite_building_pixels = building_pixels & np.isfinite(ndsm)
    imputed_pixels = finite_building_pixels & (ndsm <= 0)
    ndsm_buildings_only = np.where(building_pixels, ndsm, np.nan).astype(np.float32)
    ndsm_buildings_only[imputed_pixels] = minimum_height_m
    audit = {
        "building_agl_minimum_imputed_pixels": int(imputed_pixels.sum()),
        "building_agl_remaining_zero_pixels": int(
            ((ndsm_buildings_only == 0) & building_pixels & np.isfinite(ndsm_buildings_only)).sum()
        ),
        "building_agl_remaining_negative_pixels": int(
            ((ndsm_buildings_only < 0) & building_pixels & np.isfinite(ndsm_buildings_only)).sum()
        ),
    }
    return ndsm_buildings_only, audit


def write_multiband_output(
    template_path: Path,
    output_path: Path,
    bands: list[tuple[str, np.ndarray]],
    tags: dict[str, str],
) -> None:
    """Write a multiband GeoTIFF using the Planet template metadata."""
    import rasterio

    with rasterio.open(template_path) as template:
        profile = template.profile.copy()
        profile.update(
            driver="GTiff",
            count=len(bands),
            dtype="float32",
            nodata=NODATA_VALUE,
            compress="deflate",
            predictor=2,
            tiled=True,
            blockxsize=512,
            blockysize=512,
            BIGTIFF="IF_SAFER",
        )
        profile.pop("photometric", None)

        with rasterio.open(output_path, "w", **profile) as dst:
            for band_index, (description, array) in enumerate(bands, start=1):
                out = np.where(np.isfinite(array), array, NODATA_VALUE).astype(np.float32)
                dst.write(out, band_index)
                dst.set_band_description(band_index, description)
            dst.update_tags(**tags)


def rgb_band_indexes(template_path: Path) -> list[int]:
    """Return 1-based RGB band indexes for Planet 4-band or 8-band SR imagery."""
    import rasterio

    with rasterio.open(template_path) as src:
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
    raise RuntimeError(f"Cannot infer RGB band indexes for {template_path}")


def write_single_band_raster(
    template_path: Path,
    output_path: Path,
    array: np.ndarray,
    dtype: str,
    nodata: float | int,
    description: str,
    tags: dict[str, str],
) -> None:
    """Write one single-band raster aligned to the Planet template."""
    import rasterio

    with rasterio.open(template_path) as template:
        profile = template.profile.copy()
        profile.update(
            driver="GTiff",
            count=1,
            dtype=dtype,
            nodata=nodata,
            compress="deflate",
            tiled=True,
            blockxsize=512,
            blockysize=512,
            BIGTIFF="IF_SAFER",
        )
        profile.pop("photometric", None)
        if dtype == "float32":
            profile["predictor"] = 2

        if dtype == "float32":
            out = np.where(np.isfinite(array), array, nodata).astype(np.float32)
        else:
            out = array.astype(dtype)
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(out, 1)
            dst.set_band_description(1, description)
            dst.update_tags(**tags)


def write_rgb_image(template_path: Path, output_path: Path, tags: dict[str, str]) -> list[int]:
    """Write a 3-band RGB Planet image for HTC-style model input."""
    import rasterio

    indexes = rgb_band_indexes(template_path)
    with rasterio.open(template_path) as src:
        profile = src.profile.copy()
        profile.update(
            driver="GTiff",
            count=3,
            dtype=src.dtypes[0],
            nodata=src.nodata,
            compress="deflate",
            tiled=True,
            blockxsize=512,
            blockysize=512,
            BIGTIFF="IF_SAFER",
        )
        profile.pop("photometric", None)
        with rasterio.open(output_path, "w", **profile) as dst:
            for out_index, src_index in enumerate(indexes, start=1):
                dst.write(src.read(src_index), out_index)
            dst.set_band_description(1, "red")
            dst.set_band_description(2, "green")
            dst.set_band_description(3, "blue")
            dst.update_tags(**tags, rgb_source_band_indexes=",".join(map(str, indexes)))
    return indexes


def verify_output(template_path: Path, output_path: Path) -> None:
    """Verify output grid alignment against the Planet template."""
    import rasterio

    with rasterio.open(template_path) as template, rasterio.open(output_path) as output:
        checks = {
            "crs": output.crs == template.crs,
            "transform": output.transform == template.transform,
            "width": output.width == template.width,
            "height": output.height == template.height,
            "bounds": output.bounds == template.bounds,
            "resolution": output.res == template.res,
        }
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            raise RuntimeError(
                f"Output alignment check failed for {output_path}: {', '.join(failed)}"
            )


def write_summary(summary_path: Path, row: dict[str, Any]) -> None:
    """Write a one-row CSV summary of the nDSM build."""
    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def write_window_raster(
    source_path: Path,
    output_path: Path,
    window: Any,
    descriptions: list[str],
) -> None:
    """Write a raster chip from a source raster window."""
    import rasterio

    with rasterio.open(source_path) as src:
        profile = src.profile.copy()
        profile.update(
            height=window.height,
            width=window.width,
            transform=src.window_transform(window),
            compress="deflate",
            tiled=False,
        )
        profile.pop("blockxsize", None)
        profile.pop("blockysize", None)
        data = src.read(window=window)
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(data)
            for index, description in enumerate(descriptions, start=1):
                dst.set_band_description(index, description)
            dst.update_tags(parent_raster=relative_path(source_path))


def split_chip_ids(chip_ids: list[str], seed: int) -> dict[str, list[str]]:
    """Create deterministic train/val/test splits."""
    shuffled = chip_ids[:]
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    n = len(shuffled)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    return {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
        "all": shuffled,
    }


def write_split_files(output_dir: Path, splits: dict[str, list[str]]) -> None:
    """Write HTC split text files."""
    for name, ids in splits.items():
        path = output_dir / f"{name}.txt"
        path.write_text("".join(f"{chip_id}\n" for chip_id in ids), encoding="utf-8")


def compute_image_stats(image_paths: list[Path], agl_paths: list[Path], stats_path: Path) -> dict[str, Any]:
    """Compute simple channel statistics for HTC preprocessing."""
    import rasterio

    channel_sum = None
    channel_sum_sq = None
    channel_count = 0
    agl_sum = 0.0
    agl_sum_sq = 0.0
    agl_count = 0

    for image_path, agl_path in zip(image_paths, agl_paths, strict=True):
        with rasterio.open(image_path) as image:
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

        with rasterio.open(agl_path) as agl:
            target = agl.read(1, masked=True).astype("float64")
            vals = target[(target > 0) & ~np.ma.getmaskarray(target)].compressed()
            agl_sum += vals.sum()
            agl_sum_sq += np.square(vals).sum()
            agl_count += len(vals)

    if channel_sum is None or channel_count == 0:
        raise RuntimeError("No valid image pixels available for stats.")

    image_mean = channel_sum / channel_count
    image_var = np.maximum(channel_sum_sq / channel_count - np.square(image_mean), 0)
    image_std = np.sqrt(image_var)
    if agl_count:
        agl_mean = agl_sum / agl_count
        agl_std = float(np.sqrt(max(agl_sum_sq / agl_count - agl_mean**2, 0)))
    else:
        agl_mean = float("nan")
        agl_std = float("nan")

    stats = {
        "image_mean": image_mean.tolist(),
        "image_std": image_std.tolist(),
        "agl_positive_mean": float(agl_mean),
        "agl_positive_std": float(agl_std),
        "image_pixel_count": int(channel_count),
        "agl_positive_pixel_count": int(agl_count),
    }
    with stats_path.open("wb") as file:
        pickle.dump(stats, file)
    return stats


def create_htc_chips(
    htc_dir: Path,
    base_id: str,
    image_path: Path,
    mask_path: Path,
    agl_path: Path,
    chip_size: int,
    chip_stride: int,
    min_positive_agl_pixels: int,
    seed: int,
) -> dict[str, Any]:
    """Create image/mask/AGL chips and split files for HTC-style training."""
    import rasterio
    from rasterio.windows import Window

    image_dir = htc_dir / "image"
    mask_dir = htc_dir / "mask"
    ndsm_dir = htc_dir / "ndsm"
    stats_dir = htc_dir / "stats"
    for directory in [image_dir, mask_dir, ndsm_dir, stats_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    chip_rows: list[dict[str, Any]] = []
    image_chip_paths: list[Path] = []
    agl_chip_paths: list[Path] = []

    with rasterio.open(image_path) as image, rasterio.open(mask_path) as mask, rasterio.open(agl_path) as agl:
        if not (
            image.width == mask.width == agl.width
            and image.height == mask.height == agl.height
            and image.transform == mask.transform == agl.transform
        ):
            raise RuntimeError("HTC full-scene image, mask, and AGL rasters are not aligned.")

        chip_index = 0
        for row_off in range(0, image.height - chip_size + 1, chip_stride):
            for col_off in range(0, image.width - chip_size + 1, chip_stride):
                window = Window(col_off, row_off, chip_size, chip_size)
                agl_arr = agl.read(1, window=window, masked=True)
                mask_arr = mask.read(1, window=window, masked=True)
                positive_agl = np.ma.filled(agl_arr > 0, False)
                positive_mask = np.ma.filled(mask_arr > 0, False)
                positive_agl_pixels = int(positive_agl.sum())
                mask_pixels = int(positive_mask.sum())
                if positive_agl_pixels < min_positive_agl_pixels or mask_pixels == 0:
                    continue

                chip_id = f"{base_id}_chip_{chip_index:06d}"
                image_chip = image_dir / f"{chip_id}_IMG.tif"
                mask_chip = mask_dir / f"{chip_id}_BLG.tif"
                agl_chip = ndsm_dir / f"{chip_id}_AGL.tif"

                write_window_raster(image_path, image_chip, window, ["red", "green", "blue"])
                write_window_raster(mask_path, mask_chip, window, ["building_mask"])
                write_window_raster(agl_path, agl_chip, window, ["agl_m"])

                chip_rows.append(
                    {
                        "chip_id": chip_id,
                        "row_off": row_off,
                        "col_off": col_off,
                        "positive_agl_pixels": positive_agl_pixels,
                        "building_mask_pixels": mask_pixels,
                        "image_path": relative_path(image_chip),
                        "mask_path": relative_path(mask_chip),
                        "agl_path": relative_path(agl_chip),
                    }
                )
                image_chip_paths.append(image_chip)
                agl_chip_paths.append(agl_chip)
                chip_index += 1

    if not chip_rows:
        raise RuntimeError(
            f"No HTC chips passed filtering for {base_id}. "
            f"Try lowering --min-positive-agl-pixels or checking mask/AGL overlap."
        )

    splits = split_chip_ids([row["chip_id"] for row in chip_rows], seed)
    write_split_files(htc_dir, splits)
    stats = compute_image_stats(image_chip_paths, agl_chip_paths, stats_dir / "image_stats.pickle")

    manifest_path = htc_dir / "chips_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(chip_rows[0]))
        writer.writeheader()
        writer.writerows(chip_rows)

    return {
        "chip_count": len(chip_rows),
        "train_chip_count": len(splits["train"]),
        "val_chip_count": len(splits["val"]),
        "test_chip_count": len(splits["test"]),
        "chips_manifest": relative_path(manifest_path),
        "image_stats_path": relative_path(stats_dir / "image_stats.pickle"),
        "image_mean": stats["image_mean"],
        "image_std": stats["image_std"],
        "agl_positive_mean": stats["agl_positive_mean"],
        "agl_positive_std": stats["agl_positive_std"],
    }


def write_htc_full_scene_files(
    template_path: Path,
    htc_dir: Path,
    base_id: str,
    building_mask: np.ndarray,
    agl: np.ndarray,
    tags: dict[str, str],
) -> tuple[Path, Path, Path, list[int]]:
    """Write full-scene HTC-style image, building mask, and AGL files."""
    full_scene_dir = htc_dir / "full_scene"
    full_scene_dir.mkdir(parents=True, exist_ok=True)

    image_path = full_scene_dir / f"{base_id}_IMG.tif"
    mask_path = full_scene_dir / f"{base_id}_BLG.tif"
    agl_path = full_scene_dir / f"{base_id}_AGL.tif"

    rgb_indexes = write_rgb_image(template_path, image_path, tags)
    write_single_band_raster(
        template_path=template_path,
        output_path=mask_path,
        array=building_mask,
        dtype="uint8",
        nodata=0,
        description="building_mask",
        tags=tags | {"htc_role": "building_mask"},
    )
    write_single_band_raster(
        template_path=template_path,
        output_path=agl_path,
        array=agl,
        dtype="float32",
        nodata=NODATA_VALUE,
        description="agl_m",
        tags=tags | {"htc_role": "agl_target"},
    )
    return image_path, mask_path, agl_path, rgb_indexes


def main() -> None:
    """Run the nDSM and HTC input build."""
    relaunch_inside_venv()
    args = parse_args()

    city = args.city
    lidar_project = args.lidar_project or DEFAULT_LIDAR_PROJECT[city]
    template_scene_id = args.template_scene_id or DEFAULT_TEMPLATE_SCENE_ID[city]
    output_label = output_label_for(city, lidar_project, args.output_label)
    footprint_path = Path(args.footprints) if args.footprints else DEFAULT_FOOTPRINT_PATH[city]
    if not footprint_path.is_absolute():
        footprint_path = PROJECT_ROOT / footprint_path

    log_path = configure_logging(output_label)

    try:
        import rasterio

        output_dir = HEIGHT_GENERATED_DIR / output_label / "lidar_ndsm"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{output_label}_lidar_ndsm_planet_aligned.tif"
        summary_path = output_dir / f"{output_label}_lidar_ndsm_planet_aligned_summary.csv"
        htc_dir = output_dir / "htc_dc_net" / template_scene_id
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(f"Output exists: {output_path}. Pass --overwrite to replace.")

        template_path = find_planet_template(city, template_scene_id)
        tile_paths = load_manifest_tiles(city, lidar_project)

        logging.info("Using Planet template: %s", relative_path(template_path))
        logging.info("Using %s LiDAR tiles from %s.", len(tile_paths), lidar_project)
        logging.info("Using footprints: %s", relative_path(footprint_path))

        with rasterio.open(template_path) as template:
            dsm, dtm_observed, ground_count, stats = accumulate_lidar_surfaces(
                tile_paths=tile_paths,
                template_crs=template.crs,
                transform=template.transform,
                width=template.width,
                height=template.height,
            )

        logging.info("Filling DTM gaps.")
        dtm_filled = fill_dtm(dtm_observed, args.ground_fill_distance_pixels)

        ndsm = dsm - dtm_filled
        ndsm = np.where(np.isfinite(ndsm), np.maximum(ndsm, 0), np.nan).astype(np.float32)

        logging.info("Rasterizing building mask.")
        building_mask = rasterize_building_mask(
            template_path=template_path,
            footprint_path=footprint_path,
            all_touched=args.all_touched_building_mask,
        )
        ndsm_buildings_only, min_agl_audit = enforce_minimum_building_agl(
            ndsm=ndsm,
            building_mask=building_mask,
            minimum_height_m=args.min_building_agl_m,
        )
        logging.info(
            "Applied %.2f m minimum AGL to %s finite nonpositive building pixels.",
            args.min_building_agl_m,
            min_agl_audit["building_agl_minimum_imputed_pixels"],
        )

        common_tags = {
            "city": city,
            "output_label": output_label,
            "lidar_project": lidar_project,
            "template_raster": relative_path(template_path),
            "building_footprints": relative_path(footprint_path),
            "created_utc": datetime.now(timezone.utc).isoformat(),
        }

        logging.info("Writing %s", relative_path(output_path))
        write_multiband_output(
            template_path=template_path,
            output_path=output_path,
            bands=diagnostic_bands(
                dsm=dsm,
                dtm_observed=dtm_observed,
                dtm_filled=dtm_filled,
                ndsm=ndsm,
                building_mask=building_mask,
                ndsm_buildings_only=ndsm_buildings_only,
            ),
            tags=common_tags,
        )
        verify_output(template_path, output_path)

        htc_summary: dict[str, Any] = {}
        if not args.no_htc_files:
            base_id = f"{output_label}_{template_scene_id}"
            logging.info("Writing HTC full-scene files under %s", relative_path(htc_dir))
            image_path, mask_path, agl_path, rgb_indexes = write_htc_full_scene_files(
                template_path=template_path,
                htc_dir=htc_dir,
                base_id=base_id,
                building_mask=building_mask,
                agl=ndsm_buildings_only,
                tags=common_tags,
            )
            for path in [image_path, mask_path, agl_path]:
                verify_output(template_path, path)
            htc_summary.update(
                {
                    "htc_dir": relative_path(htc_dir),
                    "htc_full_scene_image": relative_path(image_path),
                    "htc_full_scene_mask": relative_path(mask_path),
                    "htc_full_scene_agl": relative_path(agl_path),
                    "htc_rgb_source_band_indexes": ",".join(map(str, rgb_indexes)),
                }
            )

            if not args.no_chips:
                logging.info("Creating HTC chips.")
                chip_summary = create_htc_chips(
                    htc_dir=htc_dir,
                    base_id=base_id,
                    image_path=image_path,
                    mask_path=mask_path,
                    agl_path=agl_path,
                    chip_size=args.chip_size,
                    chip_stride=args.chip_stride,
                    min_positive_agl_pixels=args.min_positive_agl_pixels,
                    seed=args.seed,
                )
                htc_summary.update(chip_summary)

        with rasterio.open(template_path) as template:
            summary = {
                "city": city,
                "output_label": output_label,
                "template_scene_id": template_scene_id,
                "template_raster": relative_path(template_path),
                "output_raster": relative_path(output_path),
                "crs": str(template.crs),
                "width": template.width,
                "height": template.height,
                "resolution_x_m": abs(template.transform.a),
                "resolution_y_m": abs(template.transform.e),
                "lidar_project_used": lidar_project,
                "footprints": relative_path(footprint_path),
                "ground_fill_distance_pixels": args.ground_fill_distance_pixels,
                "ground_fill_distance_m": args.ground_fill_distance_pixels * abs(template.transform.a),
                "building_mask_all_touched": args.all_touched_building_mask,
                "dsm_observed_pixels": int(np.isfinite(dsm).sum()),
                "dtm_observed_pixels": int(np.isfinite(dtm_observed).sum()),
                "dtm_filled_pixels": int(np.isfinite(dtm_filled).sum()),
                "ndsm_valid_pixels": int(np.isfinite(ndsm).sum()),
                "building_mask_pixels": int((building_mask > 0).sum()),
                "ndsm_building_pixels": int(np.isfinite(ndsm_buildings_only).sum()),
                "chip_size": args.chip_size,
                "chip_stride": args.chip_stride,
                "min_positive_agl_pixels": args.min_positive_agl_pixels,
                "min_building_agl_m": args.min_building_agl_m,
                "seed": args.seed,
                "log_path": relative_path(log_path),
                **stats,
                **min_agl_audit,
                **htc_summary,
            }
        write_summary(summary_path, summary)
        logging.info("Wrote %s", relative_path(summary_path))
        logging.info("nDSM/HTC build completed successfully.")
    except Exception:
        logging.exception("nDSM/HTC build failed.")
        raise


if __name__ == "__main__":
    main()
