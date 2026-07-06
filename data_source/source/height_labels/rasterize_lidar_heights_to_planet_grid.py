"""
Rasterize LiDAR Building Heights Onto PlanetScope Grids

Environment: data_source/source/height_labels/venv_height_labels

Inputs:
    - data_source/data/height_labels/generated/<city_slug>/
      lidar_building_heights_merged_all.gpkg
    - data_source/data/planet_imagery/source/<city_slug>/**/
      *_3B_AnalyticMS_SR*_clip.tif

Outputs:
    - data_source/data/height_labels/generated/<city_slug>/
      planet_aligned_lidar_rasters/<scene_id>_
      lidar_building_heights_merged_all_planet_aligned.tif
    - data_source/data/height_labels/generated/
      planet_aligned_lidar_raster_summary.csv

Description:
    This script converts building-level LiDAR height labels from polygons into
    rasters that line up exactly with the downloaded PlanetScope imagery.

    The key idea is simple: each PlanetScope TIFF is treated as the template.
    We copy its CRS, affine transform, width, height, and pixel size, then burn
    the footprint height values into that exact grid. This avoids tiny
    alignment errors that would happen if we independently rebuilt the grid.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import logging
import os
from pathlib import Path
import sys
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
VENV_DIR = SCRIPT_DIR / "venv_height_labels"
VENV_PYTHON = VENV_DIR / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
VENV_MARKER = "HEIGHT_LABELS_VENV_ACTIVE"


CITY_SLUGS = ("los_angeles", "new_york_city")
PLANET_SOURCE_DIR = PROJECT_ROOT / "data_source/data/planet_imagery/source"
HEIGHT_GENERATED_DIR = PROJECT_ROOT / "data_source/data/height_labels/generated"

HEIGHT_BANDS = [
    "height_label_m",
    "height_mean_m",
    "height_median_m",
    "height_p50_m",
    "height_p75_m",
    "height_p90_m",
    "height_p95_m",
    "height_max_clean_m",
    "height_max_m",
    "local_ground_m",
    "usable_for_training_code",
    "quality_tier_code",
]

QUALITY_TIER_CODES = {
    "Reject": 0.0,
    "C": 1.0,
    "B": 2.0,
    "A": 3.0,
}

NODATA_VALUE = -9999.0


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
        description="Rasterize merged LiDAR height labels onto PlanetScope grids."
    )
    parser.add_argument(
        "--city",
        action="append",
        choices=CITY_SLUGS,
        help="City slug to process. May be repeated. Default: LA and NYC.",
    )
    parser.add_argument(
        "--scene-id",
        action="append",
        help=(
            "Specific Planet scene ID to process. May be repeated. "
            "Default: all downloaded LA/NYC PlanetScope clipped scenes."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output rasters.",
    )
    parser.add_argument(
        "--all-touched",
        action="store_true",
        help=(
            "Burn a polygon into every pixel it touches. Default is false, so "
            "a pixel receives a height only when its center is inside the polygon."
        ),
    )
    return parser.parse_args()


def configure_logging() -> Path:
    """Create an honest run log in the generated height-label folder."""
    HEIGHT_GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_path = HEIGHT_GENERATED_DIR / f"rasterize_lidar_heights_to_planet_grid_{timestamp}.log"
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
    """Return a project-relative path for logs and summary tables."""
    return path.resolve().relative_to(PROJECT_ROOT).as_posix()


def scene_id_from_planet_tif(path: Path) -> str:
    """Extract the Planet scene ID from a standard Planet TIFF filename."""
    marker = "_3B_"
    if marker not in path.name:
        raise ValueError(f"Could not parse scene ID from Planet TIFF name: {path}")
    return path.name.split(marker, maxsplit=1)[0]


def find_planet_templates(city: str, scene_ids: set[str] | None) -> list[Path]:
    """Find downloaded clipped PlanetScope analytic surface-reflectance TIFFs."""
    city_dir = PLANET_SOURCE_DIR / city
    if not city_dir.exists():
        raise FileNotFoundError(f"Missing Planet source city folder: {city_dir}")

    templates = []
    for path in sorted(city_dir.rglob("*_3B_AnalyticMS_SR*_clip.tif")):
        scene_id = scene_id_from_planet_tif(path)
        if scene_ids is not None and scene_id not in scene_ids:
            continue
        templates.append(path)

    if not templates:
        detail = f" for scene IDs {sorted(scene_ids)}" if scene_ids else ""
        raise FileNotFoundError(f"No clipped PlanetScope TIFFs found for {city}{detail}.")

    seen: set[str] = set()
    unique_templates: list[Path] = []
    for path in templates:
        scene_id = scene_id_from_planet_tif(path)
        if scene_id in seen:
            raise RuntimeError(
                "Multiple Planet templates found for one scene. "
                f"Scene {scene_id}, duplicate path {path}"
            )
        seen.add(scene_id)
        unique_templates.append(path)

    return unique_templates


def read_lidar_gpkg(city: str) -> Any:
    """Load the merged LiDAR height GeoPackage for one city."""
    import geopandas as gpd

    gpkg_path = HEIGHT_GENERATED_DIR / city / "lidar_building_heights_merged_all.gpkg"
    if not gpkg_path.exists():
        raise FileNotFoundError(f"Missing merged LiDAR height GeoPackage: {gpkg_path}")

    logging.info("Reading %s", relative_path(gpkg_path))
    gdf = gpd.read_file(gpkg_path)
    if gdf.empty:
        raise RuntimeError(f"LiDAR height GeoPackage is empty: {gpkg_path}")
    if gdf.crs is None:
        raise RuntimeError(f"LiDAR height GeoPackage has no CRS: {gpkg_path}")
    if "geometry" not in gdf:
        raise RuntimeError(f"LiDAR height GeoPackage has no geometry column: {gpkg_path}")

    missing_columns = [column for column in HEIGHT_BANDS if column not in gdf.columns]
    missing_columns = [
        column
        for column in missing_columns
        if column not in {"usable_for_training_code", "quality_tier_code"}
    ]
    if missing_columns:
        raise RuntimeError(
            f"Missing required height columns in {gpkg_path}: {', '.join(missing_columns)}"
        )

    # Convert nonnumeric QA labels to numeric bands so they can live in a GeoTIFF.
    if "usable_for_training" in gdf.columns:
        gdf["usable_for_training_code"] = gdf["usable_for_training"].fillna(False).astype(float)
    else:
        gdf["usable_for_training_code"] = 0.0

    if "quality_tier" in gdf.columns:
        gdf["quality_tier_code"] = (
            gdf["quality_tier"].map(QUALITY_TIER_CODES).fillna(NODATA_VALUE).astype(float)
        )
    else:
        gdf["quality_tier_code"] = NODATA_VALUE

    return gdf


def geometry_value_pairs(gdf: Any, column: str) -> list[tuple[Any, float]]:
    """Create clean geometry/value pairs for rasterio.features.rasterize."""
    values = gdf[[column, "geometry"]].copy()
    values = values[values.geometry.notna()]
    values = values[~values.geometry.is_empty]
    values = values[values[column].notna()]
    values = values[values[column] != NODATA_VALUE]
    return [(geometry, float(value)) for geometry, value in zip(values.geometry, values[column])]


def rasterize_template(
    city: str,
    lidar_gdf: Any,
    template_path: Path,
    overwrite: bool,
    all_touched: bool,
) -> dict[str, Any]:
    """Rasterize every requested LiDAR band to one Planet template grid."""
    import rasterio
    from rasterio.features import rasterize

    scene_id = scene_id_from_planet_tif(template_path)
    output_dir = HEIGHT_GENERATED_DIR / city / "planet_aligned_lidar_rasters"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{scene_id}_lidar_building_heights_merged_all_planet_aligned.tif"

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}. Pass --overwrite to replace it."
        )

    with rasterio.open(template_path) as template:
        template_crs = template.crs
        if template_crs is None:
            raise RuntimeError(f"Planet template has no CRS: {template_path}")

        logging.info(
            "Template %s: CRS=%s, size=%sx%s, resolution=%s",
            relative_path(template_path),
            template_crs,
            template.width,
            template.height,
            template.res,
        )

        projected = lidar_gdf.to_crs(template_crs)

        profile = template.profile.copy()
        profile.update(
            driver="GTiff",
            count=len(HEIGHT_BANDS),
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

        band_valid_pixels: dict[str, int] = {}
        with rasterio.open(output_path, "w", **profile) as output:
            for band_index, column in enumerate(HEIGHT_BANDS, start=1):
                pairs = geometry_value_pairs(projected, column)
                if not pairs:
                    raise RuntimeError(
                        f"No valid geometries/values available for band {column} in {city}."
                    )

                array = rasterize(
                    pairs,
                    out_shape=(template.height, template.width),
                    transform=template.transform,
                    fill=NODATA_VALUE,
                    dtype="float32",
                    all_touched=all_touched,
                )
                output.write(array, band_index)
                output.set_band_description(band_index, column)
                band_valid_pixels[column] = int((array != NODATA_VALUE).sum())

            output.update_tags(
                city=city,
                scene_id=scene_id,
                source_vector=relative_path(
                    HEIGHT_GENERATED_DIR / city / "lidar_building_heights_merged_all.gpkg"
                ),
                template_raster=relative_path(template_path),
                all_touched=str(all_touched),
                created_utc=datetime.now(timezone.utc).isoformat(),
            )

    verify_alignment(template_path, output_path)
    return {
        "city": city,
        "scene_id": scene_id,
        "template_raster": relative_path(template_path),
        "output_raster": relative_path(output_path),
        "crs": str(template_crs),
        "width": profile["width"],
        "height": profile["height"],
        "resolution_x_m": abs(profile["transform"].a),
        "resolution_y_m": abs(profile["transform"].e),
        "band_count": len(HEIGHT_BANDS),
        "height_label_valid_pixels": band_valid_pixels["height_label_m"],
        "height_p95_valid_pixels": band_valid_pixels["height_p95_m"],
        "height_max_valid_pixels": band_valid_pixels["height_max_m"],
        "all_touched": all_touched,
    }


def verify_alignment(template_path: Path, output_path: Path) -> None:
    """Fail loudly if the output raster does not exactly match its Planet grid."""
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
                f"Raster alignment check failed for {output_path}: {', '.join(failed)}"
            )


def write_summary(rows: list[dict[str, Any]]) -> Path:
    """Write one project-level CSV describing every raster created."""
    summary_path = HEIGHT_GENERATED_DIR / "planet_aligned_lidar_raster_summary.csv"
    if not rows:
        raise RuntimeError("No raster summary rows to write.")

    fieldnames = [
        "city",
        "scene_id",
        "template_raster",
        "output_raster",
        "crs",
        "width",
        "height",
        "resolution_x_m",
        "resolution_y_m",
        "band_count",
        "height_label_valid_pixels",
        "height_p95_valid_pixels",
        "height_max_valid_pixels",
        "all_touched",
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return summary_path


def main() -> None:
    """Run the rasterization workflow."""
    relaunch_inside_venv()
    args = parse_args()
    log_path = configure_logging()

    try:
        cities = args.city or list(CITY_SLUGS)
        scene_ids = set(args.scene_id) if args.scene_id else None
        summary_rows: list[dict[str, Any]] = []

        logging.info("Starting Planet-aligned LiDAR rasterization.")
        logging.info("Log path: %s", relative_path(log_path))

        for city in cities:
            templates = find_planet_templates(city, scene_ids)
            lidar_gdf = read_lidar_gpkg(city)
            logging.info("Processing %s templates for %s.", len(templates), city)
            for template_path in templates:
                summary_rows.append(
                    rasterize_template(
                        city=city,
                        lidar_gdf=lidar_gdf,
                        template_path=template_path,
                        overwrite=args.overwrite,
                        all_touched=args.all_touched,
                    )
                )

        summary_path = write_summary(summary_rows)
        logging.info("Wrote %s", relative_path(summary_path))
        logging.info("Rasterization completed successfully.")
    except Exception:
        logging.exception("Rasterization failed.")
        raise


if __name__ == "__main__":
    main()
