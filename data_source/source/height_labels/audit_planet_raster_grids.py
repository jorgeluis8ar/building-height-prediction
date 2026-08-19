#!/usr/bin/env python3
"""Audit downloaded Planet analytic raster grids before LiDAR rasterization."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import re
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIRECTORY = Path(__file__).resolve().parent
VENV_DIRECTORY = SCRIPT_DIRECTORY / "venv_height_labels"
VENV_PYTHON = VENV_DIRECTORY / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
VENV_MARKER = "HEIGHT_LABELS_VENV_ACTIVE"
DEFAULT_IMAGERY_ROOT = REPOSITORY_ROOT / "data_source/data/planet_imagery/source"
DEFAULT_OUTPUT_DIRECTORY = REPOSITORY_ROOT / (
    "data_source/data/height_labels/generated/planet_raster_grid_audit"
)
DETAIL_FIELDS = [
    "city_slug", "scene_id", "selected_asset_type", "file_path", "crs_wkt",
    "epsg_code", "pixel_width", "pixel_height", "raster_width",
    "raster_height", "bounds", "affine_transform", "band_count",
    "data_type", "nodata",
]
SUMMARY_FIELDS = [
    "city_slug", "scene_count", "epsg_codes", "same_projection_within_city",
    "same_complete_grid_within_city", "pixel_sizes", "raster_dimensions",
    "band_counts", "data_types", "nodata_values",
]


def relaunch_inside_venv() -> None:
    """Use the environment that already supports LiDAR raster processing."""
    if os.environ.get(VENV_MARKER) == "1" or Path(sys.prefix).absolute() == VENV_DIRECTORY.absolute():
        return
    if not VENV_PYTHON.is_file():
        raise SystemExit(f"ERROR: Missing height-label environment: {VENV_PYTHON}")
    environment = os.environ.copy()
    environment[VENV_MARKER] = "1"
    os.execve(str(VENV_PYTHON), [str(VENV_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]], environment)


relaunch_inside_venv()

import rasterio


def parse_arguments() -> argparse.Namespace:
    """Parse city and path overrides."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city-slug", action="append", dest="city_slugs")
    parser.add_argument("--imagery-root", type=Path, default=DEFAULT_IMAGERY_ROOT)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    return parser.parse_args()


def resolve_project_path(path: Path) -> Path:
    """Resolve relative paths from the repository and reject outside writes."""
    resolved = path.resolve() if path.is_absolute() else (REPOSITORY_ROOT / path).resolve()
    if not resolved.is_relative_to(REPOSITORY_ROOT):
        raise ValueError(f"Path is outside the repository: {resolved}")
    return resolved


def atomic_write(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    """Write a CSV atomically so an interrupted audit cannot appear complete."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def scene_id_from_name(path: Path) -> str:
    """Extract the stable Planet scene ID from a delivered analytic filename."""
    match = re.match(r"(.+?)_3B_AnalyticMS_SR(?:_8b)?_clip\.tif$", path.name)
    if not match:
        raise ValueError(f"Unexpected Planet analytic filename: {path.name}")
    return match.group(1)


def inspect_raster(city_slug: str, path: Path) -> dict[str, object]:
    """Read authoritative grid metadata from one analytic GeoTIFF header."""
    with rasterio.open(path) as dataset:
        if dataset.crs is None:
            raise RuntimeError(f"Raster has no CRS: {path}")
        transform = dataset.transform
        bounds = dataset.bounds
        return {
            "city_slug": city_slug,
            "scene_id": scene_id_from_name(path),
            "selected_asset_type": (
                "ortho_analytic_8b_sr" if "_SR_8b_" in path.name
                else "ortho_analytic_4b_sr"
            ),
            "file_path": str(path.relative_to(REPOSITORY_ROOT)),
            "crs_wkt": dataset.crs.to_wkt(),
            "epsg_code": dataset.crs.to_epsg() or "",
            "pixel_width": transform.a,
            "pixel_height": transform.e,
            "raster_width": dataset.width,
            "raster_height": dataset.height,
            "bounds": json.dumps([bounds.left, bounds.bottom, bounds.right, bounds.top]),
            "affine_transform": json.dumps([
                transform.a, transform.b, transform.c,
                transform.d, transform.e, transform.f,
            ]),
            "band_count": dataset.count,
            "data_type": ";".join(dataset.dtypes),
            "nodata": ";".join("None" if value is None else str(value) for value in dataset.nodatavals),
        }


def summarize(city_slug: str, rows: list[dict[str, object]]) -> dict[str, object]:
    """Compare both CRS identity and the stricter complete raster grid."""
    epsgs = sorted({str(row["epsg_code"]) for row in rows})
    crs_values = {str(row["crs_wkt"]) for row in rows}
    complete_grids = {
        (
            row["crs_wkt"], row["pixel_width"], row["pixel_height"],
            row["raster_width"], row["raster_height"], row["bounds"],
            row["affine_transform"],
        )
        for row in rows
    }
    return {
        "city_slug": city_slug,
        "scene_count": len(rows),
        "epsg_codes": ";".join(epsgs),
        "same_projection_within_city": len(crs_values) == 1,
        "same_complete_grid_within_city": len(complete_grids) == 1,
        "pixel_sizes": ";".join(sorted({f"{row['pixel_width']},{row['pixel_height']}" for row in rows})),
        "raster_dimensions": ";".join(sorted({f"{row['raster_width']}x{row['raster_height']}" for row in rows})),
        "band_counts": ";".join(sorted({str(row["band_count"]) for row in rows})),
        "data_types": ";".join(sorted({str(row["data_type"]) for row in rows})),
        "nodata_values": ";".join(sorted({str(row["nodata"]) for row in rows})),
    }


def main() -> None:
    """Audit every analytic SR GeoTIFF for the requested cities."""
    arguments = parse_arguments()
    imagery_root = resolve_project_path(arguments.imagery_root)
    output_directory = resolve_project_path(arguments.output_directory)
    city_slugs = arguments.city_slugs or ["los_angeles", "new_york_city"]
    detail_rows = []
    summary_rows = []
    for city_slug in city_slugs:
        city_directory = imagery_root / city_slug
        files = sorted(city_directory.rglob("*AnalyticMS_SR*_clip.tif"))
        if not files:
            raise FileNotFoundError(f"No analytic SR GeoTIFFs found for {city_slug}")
        city_rows = [inspect_raster(city_slug, path) for path in files]
        if len({row["scene_id"] for row in city_rows}) != len(city_rows):
            raise RuntimeError(f"Duplicate analytic scene IDs for {city_slug}")
        detail_rows.extend(city_rows)
        summary_rows.append(summarize(city_slug, city_rows))
    atomic_write(output_directory / "nyc_la_planet_raster_metadata.csv", DETAIL_FIELDS, detail_rows)
    atomic_write(output_directory / "nyc_la_planet_raster_grid_summary.csv", SUMMARY_FIELDS, summary_rows)
    print(f"Analytic scene rasters audited: {len(detail_rows)}")
    for row in summary_rows:
        print(
            f"{row['city_slug']}: scenes={row['scene_count']} epsg={row['epsg_codes']} "
            f"same_projection={row['same_projection_within_city']} "
            f"same_grid={row['same_complete_grid_within_city']}"
        )


if __name__ == "__main__":
    main()
