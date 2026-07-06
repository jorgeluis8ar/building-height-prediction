"""
Select 5km-Intersecting Building Footprints

Environment: data_source/source/building_footprints/venv_building_footprints

Requires (inputs from earlier stages):
    - README.md
    - data_source/data/building_footprints/source/<city_slug>/
    - data_source/data/city_aois/generated/city_buffers_5km_by_city/<city_slug>_5km.geojson

Produces (outputs for later stages):
    - data_source/data/building_footprints/generated/<city_slug>/<city_slug>_building_footprints_5km.gpkg
    - data_source/data/building_footprints/generated/building_footprints_clip_summary.csv

Description:
    Reads each current city from README.md, loads the raw building-footprint
    source file for that city, selects every footprint that intersects the
    city's 5km AOI, preserves the full original footprint geometry, and writes
    every city's processed footprint file in the same GeoPackage format.

Usage:
    python3 data_source/source/building_footprints/clip_building_footprints.py

Expected runtime: depends on the size of the raw city footprint files
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
VENV_DIR = SCRIPT_DIR / "venv_building_footprints"
VENV_PYTHON = VENV_DIR / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
VENV_MARKER = "BUILDING_FOOTPRINTS_VENV_ACTIVE"


def relaunch_inside_venv() -> None:
    """
    Relaunch this script with the local virtual environment Python.

    This keeps the command simple for the user:
        python3 data_source/source/building_footprints/clip_building_footprints.py

    If that command is run with the system Python, the script immediately
    re-executes itself with the folder-local venv Python.
    """
    if os.environ.get(VENV_MARKER) == "1":
        return

    current_python = Path(sys.executable).absolute()
    expected_python = VENV_PYTHON.absolute()

    if current_python == expected_python or Path(sys.prefix).absolute() == VENV_DIR.absolute():
        os.environ[VENV_MARKER] = "1"
        return

    if not VENV_PYTHON.exists():
        print("ERROR: Missing building_footprints virtual environment.")
        print(f"Expected Python executable: {VENV_PYTHON}")
        print("Create it with:")
        print("  python3 -m venv data_source/source/building_footprints/venv_building_footprints")
        print("  data_source/source/building_footprints/venv_building_footprints/bin/python -m pip install -r data_source/source/building_footprints/requirements.txt")
        sys.exit(1)

    env = os.environ.copy()
    env[VENV_MARKER] = "1"
    os.execve(
        str(VENV_PYTHON),
        [str(VENV_PYTHON), str(Path(__file__).resolve())] + sys.argv[1:],
        env,
    )


relaunch_inside_venv()

import geopandas as gpd
import pandas as pd
import pyogrio
from pyogrio.errors import DataSourceError
from shapely.geometry.base import BaseGeometry
from shapely.validation import make_valid


SOURCE_SUFFIXES = {
    ".geojson",
    ".json",
    ".gpkg",
    ".shp",
    ".gml",
    ".gdb",
}
IGNORED_NAMES = {".DS_Store"}
AOI_CRS = "EPSG:4326"
OUTPUT_DRIVER = "GPKG"
OUTPUT_LAYER = "building_footprints_5km"
OUTPUT_SUFFIX = "_building_footprints_5km.gpkg"


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(
        description=(
            "Select raw city building footprints that intersect the current "
            "5km city AOIs, preserve full footprint geometry, and write every "
            "processed output as GeoPackage."
        )
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=project_root / "data_source" / "data" / "building_footprints" / "source",
        help="City-specific raw building footprint source folder.",
    )
    parser.add_argument(
        "--aoi-dir",
        type=Path,
        default=project_root
        / "data_source"
        / "data"
        / "city_aois"
        / "generated"
        / "city_buffers_5km_by_city",
        help="Folder containing <city_slug>_5km.geojson AOI files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "data_source" / "data" / "building_footprints" / "generated",
        help="City-specific generated building footprint output folder.",
    )
    parser.add_argument(
        "--city",
        action="append",
        dest="cities",
        help="Limit the run to a city slug. May be supplied more than once.",
    )
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Write an empty output instead of failing when no footprints intersect the AOI.",
    )
    return parser.parse_args()


def slugify_city_name(value: str) -> str:
    return value.strip().lower().replace(" ", "_")


def current_city_slugs_from_readme(readme_path: Path) -> list[str]:
    """
    Read the active city list from the README Current Cities table.

    The README is the project-facing source of truth. Reading it here prevents
    old or experimental cities from being processed accidentally.
    """
    if not readme_path.exists():
        raise FileNotFoundError(f"Missing README file: {readme_path}")

    in_current_cities = False
    city_slugs: list[str] = []
    for raw_line in readme_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()

        if line == "### Current Cities":
            in_current_cities = True
            continue
        if in_current_cities and line.startswith("### "):
            break
        if not in_current_cities:
            continue
        if not line.startswith("|") or "Cities" in line or "---" in line:
            continue

        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) < 2:
            continue
        for city_name in cells[1].split(","):
            city_name = city_name.strip()
            if city_name:
                city_slugs.append(slugify_city_name(city_name))

    if not city_slugs:
        raise ValueError("No current cities were found in README.md")
    return city_slugs


def source_candidates(city_source_dir: Path) -> list[Path]:
    """
    Find likely vector data sources for one city.

    Sidecar files such as .dbf, .prj, and .shx are intentionally skipped; the
    .shp file is the readable dataset entry point. A .gdb folder is treated as
    a dataset even though it is a directory.
    """
    candidates: list[Path] = []
    if not city_source_dir.exists():
        return candidates

    for path in sorted(city_source_dir.rglob("*")):
        if path.name in IGNORED_NAMES or path.name.startswith("._"):
            continue
        if path.is_dir() and path.suffix.lower() == ".gdb":
            candidates.append(path)
        elif path.is_file() and path.suffix.lower() in SOURCE_SUFFIXES:
            candidates.append(path)
        elif path.is_file() and path.suffix == "" and looks_like_geojson(path):
            candidates.append(path)

    return candidates


def looks_like_geojson(path: Path) -> bool:
    """
    Detect extensionless GeoJSON files without loading the full file.

    Some downloaded sources, especially Google Open Buildings extracts, arrive
    without a .geojson suffix even though the content is valid GeoJSON.
    """
    try:
        with path.open("r", encoding="utf-8") as handle:
            return "FeatureCollection" in handle.read(300)
    except UnicodeDecodeError:
        return False


def readable_layers(path: Path) -> list[str | None]:
    """
    Return layer names for a vector dataset.

    Single-layer files can be read with layer=None. Multi-layer containers such
    as GeoPackage and FileGDB need explicit layer names.
    """
    if path.suffix.lower() in {".gpkg", ".gdb"}:
        layers = pyogrio.list_layers(path)
        return [str(layer_name) for layer_name, geometry_type in layers if geometry_type]
    return [None]


def geometry_is_polygonal(geometry: BaseGeometry | None) -> bool:
    if geometry is None or geometry.is_empty:
        return False
    return geometry.geom_type in {"Polygon", "MultiPolygon"}


def repair_geometry(geometry: BaseGeometry | None) -> BaseGeometry | None:
    """
    Repair invalid geometries before intersection.

    Some official footprint files contain self-intersections or holes that GEOS
    cannot assign to a shell. `make_valid` preserves geometry better than a
    blind zero-width buffer and prevents one bad feature from stopping a whole
    city run.
    """
    if geometry is None or geometry.is_empty:
        return geometry
    if geometry.is_valid:
        return geometry
    return make_valid(geometry)


def read_and_clip_candidate(
    source_path: Path,
    aoi: gpd.GeoDataFrame,
    city_slug: str,
) -> tuple[gpd.GeoDataFrame, list[dict]]:
    """
    Read one source dataset and select full polygons intersecting the city AOI.

    Each layer is attempted separately. Failures are returned as summary rows
    so the final log can explain exactly what happened.
    """
    selected_layers: list[gpd.GeoDataFrame] = []
    layer_summaries: list[dict] = []

    try:
        layers = readable_layers(source_path)
    except Exception as error:
        layer_summaries.append(
            layer_summary(city_slug, source_path, None, "layer_error", 0, 0, str(error))
        )
        return gpd.GeoDataFrame(), layer_summaries

    for layer in layers:
        layer_label = layer or source_path.stem
        try:
            source_gdf = gpd.read_file(source_path, layer=layer)
        except Exception as error:
            layer_summaries.append(
                layer_summary(
                    city_slug,
                    source_path,
                    layer_label,
                    "read_error",
                    0,
                    0,
                    str(error),
                )
            )
            continue

        source_count = len(source_gdf)
        if source_gdf.empty:
            layer_summaries.append(
                layer_summary(city_slug, source_path, layer_label, "empty_source", 0, 0, "")
            )
            continue
        if source_gdf.crs is None:
            layer_summaries.append(
                layer_summary(
                    city_slug,
                    source_path,
                    layer_label,
                    "missing_crs",
                    source_count,
                    0,
                    "Source layer has no CRS; refusing to guess.",
                )
            )
            continue

        source_gdf = source_gdf[source_gdf.geometry.notna()].copy()
        source_gdf["geometry"] = source_gdf.geometry.map(repair_geometry)
        source_gdf = source_gdf[source_gdf.geometry.map(geometry_is_polygonal)].copy()
        if source_gdf.empty:
            layer_summaries.append(
                layer_summary(
                    city_slug,
                    source_path,
                    layer_label,
                    "no_polygon_features",
                    source_count,
                    0,
                    "",
                )
            )
            continue

        source_gdf = source_gdf.to_crs(aoi.crs)
        aoi_geometry = aoi.geometry.union_all()
        intersects_aoi = source_gdf.geometry.intersects(aoi_geometry)
        selected = source_gdf[intersects_aoi].copy()
        selected = selected[selected.geometry.notna()].copy()
        selected = selected[~selected.geometry.is_empty].copy()
        selected_count = len(selected)

        if selected_count == 0:
            layer_summaries.append(
                layer_summary(
                    city_slug,
                    source_path,
                    layer_label,
                    "no_intersection",
                    source_count,
                    0,
                    "",
                )
            )
            continue

        selected["city_slug"] = city_slug
        selected["source_dataset"] = source_path.name
        selected["source_layer"] = layer_label
        selected["aoi_selection_rule"] = "intersects_5km_aoi_preserve_full_geometry"
        selected_layers.append(selected)
        layer_summaries.append(
            layer_summary(
                city_slug,
                source_path,
                layer_label,
                "selected_intersecting",
                source_count,
                selected_count,
                "",
            )
        )

    if not selected_layers:
        return gpd.GeoDataFrame(), layer_summaries

    combined = pd.concat(selected_layers, ignore_index=True)
    return gpd.GeoDataFrame(combined, geometry="geometry", crs=aoi.crs), layer_summaries


def layer_summary(
    city_slug: str,
    source_path: Path,
    layer: str | None,
    status: str,
    source_features: int,
    selected_features: int,
    message: str,
) -> dict:
    return {
        "city_slug": city_slug,
        "source_path": source_path.as_posix(),
        "source_layer": layer or "",
        "status": status,
        "source_features": source_features,
        "selected_features": selected_features,
        "message": message,
    }


def load_aoi(aoi_path: Path) -> gpd.GeoDataFrame:
    aoi = gpd.read_file(aoi_path)
    if aoi.empty:
        raise ValueError(f"AOI file is empty: {aoi_path}")
    if aoi.crs is None:
        aoi = aoi.set_crs(AOI_CRS)
    else:
        aoi = aoi.to_crs(AOI_CRS)
    return aoi[["geometry"]].dissolve()


def clean_output_directory(city_output_dir: Path) -> None:
    """
    Remove old generated files for one city before writing the new GeoPackage.

    This touches only data_source/data/building_footprints/generated/<city>/,
    never the raw source folder.
    """
    city_output_dir.mkdir(parents=True, exist_ok=True)
    for path in city_output_dir.iterdir():
        if path.is_file() and (
            path.name.endswith(OUTPUT_SUFFIX) or path.name.endswith(".tmp.gpkg")
        ):
            path.unlink()


def write_city_output(city_gdf: gpd.GeoDataFrame, output_path: Path) -> None:
    """
    Atomically write a city GeoPackage when possible.

    GeoPackage writing creates one file, so the script writes to a temporary
    file first and then replaces the final file. This avoids leaving a half
    written final output if the process fails.
    """
    temporary_path = output_path.with_suffix(".tmp.gpkg")
    if temporary_path.exists():
        temporary_path.unlink()
    city_gdf.to_file(temporary_path, layer=OUTPUT_LAYER, driver=OUTPUT_DRIVER)
    temporary_path.replace(output_path)


def write_summary(summary_rows: Iterable[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "city_slug",
        "source_path",
        "source_layer",
        "status",
        "source_features",
        "selected_features",
        "message",
    ]
    temporary_path = output_path.with_suffix(".tmp.csv")
    with temporary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    temporary_path.replace(output_path)


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[3]
    readme_path = project_root / "README.md"
    current_city_slugs = current_city_slugs_from_readme(readme_path)

    if args.cities:
        requested = set(args.cities)
        unknown = requested - set(current_city_slugs)
        if unknown:
            raise SystemExit(f"City slug(s) not listed in README current cities: {sorted(unknown)}")
        city_slugs = [slug for slug in current_city_slugs if slug in requested]
    else:
        city_slugs = current_city_slugs

    if not city_slugs:
        raise SystemExit("No cities selected for clipping.")

    start_time = time.time()
    all_summary_rows: list[dict] = []
    failed_cities: list[str] = []

    print(f"Selecting AOI-intersecting building footprints for {len(city_slugs)} cities.")
    print(f"Output format: {OUTPUT_DRIVER} ({OUTPUT_SUFFIX})")

    for city_number, city_slug in enumerate(city_slugs, start=1):
        print(f"[{city_number}/{len(city_slugs)}] {city_slug}", flush=True)

        city_source_dir = args.source_dir / city_slug
        aoi_path = args.aoi_dir / f"{city_slug}_5km.geojson"
        city_output_dir = args.output_dir / city_slug
        output_path = city_output_dir / f"{city_slug}{OUTPUT_SUFFIX}"

        if not city_source_dir.exists():
            failed_cities.append(city_slug)
            all_summary_rows.append(
                layer_summary(city_slug, city_source_dir, None, "missing_source_dir", 0, 0, "")
            )
            continue
        if not aoi_path.exists():
            failed_cities.append(city_slug)
            all_summary_rows.append(
                layer_summary(city_slug, aoi_path, None, "missing_aoi", 0, 0, "")
            )
            continue

        candidates = source_candidates(city_source_dir)
        if not candidates:
            failed_cities.append(city_slug)
            all_summary_rows.append(
                layer_summary(city_slug, city_source_dir, None, "no_readable_source", 0, 0, "")
            )
            continue

        aoi = load_aoi(aoi_path)
        city_selected_layers: list[gpd.GeoDataFrame] = []

        for source_path in candidates:
            selected, source_summary_rows = read_and_clip_candidate(
                source_path=source_path,
                aoi=aoi,
                city_slug=city_slug,
            )
            all_summary_rows.extend(source_summary_rows)
            if not selected.empty:
                city_selected_layers.append(selected)

        if not city_selected_layers:
            if args.allow_empty:
                city_result = gpd.GeoDataFrame(
                    {"city_slug": pd.Series(dtype="str")},
                    geometry=gpd.GeoSeries([], crs=aoi.crs),
                )
            else:
                failed_cities.append(city_slug)
                print(f"  ERROR: no AOI-intersecting building footprints produced for {city_slug}")
                continue
        else:
            city_result = gpd.GeoDataFrame(
                pd.concat(city_selected_layers, ignore_index=True),
                geometry="geometry",
                crs=aoi.crs,
            )
            city_result = city_result.reset_index(drop=True)
            city_result["building_footprint_id"] = [
                f"{city_slug}_{index + 1:08d}" for index in city_result.index
            ]
            leading_columns = [
                "building_footprint_id",
                "city_slug",
                "source_dataset",
                "source_layer",
                "geometry",
            ]
            other_columns = [
                column for column in city_result.columns if column not in leading_columns
            ]
            city_result = city_result[leading_columns + other_columns]

        clean_output_directory(city_output_dir)
        write_city_output(city_result, output_path)
        print(f"  WROTE {output_path} ({len(city_result):,} features)", flush=True)

    summary_path = args.output_dir / "building_footprints_clip_summary.csv"
    write_summary(all_summary_rows, summary_path)
    print(f"WROTE {summary_path} ({len(all_summary_rows):,} source/layer rows)")

    elapsed = time.time() - start_time
    if failed_cities:
        raise SystemExit(
            f"FAILED for {len(failed_cities)} city/cities: {failed_cities}. "
            f"See {summary_path} for details. Elapsed: {elapsed:.1f}s"
        )

    print(f"SUCCESS: selected footprints for {len(city_slugs)} city/cities in {elapsed:.1f}s")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted. Existing completed city outputs were preserved.")
        sys.exit(130)
