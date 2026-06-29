"""
Derive LiDAR Building-Height Labels for New York City and Los Angeles

Environment: data_source/source/height_labels/venv_height_labels

Requires (inputs from earlier stages):
    - data_source/data/height_labels/generated/usgs_3dep_tile_manifest.csv
      (produced by download_usgs_3dep_lidar.py)
    - data_source/data/building_footprints/generated/<city_slug>/
      <city_slug>_building_footprints_5km.gpkg
      (produced by clip_building_footprints.py)
    - data_source/data/height_labels/source/<city_slug>/usgs_3dep/
      <project_name>/*.laz
      (produced by download_usgs_3dep_lidar.py)

Produces (outputs for later stages):
    - data_source/data/height_labels/generated/lidar_tile_inventory.csv
    - data_source/data/height_labels/generated/<city_slug>/
      building_height_diagnostics_sample.csv
    - data_source/data/height_labels/generated/<city_slug>/temp_samples/
      building_height_diagnostics_sample_run_XX.csv
    - data_source/data/height_labels/generated/<city_slug>/
      height_definition_comparison.csv
    - data_source/data/height_labels/generated/<city_slug>/
      quality_tier_summary.csv
    - data_source/data/height_labels/generated/<city_slug>/
      lidar_building_heights.gpkg

Description:
    Creates footprint-level LiDAR-derived height candidates from USGS 3DEP
    point clouds. The script starts safely with a diagnostic sample of 500
    buildings per city. Use --all-buildings only after reviewing the diagnostic
    output, because full-city point-cloud processing can take a long time.

Usage:
    python3 data_source/source/height_labels/derive_lidar_building_heights.py
    python3 data_source/source/height_labels/derive_lidar_building_heights.py --sample-size 100
    python3 data_source/source/height_labels/derive_lidar_building_heights.py --sample-runs 15
    python3 data_source/source/height_labels/derive_lidar_building_heights.py --all-buildings
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import logging
import math
import os
from pathlib import Path
import sys
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
VENV_DIR = SCRIPT_DIR / "venv_height_labels"
VENV_PYTHON = VENV_DIR / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
VENV_MARKER = "HEIGHT_LABELS_VENV_ACTIVE"


CITY_CONFIG = {
    "new_york_city": {
        "lidar_epsg": 6347,
        "official_height_field": "HEIGHT_ROO",
        "official_ground_field": "GROUND_ELE",
        "date_field": "CONSTRUCTI",
        "lidar_collect_start": "2013-08-01",
        "lidar_collect_end": "2014-04-30",
    },
    "los_angeles": {
        "lidar_epsg": 6340,
        "official_height_field": "HEIGHT",
        "official_ground_field": "ELEV",
        "date_field": "DATE_",
        "lidar_collect_start": "2023-01-01",
        "lidar_collect_end": "2024-01-31",
    },
}

ROOF_CLASSES = {1, 6}
GROUND_CLASSES = {2}
EXCLUDED_CLASSES = {7, 9, 17, 18}
HEIGHT_DEFINITION = "lidar_ndsm_roof_p90_minus_local_ground"
MIN_NONZERO_BUILDING_HEIGHT_M = 2.4


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
        description="Derive building-level LiDAR height diagnostics for NYC and LA."
    )
    parser.add_argument(
        "--city",
        action="append",
        choices=sorted(CITY_CONFIG),
        help="Limit the run to one or more city slugs. May be repeated.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=500,
        help="Diagnostic sample size per city. Default: 500.",
    )
    parser.add_argument(
        "--all-buildings",
        action="store_true",
        help="Process every footprint instead of the diagnostic sample.",
    )
    parser.add_argument(
        "--sample-runs",
        type=int,
        default=1,
        help=(
            "Number of independent with-replacement diagnostic samples per city. "
            "Ignored with --all-buildings. Default: 1."
        ),
    )
    parser.add_argument(
        "--official-height-units",
        choices=["feet", "meters"],
        default="feet",
        help="Units for official footprint height fields. Default: feet.",
    )
    parser.add_argument(
        "--official-units-confirmed",
        action="store_true",
        help="Mark official units as confirmed by source metadata.",
    )
    parser.add_argument(
        "--skip-sha256",
        action="store_true",
        help="Skip checksum verification for faster diagnostic runs.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Optional reproducible diagnostic sample seed. Default: random each run.",
    )
    return parser.parse_args()


def import_dependencies() -> dict[str, Any]:
    """Import heavy dependencies after --help has had a chance to run."""
    try:
        import geopandas as gpd
        import laspy
        import numpy as np
        import pandas as pd
        from shapely import contains_xy, make_valid
        from shapely.geometry import box
        from shapely.ops import unary_union
    except ModuleNotFoundError as error:
        missing_name = error.name or "unknown package"
        print(f"ERROR: Missing Python dependency: {missing_name}")
        print("Install the task requirements from the repository root:")
        print(
            "  data_source/source/height_labels/venv_height_labels/bin/python "
            "-m pip install -r data_source/source/height_labels/requirements.txt"
        )
        sys.exit(1)

    return {
        "gpd": gpd,
        "laspy": laspy,
        "np": np,
        "pd": pd,
        "contains_xy": contains_xy,
        "make_valid": make_valid,
        "box": box,
        "unary_union": unary_union,
    }


def project_path(*parts: str) -> Path:
    """Build a path from the detected repository root."""
    return PROJECT_ROOT.joinpath(*parts)


def relative_project_path(path: Path) -> str:
    """Store portable repository-relative paths in outputs."""
    resolved = path.resolve()
    if not resolved.is_relative_to(PROJECT_ROOT):
        raise ValueError(f"Path is outside the project repository: {resolved}")
    return str(resolved.relative_to(PROJECT_ROOT))


def setup_logging() -> Path:
    """Create a dated log file for this run."""
    log_dir = project_path("data_source", "data", "height_labels", "generated")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = log_dir / f"derive_lidar_building_heights_{timestamp}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logging.info("Run started. Log path: %s", log_path)
    return log_path


def require_file(path: Path, description: str) -> None:
    """Fail loudly when a required input is missing."""
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    """Calculate a SHA-256 checksum for one local source file."""
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        while True:
            chunk = file_obj.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def safe_numeric(value: Any) -> float | None:
    """Convert source values to floats while preserving missing values."""
    try:
        if value is None:
            return None
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    return number


def official_height_to_meters(value: Any, units: str) -> float | None:
    """Convert an official footprint height value to meters."""
    number = safe_numeric(value)
    if number is None:
        return None
    if units == "feet":
        return number * 0.3048
    return number


def enforce_minimum_nonzero_height(height_m: float) -> float:
    """Set positive sub-minimum building heights to the minimum occupied height."""
    if math.isnan(height_m):
        return height_m
    if 0 < height_m < MIN_NONZERO_BUILDING_HEIGHT_M:
        return MIN_NONZERO_BUILDING_HEIGHT_M
    return height_m


def temporal_mismatch_reason(city_slug: str, building: Any) -> str:
    """Flag buildings whose footprint date conflicts with the LiDAR date."""
    if city_slug == "new_york_city":
        construction_year = safe_numeric(building.get("CONSTRUCTI"))
        if construction_year is not None and construction_year > 2014:
            return "constructed_after_lidar"
    if city_slug == "los_angeles":
        status = str(building.get("STATUS") or "").strip().lower()
        source_year = safe_numeric(building.get("DATE_"))
        if status in {"demolished", "removed"}:
            return "footprint_status_demolished_or_removed"
        if source_year is not None and source_year > 2024:
            return "footprint_date_after_lidar"
    return ""


@dataclass
class TilePoints:
    """Hold the point arrays needed from one LAZ tile."""

    x: Any
    y: Any
    z: Any
    classification: Any
    withheld: Any


def load_manifest(deps: dict[str, Any], cities: list[str]) -> Any:
    """Read the USGS tile manifest and keep verified local NYC/LA rows."""
    pd = deps["pd"]
    manifest_path = project_path(
        "data_source", "data", "height_labels", "generated", "usgs_3dep_tile_manifest.csv"
    )
    require_file(manifest_path, "USGS 3DEP tile manifest")
    manifest = pd.read_csv(manifest_path)

    required = {
        "city_slug",
        "project_directory",
        "filename",
        "local_path",
        "download_status",
        "expected_bytes",
        "sha256",
        "min_lon",
        "min_lat",
        "max_lon",
        "max_lat",
    }
    missing = sorted(required.difference(manifest.columns))
    if missing:
        raise ValueError(f"Tile manifest is missing required columns: {missing}")

    manifest = manifest[manifest["city_slug"].isin(cities)].copy()
    manifest = manifest[manifest["download_status"].isin(["existing", "downloaded"])].copy()
    if manifest.empty:
        raise ValueError("No downloaded manifest rows found for requested cities.")

    manifest["absolute_path"] = manifest["local_path"].map(lambda text: project_path(text))
    for path in manifest["absolute_path"]:
        require_file(path, "downloaded LAZ tile listed in manifest")
    return manifest


def validate_tile_files(deps: dict[str, Any], manifest: Any, skip_sha256: bool) -> Any:
    """Check local source files before using them."""
    rows = []
    for _, row in manifest.iterrows():
        path = Path(row["absolute_path"])
        expected_bytes = int(row["expected_bytes"])
        actual_bytes = path.stat().st_size
        if actual_bytes != expected_bytes:
            raise ValueError(
                f"Byte-size mismatch for {relative_project_path(path)}: "
                f"expected {expected_bytes}, found {actual_bytes}"
            )
        actual_sha256 = ""
        if not skip_sha256:
            actual_sha256 = sha256_file(path)
            if actual_sha256 != row["sha256"]:
                raise ValueError(
                    f"SHA-256 mismatch for {relative_project_path(path)}. "
                    "The raw source tile should not be used."
                )
        rows.append(
            {
                "city_slug": row["city_slug"],
                "project_directory": row["project_directory"],
                "filename": row["filename"],
                "local_path": row["local_path"],
                "expected_bytes": expected_bytes,
                "actual_bytes": actual_bytes,
                "sha256_manifest": row["sha256"],
                "sha256_checked": not skip_sha256,
                "sha256_actual": actual_sha256,
            }
        )
    return deps["pd"].DataFrame(rows)


def write_tile_inventory(deps: dict[str, Any], manifest: Any, validation_rows: Any) -> None:
    """Write a lightweight inventory from LAZ headers and manifest checks."""
    laspy = deps["laspy"]
    pd = deps["pd"]
    inventory_rows = []
    for _, row in manifest.iterrows():
        path = Path(row["absolute_path"])
        with laspy.open(path) as laz:
            header = laz.header
            crs = header.parse_crs()
            inventory_rows.append(
                {
                    "city_slug": row["city_slug"],
                    "project_directory": row["project_directory"],
                    "filename": row["filename"],
                    "local_path": row["local_path"],
                    "point_count": header.point_count,
                    "las_version": str(header.version),
                    "point_format": header.point_format.id,
                    "min_x": header.mins[0],
                    "min_y": header.mins[1],
                    "min_z": header.mins[2],
                    "max_x": header.maxs[0],
                    "max_y": header.maxs[1],
                    "max_z": header.maxs[2],
                    "crs": crs.to_string() if crs else "",
                }
            )
    inventory = pd.DataFrame(inventory_rows)
    inventory = inventory.merge(
        validation_rows,
        on=["city_slug", "project_directory", "filename", "local_path"],
        how="left",
    )
    output_path = project_path(
        "data_source", "data", "height_labels", "generated", "lidar_tile_inventory.csv"
    )
    inventory.to_csv(output_path, index=False)
    logging.info("Wrote tile inventory: %s", relative_project_path(output_path))


def load_city_footprints(deps: dict[str, Any], city_slug: str) -> Any:
    """Read, validate, and repair one city's clipped footprint file."""
    gpd = deps["gpd"]
    make_valid = deps["make_valid"]
    path = project_path(
        "data_source",
        "data",
        "building_footprints",
        "generated",
        city_slug,
        f"{city_slug}_building_footprints_5km.gpkg",
    )
    require_file(path, f"{city_slug} clipped building footprints")
    footprints = gpd.read_file(path)
    if footprints.empty:
        raise ValueError(f"No footprints found in {path}")
    if "building_footprint_id" not in footprints.columns:
        raise ValueError(f"Missing building_footprint_id column in {path}")

    official_field = CITY_CONFIG[city_slug]["official_height_field"]
    if official_field not in footprints.columns:
        raise ValueError(f"Missing official height field {official_field} in {path}")

    footprints = footprints.copy()
    footprints["geometry"] = footprints.geometry.map(make_valid)
    footprints = footprints[~footprints.geometry.is_empty & footprints.geometry.notna()].copy()
    if footprints.empty:
        raise ValueError(f"All footprints became empty after geometry repair: {path}")
    return footprints


def choose_buildings(deps: dict[str, Any], footprints: Any, city_slug: str, args: argparse.Namespace) -> Any:
    """Choose either every building or a reproducible diagnostic sample."""
    gpd = deps["gpd"]
    if args.all_buildings:
        selected = footprints.copy()
        selected["diagnostic_sample"] = False
        selected["sample_run"] = 1
        selected["sample_seed"] = args.random_seed
        selected["sample_draw_index"] = range(1, len(selected) + 1)
        selected["sample_occurrence_id"] = [
            f"{city_slug}_all_{position:08d}" for position in range(1, len(selected) + 1)
        ]
        logging.info("Selected all %s footprints for %s.", len(selected), city_slug)
        return gpd.GeoDataFrame(selected, geometry="geometry", crs=footprints.crs)

    np = deps["np"]
    pd = deps["pd"]
    sample_size = args.sample_size
    if sample_size <= 0:
        raise ValueError("--sample-size must be positive unless --all-buildings is used.")
    if args.sample_runs <= 0:
        raise ValueError("--sample-runs must be positive unless --all-buildings is used.")

    if args.random_seed is None:
        seed_sequence = np.random.SeedSequence()
        run_seeds = [int(seed) for seed in seed_sequence.generate_state(args.sample_runs)]
    else:
        run_seeds = [int(args.random_seed + sample_run - 1) for sample_run in range(1, args.sample_runs + 1)]

    sampled_frames = []
    for sample_run in range(1, args.sample_runs + 1):
        run_seed = run_seeds[sample_run - 1]
        run_sample = footprints.sample(
            n=sample_size,
            replace=True,
            random_state=int(run_seed),
        ).copy()
        run_sample["sample_run"] = sample_run
        run_sample["sample_seed"] = run_seed
        run_sample["sample_draw_index"] = range(1, sample_size + 1)
        run_sample["sample_occurrence_id"] = [
            f"{city_slug}_run{sample_run:02d}_draw{position:04d}"
            for position in range(1, sample_size + 1)
        ]
        sampled_frames.append(run_sample)

    selected = pd.concat(sampled_frames, ignore_index=True)
    selected["diagnostic_sample"] = True
    logging.info(
        "Selected %s with-replacement diagnostic draws for %s across %s run(s). "
        "Unique buildings drawn: %s.",
        len(selected),
        city_slug,
        args.sample_runs,
        selected["building_footprint_id"].nunique(),
    )
    return gpd.GeoDataFrame(selected, geometry="geometry", crs=footprints.crs)


def prepare_projected_geometries(deps: dict[str, Any], footprints: Any, selected: Any, city_slug: str) -> Any:
    """Project footprints and create roof/ground sampling geometries."""
    gpd = deps["gpd"]
    unary_union = deps["unary_union"]
    lidar_epsg = CITY_CONFIG[city_slug]["lidar_epsg"]

    projected_all = footprints.to_crs(epsg=lidar_epsg).copy()
    selected_unique = selected.drop_duplicates("building_footprint_id").copy()
    projected_selected = selected_unique.to_crs(epsg=lidar_epsg).copy()
    projected_selected["footprint_area_m2"] = projected_selected.geometry.area

    spatial_index = projected_all.sindex
    roof_geometries = []
    ring_geometries = []
    for position, geometry in enumerate(projected_selected.geometry):
        roof_geometry = geometry.buffer(-1.0)
        erosion_used_m = 1.0
        if roof_geometry.is_empty:
            roof_geometry = geometry.buffer(-0.5)
            erosion_used_m = 0.5
        if roof_geometry.is_empty:
            roof_geometry = geometry
            erosion_used_m = 0.0

        outer = geometry.buffer(5.0)
        inner = geometry.buffer(1.0)
        raw_ring = outer.difference(inner)

        neighbor_indices = list(spatial_index.query(outer, predicate="intersects"))
        this_id = projected_selected.iloc[position]["building_footprint_id"]
        neighbor_geometries = []
        for neighbor_index in neighbor_indices:
            neighbor_row = projected_all.iloc[neighbor_index]
            if neighbor_row["building_footprint_id"] == this_id:
                continue
            neighbor_geometries.append(neighbor_row.geometry)
        if neighbor_geometries:
            ring_geometry = raw_ring.difference(unary_union(neighbor_geometries))
        else:
            ring_geometry = raw_ring

        roof_geometries.append(roof_geometry)
        ring_geometries.append(ring_geometry)
        projected_selected.loc[projected_selected.index[position], "roof_erosion_m"] = erosion_used_m

    projected_selected["roof_geometry"] = roof_geometries
    projected_selected["ground_ring_geometry"] = ring_geometries
    return gpd.GeoDataFrame(projected_selected, geometry="geometry", crs=f"EPSG:{lidar_epsg}")


def load_tile_points(deps: dict[str, Any], path: Path) -> TilePoints:
    """Read only the point fields needed for this estimator."""
    laspy = deps["laspy"]
    np = deps["np"]
    las = laspy.read(path)
    withheld = getattr(las, "withheld", None)
    if withheld is None:
        withheld = np.zeros(len(las.x), dtype=bool)
    else:
        withheld = np.asarray(withheld, dtype=bool)
    return TilePoints(
        x=np.asarray(las.x),
        y=np.asarray(las.y),
        z=np.asarray(las.z),
        classification=np.asarray(las.classification),
        withheld=withheld,
    )


def append_points_inside_geometry(
    deps: dict[str, Any],
    tile_points: TilePoints,
    geometry: Any,
    class_mask: Any,
) -> Any:
    """Return elevations for tile points that fall inside one geometry."""
    np = deps["np"]
    contains_xy = deps["contains_xy"]
    if geometry.is_empty:
        return np.array([], dtype=float)
    min_x, min_y, max_x, max_y = geometry.bounds
    bbox_mask = (
        (tile_points.x >= min_x)
        & (tile_points.x <= max_x)
        & (tile_points.y >= min_y)
        & (tile_points.y <= max_y)
        & class_mask
        & (~tile_points.withheld)
    )
    if not bbox_mask.any():
        return np.array([], dtype=float)
    inside = contains_xy(geometry, tile_points.x[bbox_mask], tile_points.y[bbox_mask])
    if not inside.any():
        return np.array([], dtype=float)
    return tile_points.z[bbox_mask][inside]


def process_city(
    deps: dict[str, Any],
    city_slug: str,
    manifest: Any,
    args: argparse.Namespace,
) -> None:
    """Process one city's selected buildings and write diagnostics."""
    np = deps["np"]
    pd = deps["pd"]
    box = deps["box"]
    logging.info("Processing city: %s", city_slug)

    footprints = load_city_footprints(deps, city_slug)
    selected = choose_buildings(deps, footprints, city_slug, args)
    projected = prepare_projected_geometries(deps, footprints, selected, city_slug)

    records: dict[str, dict[str, Any]] = {}
    for _, building in projected.iterrows():
        building_id = building["building_footprint_id"]
        official_field = CITY_CONFIG[city_slug]["official_height_field"]
        official_height_m = official_height_to_meters(
            building.get(official_field),
            args.official_height_units,
        )
        records[building_id] = {
            "building_footprint_id": building_id,
            "city_slug": city_slug,
            "footprint_area_m2": building["footprint_area_m2"],
            "diagnostic_sample": bool(building["diagnostic_sample"]),
            "roof_erosion_m": building["roof_erosion_m"],
            "official_height_source": official_field,
            "official_height_raw": building.get(official_field),
            "official_height_units": args.official_height_units,
            "official_height_units_confirmed": bool(args.official_units_confirmed),
            "official_height_m": official_height_m,
            "official_ground_source": CITY_CONFIG[city_slug]["official_ground_field"],
            "official_ground_raw": building.get(CITY_CONFIG[city_slug]["official_ground_field"]),
            "source_date_raw": building.get(CITY_CONFIG[city_slug]["date_field"]),
            "lidar_collect_start": CITY_CONFIG[city_slug]["lidar_collect_start"],
            "lidar_collect_end": CITY_CONFIG[city_slug]["lidar_collect_end"],
            "temporal_mismatch_flag": temporal_mismatch_reason(city_slug, building),
            "tiles_used": set(),
            "roof_z": [],
            "ground_z": [],
        }

    city_tiles = manifest[manifest["city_slug"] == city_slug].copy()
    if city_tiles.empty:
        raise ValueError(f"No manifest tiles available for {city_slug}")

    selected_wgs84 = selected.to_crs(epsg=4326)
    for _, tile_row in city_tiles.iterrows():
        tile_bbox = box(
            tile_row["min_lon"],
            tile_row["min_lat"],
            tile_row["max_lon"],
            tile_row["max_lat"],
        )
        candidate_ids = selected_wgs84[selected_wgs84.intersects(tile_bbox)]["building_footprint_id"]
        if candidate_ids.empty:
            continue

        path = Path(tile_row["absolute_path"])
        logging.info(
            "Reading tile %s for %s candidate buildings.",
            relative_project_path(path),
            len(candidate_ids),
        )
        tile_points = load_tile_points(deps, path)
        classes = tile_points.classification
        valid_roof_mask = np.isin(classes, list(ROOF_CLASSES)) & ~np.isin(
            classes, list(EXCLUDED_CLASSES)
        )
        ground_mask = np.isin(classes, list(GROUND_CLASSES))

        candidate_buildings = projected[
            projected["building_footprint_id"].isin(candidate_ids)
        ]
        for _, building in candidate_buildings.iterrows():
            building_id = building["building_footprint_id"]
            roof_z = append_points_inside_geometry(
                deps,
                tile_points,
                building["roof_geometry"],
                valid_roof_mask,
            )
            ground_z = append_points_inside_geometry(
                deps,
                tile_points,
                building["ground_ring_geometry"],
                ground_mask,
            )
            if len(roof_z) or len(ground_z):
                records[building_id]["tiles_used"].add(tile_row["filename"])
            if len(roof_z):
                records[building_id]["roof_z"].append(roof_z)
            if len(ground_z):
                records[building_id]["ground_z"].append(ground_z)

    output_rows = []
    for building_id, record in records.items():
        roof_arrays = record.pop("roof_z")
        ground_arrays = record.pop("ground_z")
        roof_z = np.concatenate(roof_arrays) if roof_arrays else np.array([], dtype=float)
        ground_z = np.concatenate(ground_arrays) if ground_arrays else np.array([], dtype=float)
        tiles_used = sorted(record.pop("tiles_used"))

        ground_elevation_m = float(np.median(ground_z)) if len(ground_z) else math.nan
        heights = {}
        if len(roof_z) and len(ground_z):
            for percentile in [50, 75, 90, 95]:
                heights[f"height_p{percentile}_m"] = enforce_minimum_nonzero_height(
                    float(np.percentile(roof_z, percentile) - ground_elevation_m)
                )
            heights["height_max_clean_m"] = enforce_minimum_nonzero_height(
                float(np.percentile(roof_z, 99) - ground_elevation_m)
            )
            heights["height_max_m"] = enforce_minimum_nonzero_height(
                float(np.max(roof_z) - ground_elevation_m)
            )
            heights["height_m"] = heights["height_p90_m"]
            heights["height_label_m"] = heights["height_p90_m"]
        else:
            for percentile in [50, 75, 90, 95]:
                heights[f"height_p{percentile}_m"] = math.nan
            heights["height_max_clean_m"] = math.nan
            heights["height_max_m"] = math.nan
            heights["height_m"] = math.nan
            heights["height_label_m"] = math.nan

        roof_pixel_count = int(len(roof_z))
        ground_point_count = int(len(ground_z))
        roof_coverage_fraction = min(1.0, roof_pixel_count / 50.0)
        ground_ring_coverage = min(1.0, ground_point_count / 50.0)
        height_value = heights["height_m"]

        reject_reasons = []
        if roof_pixel_count < 10:
            reject_reasons.append("fewer_than_10_roof_points")
        if ground_point_count < 10:
            reject_reasons.append("fewer_than_10_ground_points")
        if math.isnan(height_value):
            reject_reasons.append("missing_height")
        elif height_value <= 0:
            reject_reasons.append("implausible_height")
        if record["temporal_mismatch_flag"]:
            reject_reasons.append(record["temporal_mismatch_flag"])

        if reject_reasons:
            quality_tier = "Reject"
        elif roof_pixel_count >= 100 and ground_point_count >= 100:
            quality_tier = "A"
        elif roof_pixel_count >= 40 and ground_point_count >= 40:
            quality_tier = "B"
        else:
            quality_tier = "C"

        official_height_m = record["official_height_m"]
        residual_m = (
            height_value - official_height_m
            if official_height_m is not None and not math.isnan(height_value)
            else math.nan
        )

        output_rows.append(
            {
                **record,
                **heights,
                "height_definition": HEIGHT_DEFINITION,
                "ground_elevation_m": ground_elevation_m,
                "local_ground_m": ground_elevation_m,
                "primary_label_column": "height_label_m",
                "primary_label_source_column": "height_p90_m",
                "roof_point_count": roof_pixel_count,
                "roof_pixel_count": roof_pixel_count,
                "ground_point_count": ground_point_count,
                "roof_coverage_fraction": roof_coverage_fraction,
                "ground_ring_coverage": ground_ring_coverage,
                "quality_tier": quality_tier,
                "usable_for_training": quality_tier in {"A", "B"},
                "usable_for_validation": quality_tier in {"A", "B", "C"},
                "reject_reason": ";".join(reject_reasons),
                "tiles_used": "|".join(tiles_used),
                "official_minus_lidar_m": (
                    official_height_m - height_value
                    if official_height_m is not None and not math.isnan(height_value)
                    else math.nan
                ),
                "lidar_minus_official_m": residual_m,
            }
        )

    building_level_output = pd.DataFrame(output_rows)
    sample_columns = [
        "building_footprint_id",
        "city_slug",
        "sample_run",
        "sample_seed",
        "sample_draw_index",
        "sample_occurrence_id",
        "diagnostic_sample",
    ]
    sample_draws = selected[sample_columns].copy()
    output = sample_draws.merge(
        building_level_output.drop(columns=["diagnostic_sample"], errors="ignore"),
        on=["building_footprint_id", "city_slug"],
        how="left",
        validate="many_to_one",
    )
    output_dir = project_path("data_source", "data", "height_labels", "generated", city_slug)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_stem = "building_height_labels_all" if args.all_buildings else "building_height_diagnostics_sample"
    temp_dir = output_dir / ("temp_all_buildings" if args.all_buildings else "temp_samples")
    temp_dir.mkdir(parents=True, exist_ok=True)

    run_paths = []
    for sample_run, run_output in output.groupby("sample_run", sort=True):
        run_path = temp_dir / f"{output_stem}_run_{int(sample_run):02d}.csv"
        run_output = run_output.sort_values("sample_draw_index").copy()
        run_output.to_csv(run_path, index=False)
        run_paths.append(run_path)
        logging.info("Wrote temporary sample run: %s", relative_project_path(run_path))

    output = pd.concat([pd.read_csv(path) for path in run_paths], ignore_index=True)
    diagnostics_path = output_dir / f"{output_stem}.csv"
    output.to_csv(diagnostics_path, index=False)
    logging.info(
        "Wrote merged diagnostics from %s sample run file(s): %s",
        len(run_paths),
        relative_project_path(diagnostics_path),
    )

    suffix = "_all" if args.all_buildings else ""
    write_comparison(deps, output, output_dir, suffix=suffix)
    write_quality_summary(output, output_dir, suffix=suffix)
    write_geopackage(deps, selected, output, output_dir, suffix=suffix)


def write_comparison(deps: dict[str, Any], output: Any, output_dir: Path, suffix: str = "") -> None:
    """Compare candidate LiDAR height definitions against official fields."""
    np = deps["np"]
    pd = deps["pd"]
    rows = []
    valid_official = output["official_height_m"].notna()
    for column in [
        "height_p50_m",
        "height_p75_m",
        "height_p90_m",
        "height_p95_m",
        "height_max_clean_m",
        "height_max_m",
    ]:
        valid = valid_official & output[column].notna()
        if valid.sum() == 0:
            rows.append(
                {
                    "height_definition_column": column,
                    "n": 0,
                    "mae_m": math.nan,
                    "rmse_m": math.nan,
                    "bias_lidar_minus_official_m": math.nan,
                    "median_absolute_error_m": math.nan,
                    "r_squared": math.nan,
                }
            )
            continue
        errors = output.loc[valid, column] - output.loc[valid, "official_height_m"]
        official = output.loc[valid, "official_height_m"]
        ss_res = float(np.sum(errors**2))
        ss_tot = float(np.sum((official - official.mean()) ** 2))
        rows.append(
            {
                "height_definition_column": column,
                "n": int(valid.sum()),
                "mae_m": float(np.mean(np.abs(errors))),
                "rmse_m": float(np.sqrt(np.mean(errors**2))),
                "bias_lidar_minus_official_m": float(np.mean(errors)),
                "median_absolute_error_m": float(np.median(np.abs(errors))),
                "r_squared": 1 - ss_res / ss_tot if ss_tot > 0 else math.nan,
            }
        )
    path = output_dir / f"height_definition_comparison{suffix}.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    logging.info("Wrote comparison: %s", relative_project_path(path))


def write_quality_summary(output: Any, output_dir: Path, suffix: str = "") -> None:
    """Summarize confidence tiers and training usability."""
    summary = (
        output.groupby(["quality_tier", "usable_for_training", "usable_for_validation"], dropna=False)
        .size()
        .reset_index(name="building_count")
        .sort_values(["quality_tier", "usable_for_training", "usable_for_validation"])
    )
    path = output_dir / f"quality_tier_summary{suffix}.csv"
    summary.to_csv(path, index=False)
    logging.info("Wrote quality summary: %s", relative_project_path(path))


def write_geopackage(deps: dict[str, Any], selected: Any, output: Any, output_dir: Path, suffix: str = "") -> None:
    """Write the diagnostic labels back to footprint geometries."""
    keys = ["building_footprint_id", "city_slug"]
    if "sample_occurrence_id" in selected.columns and "sample_occurrence_id" in output.columns:
        keys.append("sample_occurrence_id")
    output_columns = keys + [
        column for column in output.columns if column not in selected.columns and column not in keys
    ]
    merged = selected.merge(output[output_columns], on=keys, how="left")
    path = output_dir / f"lidar_building_heights{suffix}.gpkg"
    merged.to_file(path, driver="GPKG", layer="lidar_building_heights")
    logging.info("Wrote GeoPackage: %s", relative_project_path(path))


def main() -> None:
    """Run the height-label diagnostic pipeline."""
    relaunch_inside_venv()
    args = parse_args()
    log_path = setup_logging()
    deps = import_dependencies()

    cities = args.city or sorted(CITY_CONFIG)
    if args.all_buildings:
        logging.warning(
            "Full-city processing requested. This can be slow because every "
            "selected building is tested against point-cloud tiles."
        )
    if not args.official_units_confirmed:
        logging.warning(
            "Official footprint height units are not marked confirmed. "
            "Outputs will record official_height_units_confirmed=False."
        )

    try:
        manifest = load_manifest(deps, cities)
        validation_rows = validate_tile_files(deps, manifest, args.skip_sha256)
        write_tile_inventory(deps, manifest, validation_rows)
        for city_slug in cities:
            process_city(deps, city_slug, manifest, args)
        logging.info("Run completed cleanly.")
        logging.info("Final log path: %s", log_path)
    except Exception:
        logging.exception("Run failed. Outputs from this run must be treated as partial.")
        raise


if __name__ == "__main__":
    main()
