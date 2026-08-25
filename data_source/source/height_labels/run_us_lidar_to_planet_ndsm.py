"""Download USGS LiDAR, build Planet-aligned nDSMs, and optionally clean LAZ files.

This is the US-only, resumable city orchestrator for the 54 training cities
whose official source is USGS 3DEP. It deliberately processes one city at a
time so a failure cannot corrupt another city's manifest or delete its data.

Modes
-----
``--dry-run``
    Query small USGS project metadata responses, choose the latest acquisition
    set, validate local Planet/footprint inputs, and write the city plan. No
    LiDAR payload is downloaded and nothing is deleted.

``--confirm-download``
    Download only the planned AOI-intersecting LAZ files and build one validated
    three-band nDSM. Add ``--confirm-delete-lidar`` to delete only the manifest-
    listed LAZ files after the nDSM passes every validation.

Three-band output
-----------------
1. Continuous nDSM in metres.
2. Building-footprint-masked nDSM in metres.
3. QA code: 0=invalid/uncovered, 1=valid outside buildings, 2=valid in building.
"""

from __future__ import annotations

import argparse
import atexit
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time
from typing import Any
import urllib.error
import urllib.parse
import urllib.request


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
VENV_DIR = SCRIPT_DIR / "venv_height_labels"
VENV_PYTHON = VENV_DIR / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
VENV_MARKER = "HEIGHT_LABELS_VENV_ACTIVE"


def relaunch_inside_venv() -> None:
    """Restart inside the project environment and wait correctly on Windows."""
    if os.environ.get(VENV_MARKER) == "1" or Path(sys.prefix).resolve() == VENV_DIR.resolve():
        return
    if not VENV_PYTHON.exists():
        raise SystemExit(f"Missing height-label environment: {VENV_PYTHON}")
    environment = os.environ.copy()
    environment[VENV_MARKER] = "1"
    if os.name == "nt":
        # Windows does not provide Unix-style process replacement reliably for
        # CMD loops. Wait for the child to finish so two manifest writers cannot
        # run at the same time.
        import subprocess

        completed = subprocess.run(
            [str(VENV_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]],
            env=environment,
            check=False,
        )
        raise SystemExit(completed.returncode)
    os.execve(
        str(VENV_PYTHON),
        [str(VENV_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]],
        environment,
    )


relaunch_inside_venv()

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from pyproj import CRS, Transformer
from shapely.geometry import box, shape
from shapely.ops import transform as shapely_transform, unary_union


INVENTORY_PATH = PROJECT_ROOT / (
    "data_source/data/height_labels/generated/training_open_lidar/"
    "training_cities_with_open_lidar.csv"
)
TILE_METADATA_PATH = (
    PROJECT_ROOT / "data_source/data/height_labels/generated/usgs_3dep_global_city_tile_metadata.csv"
)
SELECTED_SCENES_PATH = PROJECT_ROOT / (
    "data_source/data/planet_imagery/generated/training_lidar_year_scene_selection/"
    "selected_training_lidar_city_planet_scenes.csv"
)
PLANET_ROOT = PROJECT_ROOT / "data_source/data/planet_imagery/source/training_lidar_94"
FOOTPRINT_ROOT = PROJECT_ROOT / "data_source/data/building_footprints/generated"
LIDAR_SOURCE_ROOT = PROJECT_ROOT / "data_source/data/height_labels/source/us_training_54"
OUTPUT_ROOT = PROJECT_ROOT / "data_source/data/height_labels/generated/us_training_planet_ndsm"
RUN_MANIFEST = OUTPUT_ROOT / "us_lidar_to_planet_ndsm_manifest.csv"
ELEVATION_INDEX_URL = (
    "https://index.nationalmap.gov/arcgis/rest/services/"
    "3DEPElevationIndex/MapServer/8/query"
)
USER_AGENT = "building-height-prediction/1.0 US-training-nDSM"
MINIMUM_COVERAGE_PERCENT = 99.0
DOWNLOAD_CHUNK_BYTES = 8 * 1024 * 1024
NODATA_VALUE = -9999.0

MANIFEST_COLUMNS = [
    "city_slug", "city_name", "status", "selected_project_directories",
    "lidar_collect_start", "lidar_collect_end", "planet_scene_count",
    "lidar_date_source", "project_selection_basis",
    "planet_scenes_on_or_after_lidar", "temporal_fallback_used",
    "planned_tile_count", "planned_coverage_percent", "footprint_path",
    "planet_template_path", "planet_majority_scene_count", "planet_outlier_scene_count",
    "lidar_tile_manifest_path", "downloaded_tile_count", "downloaded_bytes",
    "classification_audit_path",
    "output_ndsm_path", "output_valid_pixels", "output_building_pixels",
    "lidar_deleted", "last_checked_utc", "error_message",
]


class StreamTee:
    """Mirror terminal output and tracebacks into one honest run log."""

    def __init__(self, terminal: Any, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.terminal = terminal
        self.file = path.open("w", encoding="utf-8")

    def write(self, message: str) -> int:
        self.terminal.write(message)
        self.file.write(message)
        self.file.flush()
        return len(message)

    def flush(self) -> None:
        self.terminal.flush()
        self.file.flush()

    def close(self) -> None:
        if not self.file.closed:
            self.file.close()


def start_log() -> Path:
    """Start a dated log before validation so failures cannot disappear."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = OUTPUT_ROOT / "logs" / f"run_us_lidar_to_planet_ndsm_{timestamp}.log"
    tee = StreamTee(sys.__stdout__, path)
    sys.stdout = tee
    sys.stderr = tee

    def restore() -> None:
        if sys.stdout is tee:
            sys.stdout = tee.terminal
        if sys.stderr is tee:
            sys.stderr = sys.__stderr__
        tee.close()

    atexit.register(restore)
    return path


def parse_args() -> argparse.Namespace:
    """Read explicit modes and bounded, resumable city options."""
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--confirm-download", action="store_true")
    parser.add_argument(
        "--confirm-delete-lidar",
        action="store_true",
        help="Delete manifest-listed LAZ files only after validated nDSM success.",
    )
    parser.add_argument("--city-slug", action="append", dest="city_slugs")
    parser.add_argument("--city-offset", type=int, default=0)
    parser.add_argument("--city-limit", type=int, default=1)
    parser.add_argument("--minimum-free-gb", type=float, default=100.0)
    parser.add_argument("--ground-class", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def relative(path: Path) -> str:
    """Return a portable repository-relative path."""
    resolved = path.resolve()
    if not resolved.is_relative_to(PROJECT_ROOT):
        raise ValueError(f"Path is outside repository: {resolved}")
    return resolved.relative_to(PROJECT_ROOT).as_posix()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    """Checkpoint one complete CSV without exposing a partially written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, quoting=csv.QUOTE_MINIMAL)
    temporary.replace(path)


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the 54-city inventory, detailed USGS tiles, and selected scenes."""
    for path in [INVENTORY_PATH, TILE_METADATA_PATH, SELECTED_SCENES_PATH]:
        if not path.is_file():
            raise FileNotFoundError(f"Missing required input: {path}")
    inventory = pd.read_csv(INVENTORY_PATH, dtype={"city_slug": str}).fillna("")
    inventory = inventory[
        (inventory["country"] == "United States of America")
        & (inventory["lidar_source_program"] == "USGS 3D Elevation Program (3DEP)")
    ].copy()
    if len(inventory) != 54 or inventory["city_slug"].duplicated().any():
        raise RuntimeError(f"Expected 54 unique USGS training cities; found {len(inventory)}")
    tiles = pd.read_csv(TILE_METADATA_PATH, dtype={"city_slug": str}).fillna("")
    tiles = tiles[tiles["city_slug"].isin(inventory["city_slug"])].copy()
    scenes = pd.read_csv(SELECTED_SCENES_PATH, dtype={"city_slug": str, "scene_id": str})
    scenes = scenes[scenes["city_slug"].isin(inventory["city_slug"])].copy()
    if scenes.groupby("city_slug")["scene_id"].nunique().ne(8).any():
        raise RuntimeError("Every US city must have exactly eight unique selected Planet scenes")
    return inventory.sort_values("randomized_city_rank"), tiles, scenes


def load_aoi(path_value: str) -> Any:
    path = PROJECT_ROOT / path_value
    document = json.loads(path.read_text(encoding="utf-8"))
    features = document.get("features") or []
    if len(features) != 1:
        raise ValueError(f"Expected one AOI feature: {path}")
    geometry = shape(features[0]["geometry"])
    if geometry.is_empty or not geometry.is_valid:
        raise ValueError(f"Invalid AOI geometry: {path}")
    return geometry


def local_projection(geometry: Any) -> tuple[Any, Transformer]:
    centroid = geometry.centroid
    zone = int((centroid.x + 180) // 6) + 1
    epsg = 32600 + zone if centroid.y >= 0 else 32700 + zone
    transformer = Transformer.from_crs(4326, CRS.from_epsg(epsg), always_xy=True)
    return shapely_transform(transformer.transform, geometry), transformer


def request_json(url: str, params: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{url}?{urllib.parse.urlencode(params)}", headers={"User-Agent": USER_AGENT}
    )
    for attempt in range(1, 5):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.load(response)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            if attempt == 4:
                raise
            time.sleep(3 * attempt)
    raise RuntimeError("USGS metadata retry loop ended unexpectedly")


def epoch_date(value: Any) -> str:
    if value in (None, ""):
        return ""
    return datetime.fromtimestamp(float(value) / 1000, timezone.utc).date().isoformat()


def query_project_dates() -> dict[str, dict[str, str]]:
    """Map every USGS staged project directory to official collection dates.

    The USGS layer currently rejects envelope-filtered requests even though its
    non-spatial query works. Read the small metadata table in deterministic
    pages once per run, then join its project directories to the already
    AOI-filtered tile inventory locally. This never requests LiDAR payloads.
    """
    grouped: dict[str, dict[str, str]] = {}
    offset = 0
    page_size = 1000
    while True:
        payload = request_json(
            ELEVATION_INDEX_URL,
            {
                "f": "json", "where": "1=1",
                "outFields": "OBJECTID,collect_start,collect_end,lpc_link",
                "returnGeometry": "false", "orderByFields": "OBJECTID ASC",
                "resultOffset": offset, "resultRecordCount": page_size,
            },
        )
        if payload.get("error"):
            raise RuntimeError(f"USGS elevation-index error: {payload['error']}")
        features = payload.get("features") or []
        for feature in features:
            values = feature.get("attributes") or {}
            match = re.search(r"/Projects/([^/]+)", str(values.get("lpc_link", "")))
            if not match:
                continue
            project = match.group(1)
            start = epoch_date(values.get("collect_start"))
            end = epoch_date(values.get("collect_end"))
            current = grouped.setdefault(project, {"start": start, "end": end})
            if start and (not current["start"] or start < current["start"]):
                current["start"] = start
            if end and (not current["end"] or end > current["end"]):
                current["end"] = end
        offset += len(features)
        if not payload.get("exceededTransferLimit") and len(features) < page_size:
            break
        if not features:
            raise RuntimeError("USGS metadata pagination stopped before completion")
    if not grouped:
        raise RuntimeError("USGS elevation index returned no project collection dates")
    return grouped


def coverage_percent(aoi: Any, rows: pd.DataFrame) -> float:
    """Calculate exact AOI coverage from advertised tile rectangles."""
    aoi_projected, transformer = local_projection(aoi)
    footprints = [
        box(float(row.min_lon), float(row.min_lat), float(row.max_lon), float(row.max_lat))
        for row in rows.itertuples()
    ]
    if not footprints:
        return 0.0
    covered = aoi.intersection(unary_union(footprints))
    projected = shapely_transform(transformer.transform, covered)
    return min(100.0, 100.0 * projected.area / aoi_projected.area)


def select_latest_acquisition(
    aoi: Any,
    city_tiles: pd.DataFrame,
    dates: dict[str, dict[str, str]],
    inventory_start: str,
    inventory_end: str,
) -> tuple[pd.DataFrame, list[str], str, str, float]:
    """Choose newest complete project, or newest greedy project set at borders.

    Official collection dates are preferred. If the USGS collection-index
    attribute query is unavailable, order the already verified project groups
    by their official product publication dates and retain the inventory's
    previously verified collection interval. The caller records this fallback
    explicitly; it is never presented as an exact project/date join.
    """
    candidates = []
    for project, rows in city_tiles.groupby("project_directory"):
        if not project:
            continue
        official_end = dates.get(project, {}).get("end", "")
        publication_end = str(rows["publication_date"].max())
        recency = official_end or publication_end
        if not recency:
            continue
        candidates.append((recency, project, coverage_percent(aoi, rows)))
    if not candidates:
        raise RuntimeError("No tile project has defensible official collection dates")
    candidates.sort(reverse=True)
    complete = [item for item in candidates if item[2] >= MINIMUM_COVERAGE_PERCENT]
    if complete:
        _, project, coverage = complete[0]
        projects = [project]
    else:
        projects = []
        coverage = 0.0
        for _, project, _ in candidates:
            projects.append(project)
            chosen = city_tiles[city_tiles["project_directory"].isin(projects)]
            coverage = coverage_percent(aoi, chosen)
            if coverage >= MINIMUM_COVERAGE_PERCENT:
                break
    if coverage < MINIMUM_COVERAGE_PERCENT:
        raise RuntimeError(f"Latest dated project set covers only {coverage:.6f}% of AOI")
    selected = city_tiles[city_tiles["project_directory"].isin(projects)].copy()
    exact_starts = [dates[project]["start"] for project in projects if dates.get(project, {}).get("start")]
    exact_ends = [dates[project]["end"] for project in projects if dates.get(project, {}).get("end")]
    if len(exact_ends) == len(projects):
        start = min(exact_starts or exact_ends)
        end = max(exact_ends)
    else:
        if not inventory_end:
            raise RuntimeError("Inventory fallback lacks a verified LiDAR acquisition end date")
        # The inventory start may belong to an older project in the same AOI.
        # Preserve only its latest verified date for the fallback rather than
        # falsely attaching a multi-project historical interval to this project.
        start, end = inventory_end, inventory_end
    return selected, projects, start, end, coverage


def find_footprints(city_slug: str) -> Path:
    city_dir = FOOTPRINT_ROOT / city_slug
    preferred = [
        city_dir / f"{city_slug}_building_footprints_merged_5km.gpkg",
        city_dir / f"{city_slug}_building_footprints_5km.gpkg",
    ]
    matches = [path for path in preferred if path.is_file()]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one preferred footprint GeoPackage for {city_slug}; found {matches}"
        )
    return matches[0]


def grid_signature(path: Path) -> tuple[Any, ...]:
    with rasterio.open(path) as src:
        if src.crs is None:
            raise RuntimeError(f"Planet raster has no CRS: {path}")
        return (
            src.crs.to_wkt(), tuple(round(v, 9) for v in tuple(src.transform)),
            src.width, src.height, tuple(round(v, 6) for v in tuple(src.bounds)),
        )


def choose_planet_grid(city_slug: str) -> tuple[Path, list[Path], list[Path]]:
    city_dir = PLANET_ROOT / city_slug
    scenes = sorted(city_dir.rglob("*_3B_AnalyticMS_SR*_clip.tif")) if city_dir.is_dir() else []
    if len(scenes) != 8:
        raise RuntimeError(f"Expected eight downloaded analytic scenes for {city_slug}; found {len(scenes)}")
    signatures = {path: grid_signature(path) for path in scenes}
    counts: dict[tuple[Any, ...], int] = {}
    for signature in signatures.values():
        counts[signature] = counts.get(signature, 0) + 1
    majority = max(counts, key=counts.get)
    compatible = [path for path in scenes if signatures[path] == majority]
    outliers = [path for path in scenes if signatures[path] != majority]
    if len(compatible) <= len(scenes) / 2:
        raise RuntimeError(f"No strict majority Planet grid for {city_slug}")
    return compatible[0], compatible, outliers


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(DOWNLOAD_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_tile_manifest(city_slug: str, rows: pd.DataFrame) -> Path:
    """Write the exact per-city allow-list used for download and deletion."""
    path = OUTPUT_ROOT / city_slug / f"{city_slug}_usgs_lidar_tile_manifest.csv"
    records = []
    for row in rows.itertuples():
        filename = Path(urllib.parse.urlparse(str(row.download_url)).path).name
        local_path = LIDAR_SOURCE_ROOT / city_slug / str(row.project_directory) / filename
        exists = local_path.is_file()
        records.append(
            {
                "city_slug": city_slug,
                "project_directory": row.project_directory,
                "download_url": row.download_url,
                "local_path": relative(local_path),
                "download_status": "downloaded_verified" if exists else "planned",
                "downloaded_bytes": local_path.stat().st_size if exists else "",
                "sha256": sha256_file(local_path) if exists else "",
            }
        )
    atomic_write_csv(pd.DataFrame(records), path)
    return path


def download_tiles(city_slug: str, rows: pd.DataFrame, minimum_free_gb: float) -> tuple[list[Path], int]:
    """Download planned files sequentially and atomically for one city."""
    paths, total_bytes = [], 0
    for row in rows.itertuples():
        filename = Path(urllib.parse.urlparse(str(row.download_url)).path).name
        if not filename.lower().endswith((".las", ".laz")):
            raise RuntimeError(f"Unexpected USGS point-cloud filename: {filename}")
        path = LIDAR_SOURCE_ROOT / city_slug / str(row.project_directory) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_file() and path.stat().st_size > 0:
            paths.append(path)
            total_bytes += path.stat().st_size
            continue
        free_gb = shutil.disk_usage(path.parent).free / 1e9
        if free_gb < minimum_free_gb:
            raise OSError(f"Free disk {free_gb:.2f} GB is below {minimum_free_gb:.2f} GB")
        partial = path.with_suffix(path.suffix + ".partial")
        request = urllib.request.Request(str(row.download_url), headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=300) as response, partial.open("wb") as file:
                while True:
                    chunk = response.read(DOWNLOAD_CHUNK_BYTES)
                    if not chunk:
                        break
                    file.write(chunk)
            if partial.stat().st_size == 0:
                raise IOError(f"Empty USGS download: {row.download_url}")
            partial.replace(path)
        except Exception:
            if partial.exists():
                partial.unlink()
            raise
        paths.append(path)
        total_bytes += path.stat().st_size
    return paths, total_bytes


def load_builder_module() -> Any:
    path = SCRIPT_DIR / "build_lidar_ndsm_raster.py"
    spec = importlib.util.spec_from_file_location("build_lidar_ndsm_raster", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load nDSM builder: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_three_band_ndsm(
    city_slug: str,
    lidar_paths: list[Path],
    footprint_path: Path,
    template_path: Path,
    compatible_scenes: list[Path],
    ground_class: int,
    projects: list[str],
    overwrite: bool,
) -> tuple[Path, Path, int, int]:
    """Use tested builder functions and retain only the requested three bands."""
    builder = load_builder_module()
    city_dir = OUTPUT_ROOT / city_slug
    city_dir.mkdir(parents=True, exist_ok=True)
    output = city_dir / f"{city_slug}_usgs_lidar_planet_aligned_ndsm.tif"
    class_audit = city_dir / f"{city_slug}_lidar_classification_audit.csv"
    if output.exists() and not overwrite:
        for scene in compatible_scenes:
            builder.verify_output(scene, output)
        with rasterio.open(output) as src:
            if src.count != 3:
                raise RuntimeError(f"Existing output is not three-band: {output}")
            if not class_audit.is_file():
                raise RuntimeError(
                    "Existing nDSM lacks its classification audit; pass --overwrite "
                    "to rebuild before allowing cleanup"
                )
            qa = src.read(3)
            return output, class_audit, int((qa > 0).sum()), int((qa == 2).sum())

    builder.audit_point_classifications(lidar_paths, ground_class, class_audit)
    with rasterio.open(template_path) as template:
        dsm, dtm_observed, _, _ = builder.accumulate_lidar_surfaces(
            lidar_paths, template.crs, template.transform,
            template.width, template.height, ground_class,
        )
    dtm = builder.fill_dtm(dtm_observed, 250.0)
    ndsm = np.where(np.isfinite(dsm - dtm), np.maximum(dsm - dtm, 0), np.nan).astype(np.float32)
    building_mask = builder.rasterize_building_mask(template_path, footprint_path, False)
    building_ndsm, _ = builder.enforce_minimum_building_agl(ndsm, building_mask, 2.4)
    valid = np.isfinite(ndsm)
    qa = np.zeros(ndsm.shape, dtype=np.float32)
    qa[valid] = 1
    qa[valid & (building_mask > 0)] = 2
    builder.write_multiband_output(
        template_path,
        output,
        [("continuous_ndsm_m", ndsm), ("building_only_ndsm_m", building_ndsm), ("qa_code", qa)],
        {
            "city_slug": city_slug,
            "usgs_projects": ";".join(projects),
            "qa_codes": "0=invalid;1=valid_nonbuilding;2=valid_building",
            "created_utc": utc_now(),
        },
    )
    for scene in compatible_scenes:
        builder.verify_output(scene, output)
    with rasterio.open(output) as src:
        if src.count != 3 or src.descriptions != (
            "continuous_ndsm_m", "building_only_ndsm_m", "qa_code"
        ):
            raise RuntimeError(f"Three-band output validation failed: {output}")
        qa_check = src.read(3)
        unexpected = set(np.unique(qa_check).tolist()) - {0.0, 1.0, 2.0}
        if unexpected:
            raise RuntimeError(f"Unexpected QA codes {unexpected}: {output}")
    return output, class_audit, int((qa > 0).sum()), int((qa == 2).sum())


def delete_manifest_lidar(tile_manifest: Path, output: Path, city_slug: str) -> None:
    """Delete only verified input point clouds after validating the final output."""
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError("Refusing LiDAR deletion because final nDSM is missing or empty")
    frame = pd.read_csv(tile_manifest, dtype=str).fillna("")
    if frame.empty or set(frame["download_status"]) != {"downloaded_verified"}:
        raise RuntimeError("Refusing deletion because tile manifest is not fully verified")
    paths = [(PROJECT_ROOT / value).resolve() for value in frame["local_path"]]
    allowed_root = (LIDAR_SOURCE_ROOT / city_slug).resolve()
    if any(not path.is_relative_to(allowed_root) for path in paths):
        raise RuntimeError("Tile manifest contains a path outside the city LiDAR source folder")
    for path in paths:
        if path.is_file():
            path.unlink()
    remaining = [path for path in paths if path.exists()]
    if remaining:
        raise RuntimeError(f"Failed to delete {len(remaining)} manifest-listed LiDAR files")


def existing_manifest() -> pd.DataFrame:
    if not RUN_MANIFEST.is_file():
        return pd.DataFrame(columns=MANIFEST_COLUMNS)
    frame = pd.read_csv(RUN_MANIFEST, dtype=str).fillna("")
    missing = set(MANIFEST_COLUMNS) - set(frame.columns)
    if missing:
        raise RuntimeError(f"Existing run manifest lacks columns: {sorted(missing)}")
    return frame[MANIFEST_COLUMNS]


def checkpoint(records: dict[str, dict[str, Any]]) -> None:
    frame = pd.DataFrame(records.values(), columns=MANIFEST_COLUMNS).sort_values("city_slug")
    atomic_write_csv(frame, RUN_MANIFEST)


def main() -> None:
    args = parse_args()
    log_path = start_log()
    print(f"Run log: {relative(log_path)}", flush=True)
    if args.confirm_delete_lidar and not args.confirm_download:
        raise ValueError("--confirm-delete-lidar requires --confirm-download")
    if args.city_offset < 0 or args.city_limit < 0 or args.minimum_free_gb < 0:
        raise ValueError("Offsets, limits, and storage threshold must be nonnegative")
    inventory, tiles, scenes = load_inputs()
    if args.city_slugs:
        unknown = sorted(set(args.city_slugs) - set(inventory["city_slug"]))
        if unknown:
            raise ValueError(f"Unknown US training city slugs: {unknown}")
        inventory = inventory[inventory["city_slug"].isin(args.city_slugs)]
    inventory = inventory.iloc[args.city_offset:]
    if args.city_limit:
        inventory = inventory.iloc[: args.city_limit]

    prior = existing_manifest()
    records = {row["city_slug"]: row.to_dict() for _, row in prior.iterrows()}
    failures = 0
    print(f"US cities to process: {len(inventory)}", flush=True)
    print("Reading official USGS project dates (metadata only; no LiDAR download)", flush=True)
    try:
        project_dates = query_project_dates()
        print(f"Official USGS projects with dates: {len(project_dates)}", flush=True)
    except Exception as error:
        project_dates = {}
        print(
            "WARNING: USGS collection-index attributes are unavailable; "
            "project recency will use official product publication dates and "
            "the previously verified city acquisition interval. "
            f"Reason: {type(error).__name__}: {error}",
            flush=True,
        )
    for number, city in enumerate(inventory.itertuples(), start=1):
        slug = str(city.city_slug)
        print(f"[{number}/{len(inventory)}] {slug}", flush=True)
        prior_record = records.get(slug, {})
        if prior_record.get("status") == "complete_lidar_deleted":
            output_value = prior_record.get("output_ndsm_path", "")
            output = (PROJECT_ROOT / output_value).resolve() if output_value else Path()
            if not output.is_file():
                raise RuntimeError(
                    f"Manifest says {slug} is complete/deleted but its nDSM is missing: {output}"
                )
            template, compatible, _ = choose_planet_grid(slug)
            builder = load_builder_module()
            for scene in compatible:
                builder.verify_output(scene, output)
            with rasterio.open(output) as src:
                if src.count != 3:
                    raise RuntimeError(f"Completed output is not three-band: {output}")
            print("  skipped: validated nDSM exists and manifest-listed LiDAR was deleted")
            continue
        record = {column: "" for column in MANIFEST_COLUMNS}
        record.update({"city_slug": slug, "city_name": city.city_name, "last_checked_utc": utc_now()})
        try:
            aoi = load_aoi(str(city.aoi_path))
            city_tiles = tiles[tiles["city_slug"] == slug].copy()
            if city_tiles.empty:
                raise RuntimeError("Detailed USGS tile metadata is missing for city")
            selected, projects, collect_start, collect_end, coverage = select_latest_acquisition(
                aoi, city_tiles, project_dates,
                str(city.lidar_acquisition_start_date),
                str(city.lidar_acquisition_end_date),
            )
            exact_project_dates = all(
                project_dates.get(project, {}).get("end") for project in projects
            )
            footprint = find_footprints(slug)
            template, compatible, outliers = choose_planet_grid(slug)
            city_scenes = scenes[scenes["city_slug"] == slug].copy()
            acquired = pd.to_datetime(city_scenes["acquired"], utc=True, errors="raise")
            lidar_end = pd.Timestamp(collect_end, tz="UTC")
            after_count = int((acquired >= lidar_end).sum())
            temporal_fallback = after_count < len(acquired)
            record.update(
                {
                    "selected_project_directories": ";".join(projects),
                    "lidar_collect_start": collect_start, "lidar_collect_end": collect_end,
                    "lidar_date_source": (
                        "live_usgs_project_collection_dates" if exact_project_dates
                        else "verified_inventory_interval"
                    ),
                    "project_selection_basis": (
                        "collection_end_date" if exact_project_dates
                        else "official_product_publication_date_fallback"
                    ),
                    "planet_scene_count": len(city_scenes),
                    "planet_scenes_on_or_after_lidar": after_count,
                    "temporal_fallback_used": temporal_fallback,
                    "planned_tile_count": len(selected),
                    "planned_coverage_percent": f"{coverage:.6f}",
                    "footprint_path": relative(footprint),
                    "planet_template_path": relative(template),
                    "planet_majority_scene_count": len(compatible),
                    "planet_outlier_scene_count": len(outliers),
                    "status": "planned_dry_run" if args.dry_run else "downloading",
                }
            )
            records[slug] = record
            tile_manifest = write_tile_manifest(slug, selected)
            record["lidar_tile_manifest_path"] = relative(tile_manifest)
            checkpoint(records)
            if args.dry_run:
                continue

            lidar_paths, downloaded_bytes = download_tiles(slug, selected, args.minimum_free_gb)
            tile_manifest = write_tile_manifest(slug, selected)
            record.update(
                {"downloaded_tile_count": len(lidar_paths), "downloaded_bytes": downloaded_bytes,
                 "status": "downloaded_verified"}
            )
            records[slug] = record
            checkpoint(records)
            output, class_audit, valid_pixels, building_pixels = build_three_band_ndsm(
                slug, lidar_paths, footprint, template, compatible,
                args.ground_class, projects, args.overwrite,
            )
            record.update(
                {
                    "classification_audit_path": relative(class_audit),
                    "output_ndsm_path": relative(output),
                    "output_valid_pixels": valid_pixels,
                    "output_building_pixels": building_pixels,
                    "status": "ndsm_validated",
                }
            )
            if args.confirm_delete_lidar:
                delete_manifest_lidar(tile_manifest, output, slug)
                record["lidar_deleted"] = True
                record["status"] = "complete_lidar_deleted"
            else:
                record["lidar_deleted"] = False
                record["status"] = "complete_lidar_retained"
            records[slug] = record
            checkpoint(records)
        except Exception as error:
            failures += 1
            record["status"] = "failed"
            record["error_message"] = f"{type(error).__name__}: {error}"
            records[slug] = record
            checkpoint(records)
            print(f"  FAILED: {record['error_message']}", flush=True)
    if failures:
        raise RuntimeError(f"{failures} US cities failed; inspect {relative(RUN_MANIFEST)}")
    print(f"SUCCESS: manifest checkpointed at {relative(RUN_MANIFEST)}", flush=True)


if __name__ == "__main__":
    main()
