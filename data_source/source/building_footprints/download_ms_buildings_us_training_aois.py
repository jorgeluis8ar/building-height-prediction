#!/usr/bin/env python3
"""Download and clip Microsoft building footprints for 52 US training cities.

Requires (inputs from earlier stages):
    - data_source/data/height_labels/generated/training_open_lidar/
      training_cities_with_open_lidar.csv
    - data_source/data/city_aois/generated/
      wup2018_city_buffers_5km_by_city/<city_slug>_5km.geojson
    - Microsoft's public Global ML Building Footprints link index

Produces (outputs for the US LiDAR-to-nDSM stage):
    - data_source/data/building_footprints/generated/<city_slug>/
      <city_slug>_building_footprints_5km.gpkg
    - data_source/data/building_footprints/generated/
      ms_buildings_us_training_city_manifest.csv
    - data_source/data/building_footprints/generated/
      ms_buildings_us_partition_manifest.csv

The program deliberately excludes Boston and Seattle because those two cities
already have verified local footprint layers. Microsoft source partitions are
cached as immutable raw inputs. Output geometries are clipped to the exact city
AOI and saved in that AOI's declared CRS. Microsoft model-derived height values
are intentionally omitted to prevent target leakage into the height model.
"""

from __future__ import annotations

import argparse
import atexit
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import threading
import time
from typing import Any
import urllib.error
import urllib.parse
import urllib.request


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
VENV_DIR = SCRIPT_DIR / "venv_building_footprints"
VENV_PYTHON = VENV_DIR / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
VENV_MARKER = "BUILDING_FOOTPRINTS_VENV_ACTIVE"


def relaunch_inside_venv() -> None:
    """Run with the folder-specific environment on Windows, macOS, or Linux."""
    if os.environ.get(VENV_MARKER) == "1" or Path(sys.prefix).resolve() == VENV_DIR.resolve():
        return
    if not VENV_PYTHON.is_file():
        raise SystemExit(f"Missing building-footprint environment: {VENV_PYTHON}")
    environment = os.environ.copy()
    environment[VENV_MARKER] = "1"
    if os.name == "nt":
        # Waiting for the child is essential on Windows. Otherwise the parent
        # can return while the child is still writing the shared manifest.
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
import pandas as pd
from pyproj import CRS, Transformer
from shapely import make_valid
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, box, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as shapely_transform, unary_union


INVENTORY_PATH = PROJECT_ROOT / (
    "data_source/data/height_labels/generated/training_open_lidar/"
    "training_cities_with_open_lidar.csv"
)
AOI_DIR = PROJECT_ROOT / (
    "data_source/data/city_aois/generated/wup2018_city_buffers_5km_by_city"
)
SOURCE_ROOT = PROJECT_ROOT / "data_source/data/building_footprints/source/ms_buildings_us"
OUTPUT_ROOT = PROJECT_ROOT / "data_source/data/building_footprints/generated"
CITY_MANIFEST_PATH = OUTPUT_ROOT / "ms_buildings_us_training_city_manifest.csv"
PARTITION_MANIFEST_PATH = OUTPUT_ROOT / "ms_buildings_us_partition_manifest.csv"
LOG_DIR = OUTPUT_ROOT / "logs"

# This is a dated, reproducible Microsoft release rather than a moving alias.
DEFAULT_LINK_INDEX_URL = (
    "https://bfppub.blob.core.windows.net/$web/2026-07-24/dataset-links.csv"
)
MICROSOFT_REGION = "UnitedStates"
EXPECTED_US_CITY_COUNT = 54
EXCLUDED_EXISTING_CITIES = {"boston_22939", "seattle_23140"}
EXPECTED_DOWNLOAD_CITY_COUNT = 52
QUADKEY_LEVEL = 9
OUTPUT_LAYER = "building_footprints_5km"
DOWNLOAD_CHUNK_BYTES = 8 * 1024 * 1024
USER_AGENT = "building-height-prediction/1.0 ms-buildings-us-aoi"
# Clipping occurs in the AOI CRS, while the boundary audit occurs in a local
# metric CRS. Reprojection can create microscopic coordinate slivers along a
# shared boundary. Ignore only slivers lying within 5 cm of that boundary;
# anything beyond the buffered AOI remains a hard failure.
BOUNDARY_POSITION_TOLERANCE_METERS = 0.05
BOUNDARY_AREA_TOLERANCE_M2 = 0.01

CITY_MANIFEST_COLUMNS = [
    "city_slug",
    "city_name",
    "status",
    "aoi_path",
    "aoi_crs",
    "output_crs",
    "source_release_date",
    "source_link_index_url",
    "quadkeys",
    "partition_count",
    "estimated_partition_bytes",
    "candidate_feature_count",
    "intersecting_feature_count",
    "duplicate_count_removed",
    "boundary_clipped_count",
    "final_footprint_count",
    "final_footprint_area_m2",
    "output_path",
    "last_checked_utc",
    "error_message",
]

PARTITION_MANIFEST_COLUMNS = [
    "release_date",
    "region",
    "quadkey",
    "url",
    "advertised_size",
    "advertised_bytes",
    "local_path",
    "status",
    "downloaded_bytes",
    "last_checked_utc",
    "error_message",
]


class StreamTee:
    """Write every terminal message to a dated log, including thread output."""

    def __init__(self, terminal: Any, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.terminal = terminal
        self.file = path.open("w", encoding="utf-8")
        self.lock = threading.Lock()

    def write(self, message: str) -> int:
        with self.lock:
            self.terminal.write(message)
            self.file.write(message)
            self.file.flush()
        return len(message)

    def flush(self) -> None:
        with self.lock:
            self.terminal.flush()
            self.file.flush()

    def close(self) -> None:
        if not self.file.closed:
            self.file.close()


def start_log() -> Path:
    """Start logging before validation so early failures remain visible."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = LOG_DIR / f"download_ms_buildings_us_training_aois_{timestamp}.log"
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
    """Read explicit safe modes and bounded, resumable batch options."""
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan partitions only; no building partition is downloaded and no GPKG is written.",
    )
    mode.add_argument(
        "--confirm-download",
        action="store_true",
        help="Download required public partitions and write validated city GeoPackages.",
    )
    parser.add_argument("--city-slug", action="append", dest="city_slugs")
    parser.add_argument("--city-offset", type=int, default=0)
    parser.add_argument(
        "--city-limit",
        type=int,
        default=1,
        help="Cities to process after the offset; zero means all remaining cities.",
    )
    parser.add_argument("--download-workers", type=int, default=4)
    parser.add_argument("--download-retries", type=int, default=5)
    parser.add_argument("--download-timeout-seconds", type=int, default=900)
    parser.add_argument("--minimum-free-gb", type=float, default=100.0)
    parser.add_argument("--link-index-url", default=DEFAULT_LINK_INDEX_URL)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def relative(path: Path) -> str:
    """Return a portable path and reject anything outside the repository."""
    resolved = path.resolve()
    if not resolved.is_relative_to(PROJECT_ROOT):
        raise ValueError(f"Path lies outside repository: {resolved}")
    return resolved.relative_to(PROJECT_ROOT).as_posix()


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    """Replace a complete CSV atomically so partial state never looks valid."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, quoting=csv.QUOTE_MINIMAL)
    temporary.replace(path)


def load_existing_manifest(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame(columns=columns)
    frame = pd.read_csv(path, dtype=str).fillna("")
    missing = set(columns) - set(frame.columns)
    if missing:
        raise RuntimeError(f"Existing manifest {path} lacks columns: {sorted(missing)}")
    return frame[columns]


def checkpoint_city_manifest(records: dict[str, dict[str, Any]]) -> None:
    frame = pd.DataFrame(records.values(), columns=CITY_MANIFEST_COLUMNS)
    if not frame.empty:
        frame = frame.sort_values("city_slug")
    atomic_write_csv(frame, CITY_MANIFEST_PATH)


def checkpoint_partition_manifest(records: dict[str, dict[str, Any]]) -> None:
    frame = pd.DataFrame(records.values(), columns=PARTITION_MANIFEST_COLUMNS)
    if not frame.empty:
        frame = frame.sort_values(["release_date", "quadkey", "url"])
    atomic_write_csv(frame, PARTITION_MANIFEST_PATH)


def load_target_cities() -> pd.DataFrame:
    """Return exactly the 52 USGS cities that still need footprint layers."""
    if not INVENTORY_PATH.is_file():
        raise FileNotFoundError(f"Missing training/open-LiDAR inventory: {INVENTORY_PATH}")
    frame = pd.read_csv(INVENTORY_PATH, dtype={"city_slug": str}).fillna("")
    required = {"city_slug", "city_name", "country", "lidar_source_program", "aoi_path"}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"Training inventory lacks columns: {sorted(missing)}")
    us = frame[
        (frame["country"] == "United States of America")
        & (frame["lidar_source_program"] == "USGS 3D Elevation Program (3DEP)")
    ].copy()
    if len(us) != EXPECTED_US_CITY_COUNT or us["city_slug"].nunique() != EXPECTED_US_CITY_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_US_CITY_COUNT} unique USGS training cities; "
            f"found {len(us)} rows and {us['city_slug'].nunique()} slugs"
        )
    missing_exclusions = EXCLUDED_EXISTING_CITIES - set(us["city_slug"])
    if missing_exclusions:
        raise RuntimeError(f"Expected existing-footprint cities are absent: {missing_exclusions}")
    targets = us[~us["city_slug"].isin(EXCLUDED_EXISTING_CITIES)].copy()
    if len(targets) != EXPECTED_DOWNLOAD_CITY_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_DOWNLOAD_CITY_COUNT} footprint targets; found {len(targets)}"
        )
    return targets.sort_values("city_slug").reset_index(drop=True)


def load_aoi(row: Any) -> tuple[gpd.GeoDataFrame, BaseGeometry, BaseGeometry, CRS]:
    """Load one declared city AOI and retain both native and WGS84 geometry."""
    aoi_path = (PROJECT_ROOT / str(row.aoi_path)).resolve()
    expected_path = (AOI_DIR / f"{row.city_slug}_5km.geojson").resolve()
    if aoi_path != expected_path:
        raise RuntimeError(
            f"Inventory AOI path does not match authoritative WUP path for {row.city_slug}: "
            f"{aoi_path} != {expected_path}"
        )
    if not aoi_path.is_file():
        raise FileNotFoundError(f"Missing city AOI: {aoi_path}")
    aoi = gpd.read_file(aoi_path)
    if aoi.empty or aoi.crs is None:
        raise RuntimeError(f"AOI is empty or has no declared CRS: {aoi_path}")
    native_crs = CRS.from_user_input(aoi.crs)
    repaired = [make_valid(geometry) for geometry in aoi.geometry if geometry is not None]
    native_geometry = unary_union([geometry for geometry in repaired if not geometry.is_empty])
    if native_geometry.is_empty:
        raise RuntimeError(f"AOI contains no usable geometry: {aoi_path}")
    aoi = gpd.GeoDataFrame({"city_slug": [row.city_slug]}, geometry=[native_geometry], crs=native_crs)
    wgs84_geometry = aoi.to_crs(4326).geometry.iloc[0]
    return aoi, native_geometry, wgs84_geometry, native_crs


def latitude_to_tile_y(latitude: float, level: int) -> int:
    """Convert latitude to a Web Mercator tile row."""
    latitude = max(-85.05112878, min(85.05112878, latitude))
    scale = 2**level
    value = (1 - math.asinh(math.tan(math.radians(latitude))) / math.pi) / 2
    return max(0, min(scale - 1, int(math.floor(value * scale))))


def longitude_to_tile_x(longitude: float, level: int) -> int:
    """Convert longitude to a Web Mercator tile column."""
    scale = 2**level
    value = (longitude + 180.0) / 360.0
    return max(0, min(scale - 1, int(math.floor(value * scale))))


def tile_xy_to_quadkey(tile_x: int, tile_y: int, level: int) -> str:
    """Encode one Bing Maps tile coordinate as a quadkey string."""
    digits = []
    for bit in range(level, 0, -1):
        mask = 1 << (bit - 1)
        digit = (1 if tile_x & mask else 0) + (2 if tile_y & mask else 0)
        digits.append(str(digit))
    return "".join(digits)


def tile_bounds(tile_x: int, tile_y: int, level: int) -> tuple[float, float, float, float]:
    """Return a Bing tile's WGS84 bounds."""
    scale = 2**level

    def tile_latitude(y_value: int) -> float:
        mercator = math.pi * (1 - 2 * y_value / scale)
        return math.degrees(math.atan(math.sinh(mercator)))

    min_lon = tile_x / scale * 360.0 - 180.0
    max_lon = (tile_x + 1) / scale * 360.0 - 180.0
    max_lat = tile_latitude(tile_y)
    min_lat = tile_latitude(tile_y + 1)
    return min_lon, min_lat, max_lon, max_lat


def intersecting_quadkeys(aoi_wgs84: BaseGeometry) -> list[str]:
    """Compute level-9 partitions locally without uploading the AOI anywhere."""
    min_lon, min_lat, max_lon, max_lat = aoi_wgs84.bounds
    min_x = longitude_to_tile_x(min_lon, QUADKEY_LEVEL)
    max_x = longitude_to_tile_x(max_lon, QUADKEY_LEVEL)
    min_y = latitude_to_tile_y(max_lat, QUADKEY_LEVEL)
    max_y = latitude_to_tile_y(min_lat, QUADKEY_LEVEL)
    quadkeys = []
    for tile_y in range(min_y, max_y + 1):
        for tile_x in range(min_x, max_x + 1):
            bounds = tile_bounds(tile_x, tile_y, QUADKEY_LEVEL)
            if box(*bounds).intersects(aoi_wgs84):
                quadkeys.append(tile_xy_to_quadkey(tile_x, tile_y, QUADKEY_LEVEL))
    if not quadkeys:
        raise RuntimeError("AOI intersects no level-9 quadkey")
    return sorted(set(quadkeys))


def release_date_from_url(url: str) -> str:
    match = re.search(r"/(20\d{2}-\d{2}-\d{2})/", url)
    if not match:
        raise ValueError(
            "--link-index-url must contain a dated release path such as /2026-07-24/"
        )
    return match.group(1)


def download_small_index(url: str, release_date: str) -> Path:
    """Cache the public link index atomically; it contains no footprint payload."""
    path = SOURCE_ROOT / "link_indexes" / f"dataset_links_{release_date}.csv"
    if path.is_file() and path.stat().st_size > 0:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=300) as response, partial.open("wb") as file:
            while True:
                chunk = response.read(DOWNLOAD_CHUNK_BYTES)
                if not chunk:
                    break
                file.write(chunk)
        if partial.stat().st_size == 0:
            raise IOError(f"Microsoft link index was empty: {url}")
        partial.replace(path)
    except Exception:
        # The link index is small and is safe to restart. Do not leave a file
        # that could be mistaken for a complete metadata index.
        if partial.exists():
            partial.unlink()
        raise
    return path


def parse_advertised_bytes(value: str) -> int:
    """Convert Microsoft's human-readable approximate size to bytes."""
    match = re.fullmatch(r"\s*([0-9.]+)\s*([KMGT]?B)\s*", value, flags=re.IGNORECASE)
    if not match:
        return 0
    factors = {"B": 1, "KB": 1000, "MB": 1000**2, "GB": 1000**3, "TB": 1000**4}
    return int(float(match.group(1)) * factors[match.group(2).upper()])


def load_us_link_index(path: Path, release_date: str) -> dict[str, list[dict[str, Any]]]:
    """Read only the pinned United States records from Microsoft's link table."""
    output: dict[str, list[dict[str, Any]]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required = {"Location", "QuadKey", "Url", "Size", "UploadDate"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise RuntimeError(f"Microsoft link index lacks columns: {sorted(required)}")
        for row in reader:
            if row["Location"] != MICROSOFT_REGION:
                continue
            if row["UploadDate"] != release_date:
                raise RuntimeError(
                    f"Index release mismatch: URL says {release_date}, row says {row['UploadDate']}"
                )
            quadkey = str(row["QuadKey"])
            if len(quadkey) != QUADKEY_LEVEL or set(quadkey) - set("0123"):
                raise RuntimeError(f"Invalid level-{QUADKEY_LEVEL} quadkey: {quadkey}")
            output.setdefault(quadkey, []).append(
                {
                    "release_date": release_date,
                    "region": MICROSOFT_REGION,
                    "quadkey": quadkey,
                    "url": row["Url"],
                    "advertised_size": row["Size"],
                    "advertised_bytes": parse_advertised_bytes(row["Size"]),
                }
            )
    if not output:
        raise RuntimeError(f"No {MICROSOFT_REGION} partitions found in {path}")
    return output


def partition_local_path(partition: dict[str, Any]) -> Path:
    filename = Path(urllib.parse.urlparse(partition["url"]).path).name
    if not filename.endswith(".csv.gz"):
        raise RuntimeError(f"Unexpected Microsoft partition filename: {filename}")
    return SOURCE_ROOT / partition["release_date"] / partition["quadkey"] / filename


def partition_key(partition: dict[str, Any]) -> str:
    return f"{partition['release_date']}|{partition['quadkey']}|{partition['url']}"


def download_one_partition(
    partition: dict[str, Any],
    number: int,
    total: int,
    minimum_free_gb: float,
    retries: int,
    timeout_seconds: int,
) -> tuple[Path, int]:
    """Download or resume one immutable gzip partition without shared writes."""
    path = partition_local_path(partition)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.stat().st_size > 0:
        print(
            f"    partition {number}/{total}: existing {partition['quadkey']} "
            f"({path.stat().st_size / 1e6:.1f} MB)",
            flush=True,
        )
        return path, path.stat().st_size
    partial = path.with_suffix(path.suffix + ".partial")
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        free_gb = shutil.disk_usage(path.parent).free / 1e9
        if free_gb < minimum_free_gb:
            raise OSError(f"Free disk {free_gb:.2f} GB is below {minimum_free_gb:.2f} GB")
        offset = partial.stat().st_size if partial.is_file() else 0
        headers = {"User-Agent": USER_AGENT}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = urllib.request.Request(partition["url"], headers=headers)
        print(
            f"    partition {number}/{total}: quadkey={partition['quadkey']} "
            f"attempt={attempt}/{retries} resume={offset / 1e6:.1f} MB",
            flush=True,
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                status = getattr(response, "status", response.getcode())
                resume_accepted = offset > 0 and status == 206
                mode = "ab" if resume_accepted else "wb"
                starting_size = offset if resume_accepted else 0
                content_length = response.headers.get("Content-Length")
                expected_size = starting_size + int(content_length) if content_length else None
                with partial.open(mode) as file:
                    while True:
                        chunk = response.read(DOWNLOAD_CHUNK_BYTES)
                        if not chunk:
                            break
                        file.write(chunk)
            actual_size = partial.stat().st_size if partial.is_file() else 0
            if actual_size == 0:
                raise IOError(f"Empty Microsoft partition: {partition['url']}")
            if expected_size is not None and actual_size != expected_size:
                raise IOError(
                    f"Incomplete partition {partition['quadkey']}: expected {expected_size}, "
                    f"received {actual_size} bytes"
                )
            # Reading the gzip footer catches truncated or corrupt transfers.
            with gzip.open(partial, "rb") as file:
                for _ in iter(lambda: file.read(DOWNLOAD_CHUNK_BYTES), b""):
                    pass
            partial.replace(path)
            print(
                f"    partition {number}/{total}: complete {partition['quadkey']} "
                f"({actual_size / 1e6:.1f} MB)",
                flush=True,
            )
            return path, actual_size
        except urllib.error.HTTPError as error:
            last_error = error
            if error.code == 416 and partial.exists():
                partial.unlink()
        except Exception as error:
            last_error = error
        if attempt < retries:
            delay = min(60, 5 * (2 ** (attempt - 1)))
            print(
                f"    partition {number}/{total}: retrying in {delay}s after "
                f"{type(last_error).__name__}: {last_error}",
                flush=True,
            )
            time.sleep(delay)
    raise RuntimeError(
        f"Partition {partition['quadkey']} failed after {retries} attempts; "
        f"partial retained at {relative(partial)}; last error: {last_error}"
    )


def download_partitions(
    partitions: list[dict[str, Any]],
    workers: int,
    minimum_free_gb: float,
    retries: int,
    timeout_seconds: int,
    records: dict[str, dict[str, Any]],
) -> dict[str, Path]:
    """Download unique partitions concurrently; checkpoint once as coordinator."""
    unique = {partition_key(item): item for item in partitions}
    items = [unique[key] for key in sorted(unique)]
    worker_count = min(workers, len(items))
    print(f"Downloading {len(items)} unique partition(s) with {worker_count} worker(s)")
    paths: dict[str, Path] = {}
    failures = []
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(
                download_one_partition,
                partition,
                number,
                len(items),
                minimum_free_gb,
                retries,
                timeout_seconds,
            ): partition
            for number, partition in enumerate(items, start=1)
        }
        for future in as_completed(futures):
            partition = futures[future]
            key = partition_key(partition)
            record = {
                **{column: "" for column in PARTITION_MANIFEST_COLUMNS},
                **partition,
                "local_path": relative(partition_local_path(partition)),
                "last_checked_utc": utc_now(),
            }
            try:
                path, downloaded_bytes = future.result()
                paths[key] = path
                record.update({"status": "downloaded_verified", "downloaded_bytes": downloaded_bytes})
            except Exception as error:
                record.update(
                    {"status": "failed", "error_message": f"{type(error).__name__}: {error}"}
                )
                failures.append(record["error_message"])
            records[key] = record
            checkpoint_partition_manifest(records)
    if failures:
        raise RuntimeError(f"{len(failures)} Microsoft partitions failed: {' | '.join(failures)}")
    return paths


def polygonal_only(geometry: BaseGeometry) -> BaseGeometry:
    """Retain polygonal parts after validity repair or AOI intersection."""
    if geometry.is_empty:
        return Polygon()
    if isinstance(geometry, (Polygon, MultiPolygon)):
        return geometry
    if isinstance(geometry, GeometryCollection):
        polygons = [part for part in geometry.geoms if isinstance(part, (Polygon, MultiPolygon))]
        return unary_union(polygons) if polygons else Polygon()
    return Polygon()


def geometry_hash(geometry: BaseGeometry) -> str:
    """Create a stable exact-geometry identifier after clipping."""
    normalized = geometry.normalize()
    return hashlib.sha256(normalized.wkb).hexdigest()


def read_and_clip_city(
    city_slug: str,
    aoi_native: gpd.GeoDataFrame,
    aoi_wgs84: BaseGeometry,
    native_crs: CRS,
    partitions: list[dict[str, Any]],
) -> tuple[gpd.GeoDataFrame, dict[str, int]]:
    """Stream candidate GeoJSONL records and clip matches to the native AOI CRS."""
    rows: list[dict[str, Any]] = []
    candidate_count = 0
    intersecting_count = 0
    boundary_clipped_count = 0
    transformer = None
    if native_crs != CRS.from_epsg(4326):
        transformer = Transformer.from_crs(4326, native_crs, always_xy=True)
    for partition in partitions:
        path = partition_local_path(partition)
        if not path.is_file():
            raise FileNotFoundError(f"Required partition was not downloaded: {path}")
        try:
            with gzip.open(path, "rt", encoding="utf-8") as file:
                for line_number, line in enumerate(file, start=1):
                    if not line.strip():
                        continue
                    candidate_count += 1
                    try:
                        feature = json.loads(line)
                        source_geometry = polygonal_only(make_valid(shape(feature["geometry"])))
                    except Exception as error:
                        raise RuntimeError(
                            f"Invalid GeoJSONL record {path}:{line_number}: {error}"
                        ) from error
                    if source_geometry.is_empty or not source_geometry.intersects(aoi_wgs84):
                        continue
                    intersecting_count += 1
                    native_geometry = (
                        shapely_transform(transformer.transform, source_geometry)
                        if transformer is not None
                        else source_geometry
                    )
                    clipped = polygonal_only(make_valid(native_geometry.intersection(aoi_native.geometry.iloc[0])))
                    if clipped.is_empty or clipped.area <= 0:
                        continue
                    was_clipped = not clipped.equals(native_geometry)
                    boundary_clipped_count += int(was_clipped)
                    properties = feature.get("properties") or {}
                    rows.append(
                        {
                            "building_footprint_id": geometry_hash(clipped),
                            "city_slug": city_slug,
                            "footprint_source": "Microsoft Global ML Building Footprints",
                            "source_release_date": partition["release_date"],
                            "source_quadkey": partition["quadkey"],
                            "source_confidence": properties.get("confidence", -1),
                            "geometry_clipped_to_aoi": was_clipped,
                            "aoi_selection_rule": "exact_intersection_with_5km_aoi",
                            "geometry": clipped,
                        }
                    )
        except (OSError, EOFError) as error:
            raise RuntimeError(f"Corrupt gzip partition {path}: {error}") from error
    if not rows:
        raise RuntimeError(f"Microsoft returned no footprints intersecting {city_slug}")
    frame = gpd.GeoDataFrame(rows, geometry="geometry", crs=native_crs)
    before = len(frame)
    frame = frame.drop_duplicates(subset="building_footprint_id", keep="first").copy()
    frame = frame.sort_values("building_footprint_id").reset_index(drop=True)
    return frame, {
        "candidate_feature_count": candidate_count,
        "intersecting_feature_count": intersecting_count,
        "duplicate_count_removed": before - len(frame),
        "boundary_clipped_count": boundary_clipped_count,
    }


def output_path(city_slug: str) -> Path:
    """Return the exact path required by run_us_lidar_to_planet_ndsm.py."""
    return OUTPUT_ROOT / city_slug / f"{city_slug}_building_footprints_5km.gpkg"


def metric_area(frame: gpd.GeoDataFrame, aoi: gpd.GeoDataFrame) -> float:
    """Measure area in a local metric CRS without changing final output CRS."""
    metric_crs = aoi.estimate_utm_crs()
    if metric_crs is None:
        raise RuntimeError("Could not determine local metric CRS for footprint area audit")
    return float(frame.to_crs(metric_crs).geometry.area.sum())


def boundary_overflow_area_m2(
    frame: gpd.GeoDataFrame, aoi: gpd.GeoDataFrame
) -> tuple[float, float]:
    """Return raw and material footprint area outside the AOI in square metres.

    ``raw`` includes harmless floating-point slivers introduced when the
    independently reprojected AOI and clipped geometries are reconstructed.
    ``material`` excludes only geometry within five centimetres of the AOI
    boundary and is the value used for the fail-loud validation.
    """
    metric_crs = aoi.estimate_utm_crs()
    if metric_crs is None:
        raise RuntimeError("Could not determine metric CRS for boundary validation")
    metric_geometries = frame.to_crs(metric_crs).geometry
    metric_aoi = aoi.to_crs(metric_crs).geometry.iloc[0]
    raw_outside = metric_geometries.difference(metric_aoi)
    tolerated_aoi = metric_aoi.buffer(BOUNDARY_POSITION_TOLERANCE_METERS)
    material_outside = metric_geometries.difference(tolerated_aoi)
    return float(raw_outside.area.sum()), float(material_outside.area.sum())


def write_and_validate_output(
    frame: gpd.GeoDataFrame,
    aoi: gpd.GeoDataFrame,
    city_slug: str,
    native_crs: CRS,
    overwrite: bool,
) -> tuple[Path, float]:
    """Write one atomic GeoPackage and prove LiDAR-stage compatibility."""
    path = output_path(city_slug)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists; use --overwrite after inspection: {path}")
    temporary = path.with_name(path.stem + ".tmp.gpkg")
    if temporary.exists():
        temporary.unlink()
    frame.to_file(temporary, layer=OUTPUT_LAYER, driver="GPKG", engine="pyogrio")
    check = gpd.read_file(temporary, layer=OUTPUT_LAYER)
    if len(check) != len(frame):
        raise RuntimeError(f"GeoPackage feature count changed: {len(frame)} -> {len(check)}")
    if check.crs is None or CRS.from_user_input(check.crs) != native_crs:
        raise RuntimeError(f"Output CRS does not equal AOI CRS: {check.crs} != {native_crs}")
    forbidden = {column for column in check.columns if column.lower() == "height"}
    if forbidden:
        raise RuntimeError(f"Microsoft height leakage detected in output columns: {forbidden}")
    if check.empty or check.geometry.isna().any() or check.geometry.is_empty.any():
        raise RuntimeError("Output contains missing or empty geometries")
    if not check.geometry.is_valid.all():
        raise RuntimeError("Output contains invalid geometries")
    raw_outside_m2, material_outside_m2 = boundary_overflow_area_m2(check, aoi)
    if material_outside_m2 > BOUNDARY_AREA_TOLERANCE_M2:
        raise RuntimeError(
            "Output geometries materially extend outside AOI: "
            f"raw={raw_outside_m2:.6f} m2, beyond_5cm={material_outside_m2:.6f} m2"
        )
    area_m2 = float(check.to_crs(metric_crs).geometry.area.sum())
    if path.exists():
        path.unlink()
    temporary.replace(path)
    # The LiDAR script looks for exactly this filename in exactly this folder.
    expected = OUTPUT_ROOT / city_slug / f"{city_slug}_building_footprints_5km.gpkg"
    if path.resolve() != expected.resolve() or not path.is_file():
        raise RuntimeError(f"LiDAR-compatible output path validation failed: {path}")
    return path, area_m2


def validate_existing_output(city_slug: str, aoi: gpd.GeoDataFrame, native_crs: CRS) -> int:
    """Validate a prior complete output before treating a rerun as successful."""
    path = output_path(city_slug)
    if not path.is_file():
        return 0
    frame = gpd.read_file(path, layer=OUTPUT_LAYER)
    if frame.empty or frame.crs is None or CRS.from_user_input(frame.crs) != native_crs:
        raise RuntimeError(f"Existing output is empty or has the wrong CRS: {path}")
    if "height" in {column.lower() for column in frame.columns}:
        raise RuntimeError(f"Existing output contains forbidden Microsoft height: {path}")
    raw_outside_m2, material_outside_m2 = boundary_overflow_area_m2(frame, aoi)
    if material_outside_m2 > BOUNDARY_AREA_TOLERANCE_M2:
        raise RuntimeError(
            f"Existing output materially extends outside its AOI: {path}; "
            f"raw={raw_outside_m2:.6f} m2, beyond_5cm={material_outside_m2:.6f} m2"
        )
    return len(frame)


def main() -> None:
    args = parse_args()
    log_path = start_log()
    print(f"Run log: {relative(log_path)}", flush=True)
    if args.city_offset < 0 or args.city_limit < 0 or args.minimum_free_gb < 0:
        raise ValueError("Offsets, limits, and minimum free space must be nonnegative")
    if args.download_workers < 1 or args.download_retries < 1:
        raise ValueError("Download workers and retries must be at least one")
    if args.download_timeout_seconds < 1:
        raise ValueError("Download timeout must be at least one second")

    targets = load_target_cities()
    if args.city_slugs:
        unknown = sorted(set(args.city_slugs) - set(targets["city_slug"]))
        if unknown:
            raise ValueError(
                f"Unknown or already-covered Microsoft target city slugs: {unknown}"
            )
        targets = targets[targets["city_slug"].isin(args.city_slugs)]
    targets = targets.iloc[args.city_offset:]
    if args.city_limit:
        targets = targets.iloc[: args.city_limit]
    if targets.empty:
        raise RuntimeError("No cities remain after applying city filters")

    release_date = release_date_from_url(args.link_index_url)
    print(f"Pinned Microsoft release: {release_date}")
    print("Stage 1/5: reading public link index and planning local quadkeys")
    link_index_path = download_small_index(args.link_index_url, release_date)
    link_index = load_us_link_index(link_index_path, release_date)

    city_plans: dict[str, dict[str, Any]] = {}
    all_partitions: list[dict[str, Any]] = []
    for row in targets.itertuples():
        aoi, native_geometry, wgs84_geometry, native_crs = load_aoi(row)
        quadkeys = intersecting_quadkeys(wgs84_geometry)
        missing_quadkeys = [quadkey for quadkey in quadkeys if quadkey not in link_index]
        if missing_quadkeys:
            raise RuntimeError(
                f"Microsoft release lacks partitions for {row.city_slug}: {missing_quadkeys}"
            )
        partitions = [item for quadkey in quadkeys for item in link_index[quadkey]]
        city_plans[row.city_slug] = {
            "row": row,
            "aoi": aoi,
            "native_geometry": native_geometry,
            "wgs84_geometry": wgs84_geometry,
            "native_crs": native_crs,
            "quadkeys": quadkeys,
            "partitions": partitions,
        }
        all_partitions.extend(partitions)

    prior_cities = load_existing_manifest(CITY_MANIFEST_PATH, CITY_MANIFEST_COLUMNS)
    city_records = {row["city_slug"]: row.to_dict() for _, row in prior_cities.iterrows()}
    for slug, plan in city_plans.items():
        row = plan["row"]
        partitions = plan["partitions"]
        record = {column: "" for column in CITY_MANIFEST_COLUMNS}
        record.update(
            {
                "city_slug": slug,
                "city_name": row.city_name,
                "status": "planned_dry_run",
                "aoi_path": str(row.aoi_path),
                "aoi_crs": plan["native_crs"].to_string(),
                "source_release_date": release_date,
                "source_link_index_url": args.link_index_url,
                "quadkeys": ";".join(plan["quadkeys"]),
                "partition_count": len(partitions),
                "estimated_partition_bytes": sum(item["advertised_bytes"] for item in partitions),
                "output_path": relative(output_path(slug)),
                "last_checked_utc": utc_now(),
            }
        )
        city_records[slug] = record
    checkpoint_city_manifest(city_records)

    unique_partition_count = len({partition_key(item) for item in all_partitions})
    estimated_bytes = sum(
        item["advertised_bytes"]
        for item in {partition_key(value): value for value in all_partitions}.values()
    )
    print(
        f"Planned {len(city_plans)} city/cities, {unique_partition_count} unique partitions, "
        f"approximately {estimated_bytes / 1e9:.3f} GB compressed"
    )
    if args.dry_run:
        print("DRY RUN SUCCESS: no building partition downloaded and no GeoPackage written")
        return

    print("Stage 2/5: downloading and validating required public partitions")
    prior_partitions = load_existing_manifest(PARTITION_MANIFEST_PATH, PARTITION_MANIFEST_COLUMNS)
    partition_records = {
        partition_key(row.to_dict()): row.to_dict() for _, row in prior_partitions.iterrows()
    }
    download_partitions(
        all_partitions,
        args.download_workers,
        args.minimum_free_gb,
        args.download_retries,
        args.download_timeout_seconds,
        partition_records,
    )

    print("Stage 3/5: streaming, filtering, and clipping city footprints")
    failures = 0
    for number, (slug, plan) in enumerate(city_plans.items(), start=1):
        print(f"  [{number}/{len(city_plans)}] {slug}")
        record = city_records[slug]
        record.update({"status": "processing", "last_checked_utc": utc_now(), "error_message": ""})
        checkpoint_city_manifest(city_records)
        try:
            if output_path(slug).is_file() and not args.overwrite:
                count = validate_existing_output(slug, plan["aoi"], plan["native_crs"])
                record.update(
                    {
                        "status": "existing_output_validated",
                        "output_crs": plan["native_crs"].to_string(),
                        "final_footprint_count": count,
                        "last_checked_utc": utc_now(),
                    }
                )
                city_records[slug] = record
                checkpoint_city_manifest(city_records)
                print(f"    existing validated output retained ({count} footprints)")
                continue
            frame, counts = read_and_clip_city(
                slug,
                plan["aoi"],
                plan["wgs84_geometry"],
                plan["native_crs"],
                plan["partitions"],
            )
            record.update(counts)
            record["final_footprint_count"] = len(frame)
            print("Stage 4/5: writing LiDAR-compatible GeoPackage")
            path, area_m2 = write_and_validate_output(
                frame, plan["aoi"], slug, plan["native_crs"], args.overwrite
            )
            record.update(
                {
                    "status": "complete",
                    "output_crs": plan["native_crs"].to_string(),
                    "final_footprint_area_m2": f"{area_m2:.3f}",
                    "output_path": relative(path),
                    "last_checked_utc": utc_now(),
                }
            )
            city_records[slug] = record
            checkpoint_city_manifest(city_records)
            print(f"    complete: {len(frame)} footprints -> {relative(path)}")
        except Exception as error:
            failures += 1
            record.update(
                {
                    "status": "failed",
                    "last_checked_utc": utc_now(),
                    "error_message": f"{type(error).__name__}: {error}",
                }
            )
            city_records[slug] = record
            checkpoint_city_manifest(city_records)
            print(f"    FAILED: {record['error_message']}")

    print("Stage 5/5: validating complete requested city set")
    requested = set(city_plans)
    completed = {
        slug
        for slug in requested
        if city_records[slug]["status"] in {"complete", "existing_output_validated"}
    }
    if failures or completed != requested:
        missing = sorted(requested - completed)
        raise RuntimeError(
            f"Microsoft footprint run incomplete: failures={failures}, missing={missing}; "
            f"inspect {relative(CITY_MANIFEST_PATH)}"
        )
    print(
        f"SUCCESS: {len(completed)} city output(s) validated for the LiDAR workflow; "
        f"manifest={relative(CITY_MANIFEST_PATH)}"
    )


if __name__ == "__main__":
    main()
