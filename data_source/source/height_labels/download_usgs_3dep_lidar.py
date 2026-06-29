"""
Download USGS 3DEP LiDAR Tiles for U.S. City AOIs

Environment: data_source/source/height_labels/venv_height_labels

Requires:
    - data_source/data/city_aois/generated/city_buffers_5km_by_city/
      <city_slug>_5km.geojson

Produces:
    - data_source/data/height_labels/generated/usgs_3dep_projects.csv
    - data_source/data/height_labels/generated/usgs_3dep_tile_manifest.csv
    - data_source/data/height_labels/source/<city_slug>/usgs_3dep/
      <project_name>/*.laz

Description:
    Queries official USGS National Map services, retains only approved 3DEP
    projects, and downloads only LAZ tiles whose footprints intersect each
    city's 5 km AOI. Downloads are resumable and written atomically so a
    partial network transfer is never mistaken for a complete source tile.

Usage:
    python3 data_source/source/height_labels/download_usgs_3dep_lidar.py --dry-run
    python3 data_source/source/height_labels/download_usgs_3dep_lidar.py --dry-run --estimate-sizes
    python3 data_source/source/height_labels/download_usgs_3dep_lidar.py --confirm-download
    python3 data_source/source/height_labels/download_usgs_3dep_lidar.py --city boston --confirm-download

Expected runtime:
    Dry-run metadata discovery takes several minutes when estimating sizes.
    Full downloads may take hours and require substantial disk space.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
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


relaunch_inside_venv()

import pandas as pd
from pyproj import CRS, Transformer
from shapely.geometry import box, shape
from shapely.ops import transform, unary_union


TNM_PRODUCTS_URL = "https://tnmaccess.nationalmap.gov/api/v1/products"
ELEVATION_INDEX_QUERY_URL = (
    "https://index.nationalmap.gov/arcgis/rest/services/"
    "3DEPElevationIndex/MapServer/8/query"
)
HTTP_USER_AGENT = "building-height-prediction/1.0 USGS-3DEP-research"
PAGE_SIZE = 500
HTTP_ATTEMPTS = 4
DOWNLOAD_CHUNK_BYTES = 8 * 1024 * 1024

# These projects were selected after comparing all projects returned for each
# AOI. The newest project with near-complete coverage is used. New York City's
# circular AOI crosses New York and New Jersey, so two projects are necessary.
CITY_PROJECTS = {
    "boston": ["MA_CentralEastern_2021_B21"],
    "chicago": ["IL_4_County_QL1_LiDAR_2016_B16"],
    "los_angeles": ["CA_LosAngeles_B23"],
    "new_york_city": [
        "NY_New_York_CMGP_SANDY_LiDAR_15",
        "NJ_New_Jersey_SANDY_LiDAR_15",
    ],
    "san_francisco": ["CA_SanFrancisco_B23"],
    "seattle": ["WA_KingCounty_2021_B21"],
}

PROJECT_COLUMNS = [
    "city_slug",
    "project_directory",
    "project",
    "project_id",
    "workunit",
    "workunit_id",
    "collect_start",
    "collect_end",
    "quality_level",
    "specification",
    "collection_method",
    "horizontal_crs",
    "vertical_crs",
    "geoid",
    "lpc_publication_date",
    "lpc_link",
    "metadata_link",
    "selected_tile_count",
    "aoi_coverage_percent",
]

TILE_COLUMNS = [
    "city_slug",
    "project_directory",
    "source_id",
    "title",
    "publication_date",
    "format",
    "download_url",
    "filename",
    "min_lon",
    "min_lat",
    "max_lon",
    "max_lat",
    "aoi_intersection_area_m2",
    "expected_bytes",
    "local_path",
    "download_status",
    "downloaded_bytes",
    "sha256",
    "last_checked_utc",
]


def parse_args() -> argparse.Namespace:
    """Read command-line options and enforce explicit download confirmation."""
    parser = argparse.ArgumentParser(
        description="Discover and download AOI-intersecting USGS 3DEP LAZ tiles."
    )
    parser.add_argument(
        "--city",
        action="append",
        choices=sorted(CITY_PROJECTS),
        help="Limit the run to one or more city slugs. May be repeated.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Discover tiles and write manifests without downloading LiDAR.",
    )
    parser.add_argument(
        "--confirm-download",
        action="store_true",
        help="Required to download selected LAZ files.",
    )
    parser.add_argument(
        "--estimate-sizes",
        action="store_true",
        help="Send HEAD requests to estimate total download bytes.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Parallel metadata/download workers. Default: 4.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace complete local LAZ files instead of resuming around them.",
    )
    return parser.parse_args()


def project_path(*parts: str) -> Path:
    """Build a path from the detected repository root."""
    return PROJECT_ROOT.joinpath(*parts)


def relative_project_path(path: Path) -> str:
    """Store a portable repository-relative path in manifests."""
    resolved = path.resolve()
    if not resolved.is_relative_to(PROJECT_ROOT):
        raise ValueError(f"Path is outside the project repository: {resolved}")
    return str(resolved.relative_to(PROJECT_ROOT))


def request_json(url: str, parameters: dict[str, Any]) -> dict[str, Any]:
    """Request JSON with retries and an identifiable, polite user agent."""
    request_url = f"{url}?{urllib.parse.urlencode(parameters)}"
    request = urllib.request.Request(
        request_url,
        headers={"User-Agent": HTTP_USER_AGENT},
    )
    for attempt in range(1, HTTP_ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.load(response)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            if attempt == HTTP_ATTEMPTS:
                raise
            time.sleep(5 * attempt)
    raise RuntimeError("Unreachable request retry state")


def load_aoi(city_slug: str):
    """Load one city-specific 5 km AOI as a Shapely geometry."""
    path = project_path(
        "data_source",
        "data",
        "city_aois",
        "generated",
        "city_buffers_5km_by_city",
        f"{city_slug}_5km.geojson",
    )
    if not path.exists():
        raise FileNotFoundError(f"Missing city AOI: {path}")
    document = json.loads(path.read_text(encoding="utf-8"))
    features = document.get("features") or []
    if len(features) != 1:
        raise ValueError(f"Expected exactly one AOI feature in {path}")
    return shape(features[0]["geometry"])


def local_equal_area_geometry(geometry):
    """Project a city geometry to its local UTM CRS for area calculations."""
    centroid = geometry.centroid
    utm_zone = int((centroid.x + 180) // 6) + 1
    epsg = 32600 + utm_zone if centroid.y >= 0 else 32700 + utm_zone
    transformer = Transformer.from_crs(
        CRS.from_epsg(4326),
        CRS.from_epsg(epsg),
        always_xy=True,
    )
    return transform(transformer.transform, geometry), transformer


def project_directory_from_item(item: dict[str, Any]) -> str:
    """Extract the staged USGS project directory from a tile download URL."""
    download_url = item.get("downloadURL") or item.get("downloadLazURL") or ""
    match = re.search(r"/Projects/([^/]+)/", download_url)
    return match.group(1) if match else ""


def query_all_products(aoi) -> list[dict[str, Any]]:
    """Retrieve every National Map LPC product in the AOI bounding box."""
    min_lon, min_lat, max_lon, max_lat = aoi.bounds
    all_items: list[dict[str, Any]] = []
    offset = 0
    total = None

    while True:
        payload = request_json(
            TNM_PRODUCTS_URL,
            {
                "datasets": "Lidar Point Cloud (LPC)",
                "bbox": f"{min_lon},{min_lat},{max_lon},{max_lat}",
                "max": PAGE_SIZE,
                "offset": offset,
            },
        )
        if payload.get("error"):
            raise RuntimeError(f"USGS products API error: {payload['error']}")
        total = int(payload.get("total", 0))
        page = payload.get("items") or []
        all_items.extend(page)
        offset += PAGE_SIZE
        if offset >= total or not page:
            break

    # ScienceBase can occasionally return duplicate product records. Source ID
    # plus URL uniquely identifies a downloadable tile for this workflow.
    unique_items: dict[tuple[str, str], dict[str, Any]] = {}
    for item in all_items:
        url = item.get("downloadURL") or item.get("downloadLazURL") or ""
        unique_items[(str(item.get("sourceId", "")), url)] = item
    return list(unique_items.values())


def select_intersecting_tiles(
    city_slug: str,
    aoi,
    items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Keep approved-project tiles whose bounding boxes touch the actual AOI."""
    selected_projects = set(CITY_PROJECTS[city_slug])
    _, transformer = local_equal_area_geometry(aoi)
    selected: list[dict[str, Any]] = []

    for item in items:
        project_directory = project_directory_from_item(item)
        if project_directory not in selected_projects:
            continue

        bounds = item.get("boundingBox") or {}
        required_bounds = ("minX", "minY", "maxX", "maxY")
        if not all(key in bounds for key in required_bounds):
            raise ValueError(f"USGS item is missing a bounding box: {item}")

        tile_geometry = box(
            float(bounds["minX"]),
            float(bounds["minY"]),
            float(bounds["maxX"]),
            float(bounds["maxY"]),
        )
        intersection = aoi.intersection(tile_geometry)
        if intersection.is_empty:
            continue

        intersection_projected = transform(transformer.transform, intersection)
        download_url = item.get("downloadURL") or item.get("downloadLazURL")
        if not download_url:
            raise ValueError(f"USGS item is missing a download URL: {item}")
        filename = Path(urllib.parse.urlparse(download_url).path).name
        local_path = project_path(
            "data_source",
            "data",
            "height_labels",
            "source",
            city_slug,
            "usgs_3dep",
            project_directory,
            filename,
        )
        selected.append(
            {
                "city_slug": city_slug,
                "project_directory": project_directory,
                "source_id": str(item.get("sourceId", "")),
                "title": str(item.get("title", "")),
                "publication_date": str(item.get("publicationDate", "")),
                "format": str(item.get("format", "")),
                "download_url": download_url,
                "filename": filename,
                "min_lon": float(bounds["minX"]),
                "min_lat": float(bounds["minY"]),
                "max_lon": float(bounds["maxX"]),
                "max_lat": float(bounds["maxY"]),
                "aoi_intersection_area_m2": intersection_projected.area,
                "expected_bytes": "",
                "local_path": relative_project_path(local_path),
                "download_status": "selected",
                "downloaded_bytes": "",
                "sha256": "",
                "last_checked_utc": "",
            }
        )

    return selected


def epoch_milliseconds_to_date(value: Any) -> str:
    """Convert ArcGIS epoch milliseconds to an ISO date."""
    if value in (None, ""):
        return ""
    return datetime.fromtimestamp(float(value) / 1000, timezone.utc).date().isoformat()


def query_project_metadata(city_slug: str, aoi) -> list[dict[str, Any]]:
    """Read authoritative 3DEP work-unit metadata intersecting a city AOI."""
    min_lon, min_lat, max_lon, max_lat = aoi.bounds
    fields = (
        "workunit,workunit_id,project,project_id,collect_start,collect_end,"
        "ql,spec,p_method,horiz_crs,vert_crs,geoid,lpc_pub_date,lpc_link,"
        "metadata_link"
    )
    payload = request_json(
        ELEVATION_INDEX_QUERY_URL,
        {
            "f": "json",
            "where": "1=1",
            "geometry": f"{min_lon},{min_lat},{max_lon},{max_lat}",
            "geometryType": "esriGeometryEnvelope",
            "inSR": 4326,
            "spatialRel": "esriSpatialRelIntersects",
            "outFields": fields,
            "returnGeometry": "false",
            "resultRecordCount": 2000,
        },
    )
    if payload.get("error"):
        raise RuntimeError(f"USGS elevation index error: {payload['error']}")

    selected_directories = set(CITY_PROJECTS[city_slug])
    records: list[dict[str, Any]] = []
    for feature in payload.get("features") or []:
        attributes = feature["attributes"]
        lpc_link = str(attributes.get("lpc_link", ""))
        match = re.search(r"/Projects/([^/]+)", lpc_link)
        project_directory = match.group(1) if match else ""
        if project_directory not in selected_directories:
            continue
        records.append(
            {
                "city_slug": city_slug,
                "project_directory": project_directory,
                "project": attributes.get("project", ""),
                "project_id": attributes.get("project_id", ""),
                "workunit": attributes.get("workunit", ""),
                "workunit_id": attributes.get("workunit_id", ""),
                "collect_start": epoch_milliseconds_to_date(
                    attributes.get("collect_start")
                ),
                "collect_end": epoch_milliseconds_to_date(attributes.get("collect_end")),
                "quality_level": attributes.get("ql", ""),
                "specification": attributes.get("spec", ""),
                "collection_method": attributes.get("p_method", ""),
                "horizontal_crs": attributes.get("horiz_crs", ""),
                "vertical_crs": attributes.get("vert_crs", ""),
                "geoid": attributes.get("geoid", ""),
                "lpc_publication_date": epoch_milliseconds_to_date(
                    attributes.get("lpc_pub_date")
                ),
                "lpc_link": lpc_link,
                "metadata_link": attributes.get("metadata_link", ""),
                "selected_tile_count": "",
                "aoi_coverage_percent": "",
            }
        )
    return records


def calculate_project_coverage(aoi, tile_rows: list[dict[str, Any]]) -> float:
    """Calculate the percent of the AOI covered by selected tile rectangles."""
    aoi_projected, transformer = local_equal_area_geometry(aoi)
    tile_rectangles = [
        box(row["min_lon"], row["min_lat"], row["max_lon"], row["max_lat"])
        for row in tile_rows
    ]
    if not tile_rectangles:
        return 0.0
    covered = aoi.intersection(unary_union(tile_rectangles))
    covered_projected = transform(transformer.transform, covered)
    return 100 * covered_projected.area / aoi_projected.area


def head_content_length(row: dict[str, Any]) -> tuple[str, int | None]:
    """Read a tile's HTTP content length without downloading its body."""
    request = urllib.request.Request(
        row["download_url"],
        method="HEAD",
        headers={"User-Agent": HTTP_USER_AGENT},
    )
    for attempt in range(1, HTTP_ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                value = response.headers.get("Content-Length")
                return row["download_url"], int(value) if value else None
        except (urllib.error.URLError, TimeoutError):
            if attempt == HTTP_ATTEMPTS:
                return row["download_url"], None
            time.sleep(5 * attempt)
    return row["download_url"], None


def estimate_sizes(tile_rows: list[dict[str, Any]], workers: int) -> None:
    """Populate expected byte sizes with parallel HTTP HEAD requests."""
    sizes: dict[str, int | None] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(head_content_length, row) for row in tile_rows]
        for completed, future in enumerate(as_completed(futures), start=1):
            url, size = future.result()
            sizes[url] = size
            if completed % 50 == 0 or completed == len(futures):
                print(f"Size checks: {completed}/{len(futures)}", flush=True)
    for row in tile_rows:
        row["expected_bytes"] = sizes.get(row["download_url"]) or ""


def sha256_file(path: Path) -> str:
    """Calculate a SHA-256 digest without loading a large LAZ file into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(DOWNLOAD_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def recover_verified_partial(row: dict[str, Any], local_path: Path) -> bool:
    """Promote a complete, checksummed .part file left by cloud sync timing."""
    partial_path = local_path.with_suffix(local_path.suffix + ".part")
    if not partial_path.is_file():
        return False

    expected_bytes = int(row["expected_bytes"]) if row["expected_bytes"] else None
    if expected_bytes is None or partial_path.stat().st_size != expected_bytes:
        return False

    expected_sha256 = str(row.get("sha256", "")).strip()
    if expected_sha256 and sha256_file(partial_path) != expected_sha256:
        return False

    partial_path.replace(local_path)
    return True


def download_tile(row: dict[str, Any], overwrite: bool) -> dict[str, Any]:
    """Download one LAZ tile atomically and return updated manifest values."""
    local_path = project_path(*Path(row["local_path"]).parts)
    expected_bytes = int(row["expected_bytes"]) if row["expected_bytes"] else None

    if not local_path.exists() and recover_verified_partial(row, local_path):
        local_bytes = local_path.stat().st_size
        return {
            **row,
            "download_status": "recovered",
            "downloaded_bytes": local_bytes,
            "sha256": sha256_file(local_path),
            "last_checked_utc": datetime.now(timezone.utc).isoformat(),
        }

    if local_path.exists() and not overwrite:
        local_bytes = local_path.stat().st_size
        if expected_bytes is None or local_bytes == expected_bytes:
            return {
                **row,
                "download_status": "existing",
                "downloaded_bytes": local_bytes,
                "sha256": sha256_file(local_path),
                "last_checked_utc": datetime.now(timezone.utc).isoformat(),
            }

    local_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = local_path.with_suffix(local_path.suffix + ".part")

    for attempt in range(1, HTTP_ATTEMPTS + 1):
        try:
            request = urllib.request.Request(
                row["download_url"],
                headers={"User-Agent": HTTP_USER_AGENT},
            )
            with urllib.request.urlopen(request, timeout=180) as response:
                response_size = response.headers.get("Content-Length")
                if response_size:
                    expected_bytes = int(response_size)
                with partial_path.open("wb") as output:
                    shutil.copyfileobj(response, output, DOWNLOAD_CHUNK_BYTES)

            downloaded_bytes = partial_path.stat().st_size
            if expected_bytes is not None and downloaded_bytes != expected_bytes:
                raise IOError(
                    f"Byte-size mismatch: {downloaded_bytes} != {expected_bytes}"
                )
            partial_path.replace(local_path)
            return {
                **row,
                "expected_bytes": expected_bytes or "",
                "download_status": "downloaded",
                "downloaded_bytes": downloaded_bytes,
                "sha256": sha256_file(local_path),
                "last_checked_utc": datetime.now(timezone.utc).isoformat(),
            }
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            if partial_path.exists():
                partial_path.unlink()
            if attempt == HTTP_ATTEMPTS:
                return {
                    **row,
                    "download_status": f"failed:{type(error).__name__}",
                    "last_checked_utc": datetime.now(timezone.utc).isoformat(),
                }
            time.sleep(10 * attempt)
    raise RuntimeError("Unreachable download retry state")


def atomic_write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    """Write a CSV through a temporary file so interrupted writes stay valid."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(rows, columns=columns).to_csv(temporary_path, index=False)
    temporary_path.replace(path)


def merge_previous_manifest(
    discovered_rows: list[dict[str, Any]],
    manifest_path: Path,
) -> list[dict[str, Any]]:
    """Carry forward download bookkeeping for unchanged USGS tile URLs."""
    if not manifest_path.exists():
        return discovered_rows
    previous = pd.read_csv(manifest_path, dtype=str).fillna("")
    previous_by_url = previous.set_index("download_url").to_dict("index")
    bookkeeping = (
        "expected_bytes",
        "download_status",
        "downloaded_bytes",
        "sha256",
        "last_checked_utc",
    )
    for row in discovered_rows:
        old = previous_by_url.get(row["download_url"])
        if old:
            for column in bookkeeping:
                row[column] = old.get(column, row[column])
    return discovered_rows


def preserve_other_cities(
    active_rows: list[dict[str, Any]],
    existing_path: Path,
    active_cities: list[str],
) -> list[dict[str, Any]]:
    """Keep existing rows for cities outside a city-specific run."""
    if not existing_path.exists():
        return active_rows
    previous = pd.read_csv(existing_path, dtype=str).fillna("")
    inactive_rows = previous[
        ~previous["city_slug"].isin(active_cities)
    ].to_dict("records")
    return active_rows + inactive_rows


def main() -> None:
    """Discover projects/tiles, write manifests, and optionally download LAZ."""
    args = parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    if args.dry_run == args.confirm_download:
        raise SystemExit(
            "Choose exactly one mode: --dry-run or --confirm-download."
        )

    cities = args.city or list(CITY_PROJECTS)
    project_rows: list[dict[str, Any]] = []
    tile_rows: list[dict[str, Any]] = []
    for city_slug in cities:
        print(f"Discovering {city_slug}...", flush=True)
        aoi = load_aoi(city_slug)
        products = query_all_products(aoi)
        city_tiles = select_intersecting_tiles(city_slug, aoi, products)
        if not city_tiles:
            raise RuntimeError(f"No selected USGS tiles intersect {city_slug}")
        tile_rows.extend(city_tiles)

        metadata_rows = query_project_metadata(city_slug, aoi)
        metadata_by_directory = {
            row["project_directory"]: row for row in metadata_rows
        }
        for project_directory in CITY_PROJECTS[city_slug]:
            project_tiles = [
                row
                for row in city_tiles
                if row["project_directory"] == project_directory
            ]
            if project_directory not in metadata_by_directory:
                raise RuntimeError(
                    f"Missing USGS project metadata for {city_slug}: "
                    f"{project_directory}"
                )
            metadata = metadata_by_directory[project_directory]
            metadata["selected_tile_count"] = len(project_tiles)
            metadata["aoi_coverage_percent"] = calculate_project_coverage(
                aoi,
                project_tiles,
            )
            project_rows.append(metadata)

        combined_coverage = calculate_project_coverage(aoi, city_tiles)
        print(
            f"  selected_tiles={len(city_tiles)} "
            f"combined_aoi_coverage={combined_coverage:.2f}%",
            flush=True,
        )
        if combined_coverage < 95:
            raise RuntimeError(
                f"Selected projects cover only {combined_coverage:.2f}% "
                f"of the {city_slug} AOI"
            )

    manifest_path = project_path(
        "data_source",
        "data",
        "height_labels",
        "generated",
        "usgs_3dep_tile_manifest.csv",
    )
    projects_path = project_path(
        "data_source",
        "data",
        "height_labels",
        "generated",
        "usgs_3dep_projects.csv",
    )
    active_tile_rows = merge_previous_manifest(tile_rows, manifest_path)
    all_tile_rows = preserve_other_cities(
        active_tile_rows,
        manifest_path,
        cities,
    )
    all_project_rows = preserve_other_cities(
        project_rows,
        projects_path,
        cities,
    )

    if args.estimate_sizes:
        print(f"Estimating sizes for {len(active_tile_rows)} tiles...", flush=True)
        estimate_sizes(active_tile_rows, args.workers)
        # Size values changed after all_tile_rows was assembled. Rebuild the
        # combined table so the updated active-city rows are written.
        all_tile_rows = preserve_other_cities(
            active_tile_rows,
            manifest_path,
            cities,
        )

    atomic_write_csv(projects_path, all_project_rows, PROJECT_COLUMNS)
    atomic_write_csv(manifest_path, all_tile_rows, TILE_COLUMNS)
    print(f"WROTE {projects_path}", flush=True)
    print(f"WROTE {manifest_path}", flush=True)

    known_bytes = sum(
        int(row["expected_bytes"])
        for row in active_tile_rows
        if row["expected_bytes"]
    )
    unknown_sizes = sum(not bool(row["expected_bytes"]) for row in active_tile_rows)
    print(f"Selected cities: {len(cities)}", flush=True)
    print(f"Selected tiles: {len(active_tile_rows)}", flush=True)
    print(f"Known download size: {known_bytes / 1024**3:.2f} GiB", flush=True)
    print(f"Tiles with unknown size: {unknown_sizes}", flush=True)

    if args.dry_run:
        print("DRY RUN COMPLETE: no LiDAR files downloaded.", flush=True)
        return

    completed_rows: list[dict[str, Any]] = []
    print(f"Downloading with {args.workers} workers...", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(download_tile, row, args.overwrite): row
            for row in active_tile_rows
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            completed_rows.append(result)
            status = result["download_status"]
            print(
                f"[{completed}/{len(active_tile_rows)}] {result['city_slug']} "
                f"{result['filename']} status={status}",
                flush=True,
            )
            # Preserve completed work after every tile. Rows still pending keep
            # their discovery state until their futures return.
            completed_by_url = {
                row["download_url"]: row for row in completed_rows
            }
            checkpoint_rows = [
                completed_by_url.get(row["download_url"], row) for row in all_tile_rows
            ]
            atomic_write_csv(manifest_path, checkpoint_rows, TILE_COLUMNS)

    failed = [
        row for row in completed_rows if str(row["download_status"]).startswith("failed:")
    ]
    if failed:
        raise RuntimeError(
            f"{len(failed)} USGS tiles failed after retries. "
            f"See {manifest_path} for exact rows."
        )
    print("ALL SELECTED USGS LIDAR TILES DOWNLOADED.", flush=True)


if __name__ == "__main__":
    main()
