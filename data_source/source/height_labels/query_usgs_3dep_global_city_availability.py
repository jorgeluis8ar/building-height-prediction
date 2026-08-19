#!/usr/bin/env python3
"""Query USGS 3DEP LiDAR availability for the global WUP U.S. city AOIs.

This is deliberately metadata-only. It queries official USGS services,
calculates AOI coverage from returned tile footprints, and checks a small
byte-range from one representative URL per city. It never stores LAZ content.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any
import urllib.error
import urllib.parse
import urllib.request


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIRECTORY.parents[2]
VENV_DIRECTORY = SCRIPT_DIRECTORY / "venv_height_labels"
VENV_PYTHON = VENV_DIRECTORY / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
VENV_MARKER = "HEIGHT_LABELS_VENV_ACTIVE"


def relaunch_inside_venv() -> None:
    """Restart inside the task environment so spatial dependencies are stable."""
    if os.environ.get(VENV_MARKER) == "1":
        return
    if Path(sys.prefix).resolve() == VENV_DIRECTORY.resolve():
        os.environ[VENV_MARKER] = "1"
        return
    if not VENV_PYTHON.exists():
        raise FileNotFoundError(
            "Missing height-label environment. Expected executable: "
            f"{VENV_PYTHON}"
        )
    environment = os.environ.copy()
    environment[VENV_MARKER] = "1"
    os.execve(
        str(VENV_PYTHON),
        [str(VENV_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]],
        environment,
    )


relaunch_inside_venv()

from pyproj import CRS, Transformer
from shapely.geometry import box, shape
from shapely.ops import transform, unary_union


TNM_PRODUCTS_URL = "https://tnmaccess.nationalmap.gov/api/v1/products"
HTTP_USER_AGENT = "building-height-prediction/1.0 USGS-3DEP-metadata-audit"
PAGE_SIZE = 500
HTTP_ATTEMPTS = 4

CITY_FILE = (
    REPOSITORY_ROOT
    / "data_source/data/city_aois/generated/wup2018_cities_over_300k_2018.csv"
)
AOI_DIRECTORY = (
    REPOSITORY_ROOT
    / "data_source/data/city_aois/generated/wup2018_city_buffers_5km_by_city"
)
OUTPUT_DIRECTORY = REPOSITORY_ROOT / "data_source/data/height_labels/generated"
CITY_OUTPUT = OUTPUT_DIRECTORY / "usgs_3dep_global_city_availability.csv"
TILE_OUTPUT = OUTPUT_DIRECTORY / "usgs_3dep_global_city_tile_metadata.csv"
LOG_DIRECTORY = OUTPUT_DIRECTORY / "logs"

CITY_FIELDS = [
    "wup_urbancode",
    "city_slug",
    "city_name",
    "country",
    "centroid_latitude",
    "centroid_longitude",
    "query_status",
    "coverage_status",
    "aoi_coverage_percent",
    "intersecting_tile_count",
    "project_count",
    "project_directories",
    "download_urls_present",
    "representative_download_url",
    "representative_url_status",
    "representative_http_status",
    "query_timestamp_utc",
    "error_message",
]

TILE_FIELDS = [
    "wup_urbancode",
    "city_slug",
    "city_name",
    "source_id",
    "project_directory",
    "title",
    "publication_date",
    "format",
    "download_url",
    "min_lon",
    "min_lat",
    "max_lon",
    "max_lat",
    "aoi_intersection_area_m2",
]


def parse_arguments() -> argparse.Namespace:
    """Read options; none of them authorize downloading LiDAR files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--city",
        action="append",
        help="Query only this WUP city slug. May be repeated.",
    )
    parser.add_argument(
        "--skip-confirmed-pilots",
        action="store_true",
        help="Skip the six cities already represented in the old USGS manifest.",
    )
    parser.add_argument(
        "--request-delay-seconds",
        type=float,
        default=0.25,
        help="Polite delay between city queries. Default: 0.25 seconds.",
    )
    return parser.parse_args()


def request_json(url: str, parameters: dict[str, Any]) -> dict[str, Any]:
    """Request JSON with bounded retries and an identifiable user agent."""
    request_url = f"{url}?{urllib.parse.urlencode(parameters)}"
    request = urllib.request.Request(request_url, headers={"User-Agent": HTTP_USER_AGENT})
    for attempt in range(1, HTTP_ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.load(response)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            if attempt == HTTP_ATTEMPTS:
                raise
            time.sleep(5 * attempt)
    raise RuntimeError("Unreachable retry state")


def read_cities() -> list[dict[str, str]]:
    """Load all U.S. cities from the canonical WUP inventory."""
    if not CITY_FILE.is_file():
        raise FileNotFoundError(f"Missing city inventory: {CITY_FILE}")
    with CITY_FILE.open("r", encoding="utf-8-sig", newline="") as handle:
        cities = [
            row
            for row in csv.DictReader(handle)
            if row["country"] == "United States of America"
        ]
    if len(cities) != 144:
        raise RuntimeError(f"Expected 144 U.S. cities, found {len(cities)}")
    return cities


def load_aoi(city_slug: str):
    """Load one global WUP 5 km disk and validate its basic structure."""
    path = AOI_DIRECTORY / f"{city_slug}_5km.geojson"
    if not path.is_file():
        raise FileNotFoundError(f"Missing city AOI: {path}")
    document = json.loads(path.read_text(encoding="utf-8"))
    features = document.get("features") or []
    if len(features) != 1:
        raise ValueError(f"Expected exactly one feature in {path}")
    geometry = shape(features[0]["geometry"])
    if geometry.is_empty or not geometry.is_valid:
        raise ValueError(f"Invalid or empty AOI geometry: {path}")
    return geometry


def local_projection(geometry):
    """Return the AOI and transformer in a local UTM CRS for area calculations."""
    centroid = geometry.centroid
    zone = int((centroid.x + 180) // 6) + 1
    epsg = 32600 + zone if centroid.y >= 0 else 32700 + zone
    transformer = Transformer.from_crs(4326, CRS.from_epsg(epsg), always_xy=True)
    return transform(transformer.transform, geometry), transformer


def query_products(aoi) -> list[dict[str, Any]]:
    """Return all unique National Map LPC product records in an AOI bbox."""
    min_lon, min_lat, max_lon, max_lat = aoi.bounds
    items: list[dict[str, Any]] = []
    offset = 0
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
        page = payload.get("items") or []
        items.extend(page)
        offset += PAGE_SIZE
        if offset >= int(payload.get("total", 0)) or not page:
            break

    unique: dict[tuple[str, str], dict[str, Any]] = {}
    for item in items:
        url = item.get("downloadURL") or item.get("downloadLazURL") or ""
        unique[(str(item.get("sourceId", "")), str(url))] = item
    return list(unique.values())


def project_directory(item: dict[str, Any]) -> str:
    """Extract the staged USGS project directory from a product URL."""
    url = item.get("downloadURL") or item.get("downloadLazURL") or ""
    match = re.search(r"/Projects/([^/]+)/", url)
    return match.group(1) if match else ""


def select_tiles(city: dict[str, str], aoi, items: list[dict[str, Any]]):
    """Keep products whose advertised bounding boxes intersect the true AOI."""
    aoi_projected, transformer = local_projection(aoi)
    selected = []
    projected_footprints = []
    for item in items:
        bounds = item.get("boundingBox") or {}
        keys = ("minX", "minY", "maxX", "maxY")
        if not all(key in bounds for key in keys):
            continue
        footprint = box(*(float(bounds[key]) for key in keys))
        intersection = aoi.intersection(footprint)
        if intersection.is_empty:
            continue
        intersection_projected = transform(transformer.transform, intersection)
        projected_footprints.append(intersection_projected)
        url = item.get("downloadURL") or item.get("downloadLazURL") or ""
        selected.append(
            {
                "wup_urbancode": city["wup_urbancode"],
                "city_slug": city["city_slug"],
                "city_name": city["city_name"],
                "source_id": str(item.get("sourceId", "")),
                "project_directory": project_directory(item),
                "title": str(item.get("title", "")),
                "publication_date": str(item.get("publicationDate", "")),
                "format": str(item.get("format", "")),
                "download_url": str(url),
                "min_lon": bounds["minX"],
                "min_lat": bounds["minY"],
                "max_lon": bounds["maxX"],
                "max_lat": bounds["maxY"],
                "aoi_intersection_area_m2": f"{intersection_projected.area:.3f}",
            }
        )
    covered = unary_union(projected_footprints).intersection(aoi_projected) if projected_footprints else None
    coverage = 0.0 if covered is None else 100.0 * covered.area / aoi_projected.area
    return selected, min(100.0, coverage)


def check_representative_url(url: str) -> tuple[str, str]:
    """Request one byte to verify reachability without downloading a LAZ file."""
    if not url:
        return "missing_url", ""
    request = urllib.request.Request(
        url,
        headers={"User-Agent": HTTP_USER_AGENT, "Range": "bytes=0-0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            status = int(response.status)
            response.read(1)
        return ("reachable" if status in (200, 206) else "unexpected_status", str(status))
    except urllib.error.HTTPError as error:
        return "http_error", str(error.code)
    except (urllib.error.URLError, TimeoutError) as error:
        return "connection_error", str(error.reason if hasattr(error, "reason") else error)


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    """Write a complete CSV atomically so interrupted runs cannot look valid."""
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> None:
    """Run the metadata audit and preserve explicit failures in the output."""
    arguments = parse_arguments()
    cities = read_cities()
    requested = set(arguments.city or [])
    if requested:
        known = {city["city_slug"] for city in cities}
        unknown = requested - known
        if unknown:
            raise ValueError(f"Unknown U.S. city slug(s): {sorted(unknown)}")
        cities = [city for city in cities if city["city_slug"] in requested]

    pilot_names = {
        "New York-Newark",
        "Los Angeles-Long Beach-Santa Ana",
        "Chicago",
        "Boston",
        "San Francisco-Oakland",
        "Seattle",
    }
    if arguments.skip_confirmed_pilots:
        cities = [city for city in cities if city["city_name"] not in pilot_names]
    if not cities:
        raise RuntimeError("No cities remain after applying command-line filters")

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    LOG_DIRECTORY.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = LOG_DIRECTORY / f"query_usgs_3dep_global_city_availability_{timestamp}.log"
    city_rows = []
    tile_rows = []

    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"Started UTC: {datetime.now(timezone.utc).isoformat()}\n")
        log.write(f"Cities requested: {len(cities)}\n")
        log.write("Mode: metadata-only; no LAZ files will be downloaded\n")
        for index, city in enumerate(cities, start=1):
            checked_at = datetime.now(timezone.utc).isoformat()
            print(f"[{index}/{len(cities)}] {city['city_slug']}", flush=True)
            try:
                aoi = load_aoi(city["city_slug"])
                items = query_products(aoi)
                selected, coverage = select_tiles(city, aoi, items)
                urls = [row["download_url"] for row in selected if row["download_url"]]
                representative = urls[0] if urls else ""
                url_status, http_status = check_representative_url(representative)
                projects = sorted({row["project_directory"] for row in selected if row["project_directory"]})
                if not selected:
                    coverage_status = "not_found"
                elif coverage >= 99.0:
                    coverage_status = "ready_for_download"
                else:
                    coverage_status = "incomplete"
                query_status = "success"
                error_message = ""
                tile_rows.extend(selected)
            except Exception as error:  # Preserve the failure rather than hiding it.
                coverage = 0.0
                selected = []
                urls = []
                representative = ""
                url_status = "not_checked"
                http_status = ""
                projects = []
                coverage_status = "query_failed"
                query_status = "failed"
                error_message = f"{type(error).__name__}: {error}"
                print(f"  FAILED: {error_message}", flush=True)

            city_rows.append(
                {
                    "wup_urbancode": city["wup_urbancode"],
                    "city_slug": city["city_slug"],
                    "city_name": city["city_name"],
                    "country": city["country"],
                    "centroid_latitude": city["latitude"],
                    "centroid_longitude": city["longitude"],
                    "query_status": query_status,
                    "coverage_status": coverage_status,
                    "aoi_coverage_percent": f"{coverage:.6f}",
                    "intersecting_tile_count": len(selected),
                    "project_count": len(projects),
                    "project_directories": "|".join(projects),
                    "download_urls_present": len(urls),
                    "representative_download_url": representative,
                    "representative_url_status": url_status,
                    "representative_http_status": http_status,
                    "query_timestamp_utc": checked_at,
                    "error_message": error_message,
                }
            )
            log.write(json.dumps(city_rows[-1], ensure_ascii=False) + "\n")
            log.flush()
            time.sleep(max(0.0, arguments.request_delay_seconds))

        log.write(f"Completed UTC: {datetime.now(timezone.utc).isoformat()}\n")

    write_csv(CITY_OUTPUT, CITY_FIELDS, city_rows)
    write_csv(TILE_OUTPUT, TILE_FIELDS, tile_rows)
    failures = sum(row["query_status"] == "failed" for row in city_rows)
    print(f"Wrote {len(city_rows)} city records to {CITY_OUTPUT.relative_to(REPOSITORY_ROOT)}")
    print(f"Wrote {len(tile_rows)} tile records to {TILE_OUTPUT.relative_to(REPOSITORY_ROOT)}")
    print(f"Log: {log_path.relative_to(REPOSITORY_ROOT)}")
    if failures:
        raise RuntimeError(f"{failures} city queries failed; inspect the output and log")


if __name__ == "__main__":
    main()
