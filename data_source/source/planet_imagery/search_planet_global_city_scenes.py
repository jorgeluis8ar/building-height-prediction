"""
Search eligible PlanetScope scene metadata for the global WUP city AOIs.

Environment: data_source/source/planet_imagery/venv_planet_imagery

Requires (inputs from earlier stages):
    - data_source/data/city_aois/generated/wup2018_cities_over_300k_2018.csv
    - data_source/data/city_aois/generated/
      wup2018_city_buffers_5km_by_city/<city_slug>_5km.geojson

Produces (metadata-only outputs for later scene selection):
    - data_source/data/planet_imagery/generated/global_city_scene_metadata/
      by_city/<city_slug>_planet_scenes.csv
    - data_source/data/planet_imagery/generated/global_city_scene_metadata/
      search_window_manifest.csv
    - data_source/data/planet_imagery/generated/logs/
      search_planet_global_city_scenes_<UTC timestamp>.log

This script never activates assets, creates orders, or downloads imagery.
Every scene row retains the full Planet properties and item links as JSON, in
addition to flattened fields and scene/AOI coordinates used for ranking.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
VENV_DIR = SCRIPT_DIR / "venv_planet_imagery"
VENV_PYTHON = VENV_DIR / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
VENV_MARKER = "PLANET_IMAGERY_VENV_ACTIVE"
EQUAL_AREA_CRS = "EPSG:6933"


def relaunch_inside_venv() -> None:
    """Relaunch with the pinned Planet environment before third-party imports."""
    if os.environ.get(VENV_MARKER) == "1" or Path(sys.prefix).absolute() == VENV_DIR.absolute():
        return
    if not VENV_PYTHON.exists():
        raise SystemExit(
            "ERROR: Missing Planet imagery virtual environment. Recreate it "
            "from data_source/source/planet_imagery/requirements.txt."
        )
    environment = os.environ.copy()
    environment[VENV_MARKER] = "1"
    os.execve(
        str(VENV_PYTHON),
        [str(VENV_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]],
        environment,
    )


relaunch_inside_venv()

import pandas as pd
from planet import Auth, Planet, Session, data_filter
from planet.exceptions import BadGateway, PlanetError, ServerError, TooManyRequests
from pyproj import Transformer
from shapely.geometry import shape
from shapely.ops import transform


COMMON_PROPERTY_FIELDS = [
    "acquired", "published", "updated", "cloud_cover", "clear_percent",
    "visible_percent", "cloud_percent", "shadow_percent", "snow_ice_percent",
    "heavy_haze_percent", "light_haze_percent", "quality_category",
    "sun_azimuth", "sun_elevation", "satellite_azimuth", "view_angle",
    "ground_control", "gsd", "pixel_resolution", "instrument", "satellite_id",
    "strip_id", "provider", "item_type",
]
OUTPUT_COLUMNS = [
    "wup_urbancode", "city_slug", "city_name", "country", "population_people",
    "scene_id", *COMMON_PROPERTY_FIELDS, "aoi_coverage_percent",
    "aoi_centroid_longitude", "aoi_centroid_latitude",
    "scene_centroid_longitude", "scene_centroid_latitude",
    "scene_centroid_offset_km", "scene_centroid_bearing_degrees",
    "scene_centroid_direction", "scene_bbox_min_longitude",
    "scene_bbox_min_latitude", "scene_bbox_max_longitude",
    "scene_bbox_max_latitude", "scene_geometry_geojson", "properties_json",
    "item_links_json", "item_json",
]
MANIFEST_COLUMNS = [
    "city_slug", "window_start", "window_end", "status", "eligible_scene_count",
    "attempted_utc", "error",
]


def parse_args() -> argparse.Namespace:
    """Define filters and bounded batching for a large global search."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", default="2016-01-01")
    parser.add_argument(
        "--end-date",
        default=(datetime.now(timezone.utc) + timedelta(days=1)).date().isoformat(),
        help="Exclusive UTC date in YYYY-MM-DD format.",
    )
    parser.add_argument("--max-cloud-cover", type=float, default=0.30)
    parser.add_argument("--min-aoi-coverage", type=float, default=95.0)
    parser.add_argument("--city-offset", type=int, default=0)
    parser.add_argument(
        "--city-limit", type=int, default=25,
        help="Number of cities in this run. Use 0 for every remaining city.",
    )
    parser.add_argument("--city-slug", action="append", dest="city_slugs")
    parser.add_argument("--max-window-retries", type=int, default=5)
    parser.add_argument("--retry-base-delay", type=float, default=5.0)
    parser.add_argument("--request-pause", type=float, default=0.25)
    parser.add_argument(
        "--inventory", type=Path,
        default=PROJECT_ROOT / "data_source/data/city_aois/generated/wup2018_cities_over_300k_2018.csv",
    )
    parser.add_argument(
        "--aoi-dir", type=Path,
        default=PROJECT_ROOT / "data_source/data/city_aois/generated/wup2018_city_buffers_5km_by_city",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "data_source/data/planet_imagery/generated/global_city_scene_metadata",
    )
    return parser.parse_args()


def parse_date(value: str) -> datetime:
    """Parse a date and attach UTC, failing on ambiguous formats."""
    return datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def annual_windows(start: datetime, end: datetime):
    """Yield bounded yearly windows so failures can resume precisely."""
    cursor = start
    while cursor < end:
        try:
            next_cursor = cursor.replace(year=cursor.year + 1)
        except ValueError:
            next_cursor = cursor.replace(year=cursor.year + 1, day=28)
        yield cursor, min(next_cursor, end)
        cursor = next_cursor


def load_api_key() -> str | None:
    """Load the ignored local credential without printing its value."""
    credential_path = SCRIPT_DIR / "PLANET_API.py"
    if credential_path.exists():
        specification = importlib.util.spec_from_file_location("planet_credentials", credential_path)
        if specification is None or specification.loader is None:
            raise RuntimeError(f"Could not load Planet credentials from {credential_path}")
        module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(module)
        value = getattr(module, "PL_API_KEY", None)
        if value:
            return str(value)
    return os.environ.get("PL_API_KEY")


def load_inventory(path: Path) -> list[dict[str, str]]:
    """Read the global city list and reject malformed or duplicate rows."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing required city inventory: {path}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    required = {"wup_urbancode", "city_slug", "city_name", "country", "population_people"}
    missing = required.difference(rows[0].keys() if rows else [])
    if missing:
        raise ValueError(f"City inventory is missing columns: {sorted(missing)}")
    if len({row["city_slug"] for row in rows}) != len(rows):
        raise ValueError("City inventory contains duplicate city_slug values")
    return rows


def load_aoi(path: Path) -> dict:
    """Read exactly one AOI polygon from a city GeoJSON."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing required city AOI: {path}")
    with path.open(encoding="utf-8") as handle:
        document = json.load(handle)
    features = document.get("features", []) if document.get("type") == "FeatureCollection" else []
    if len(features) != 1 or not features[0].get("geometry"):
        raise ValueError(f"Expected exactly one valid AOI feature in {path}")
    return features[0]["geometry"]


def geometry_metrics(aoi_geojson: dict, scene_geojson: dict) -> dict[str, object]:
    """Calculate coverage and centroid displacement in an equal-area CRS."""
    transformer = Transformer.from_crs("EPSG:4326", EQUAL_AREA_CRS, always_xy=True)
    project = transformer.transform
    aoi_wgs84 = shape(aoi_geojson)
    scene_wgs84 = shape(scene_geojson)
    aoi_projected = transform(project, aoi_wgs84)
    scene_projected = transform(project, scene_wgs84)
    if aoi_projected.area <= 0 or scene_projected.is_empty:
        raise ValueError("AOI or scene geometry is empty")
    coverage = aoi_projected.intersection(scene_projected).area / aoi_projected.area * 100.0
    dx = scene_projected.centroid.x - aoi_projected.centroid.x
    dy = scene_projected.centroid.y - aoi_projected.centroid.y
    bearing = (math.degrees(math.atan2(dx, dy)) + 360.0) % 360.0
    directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    minimum_x, minimum_y, maximum_x, maximum_y = scene_wgs84.bounds
    return {
        "aoi_coverage_percent": round(min(100.0, max(0.0, coverage)), 6),
        "aoi_centroid_longitude": aoi_wgs84.centroid.x,
        "aoi_centroid_latitude": aoi_wgs84.centroid.y,
        "scene_centroid_longitude": scene_wgs84.centroid.x,
        "scene_centroid_latitude": scene_wgs84.centroid.y,
        "scene_centroid_offset_km": math.hypot(dx, dy) / 1_000.0,
        "scene_centroid_bearing_degrees": bearing,
        "scene_centroid_direction": directions[int((bearing + 22.5) // 45) % 8],
        "scene_bbox_min_longitude": minimum_x,
        "scene_bbox_min_latitude": minimum_y,
        "scene_bbox_max_longitude": maximum_x,
        "scene_bbox_max_latitude": maximum_y,
    }


def item_to_row(item: dict, city: dict[str, str], aoi: dict) -> dict[str, object]:
    """Preserve the full item metadata while flattening common ranking fields."""
    geometry = item.get("geometry")
    if not geometry:
        raise ValueError(f"Planet scene {item.get('id')} is missing geometry")
    properties = item.get("properties") or {}
    row: dict[str, object] = {
        "wup_urbancode": city["wup_urbancode"],
        "city_slug": city["city_slug"],
        "city_name": city["city_name"],
        "country": city["country"],
        "population_people": city["population_people"],
        "scene_id": item.get("id"),
        **{field: properties.get(field) for field in COMMON_PROPERTY_FIELDS},
        **geometry_metrics(aoi, geometry),
        "scene_geometry_geojson": json.dumps(geometry, ensure_ascii=False, separators=(",", ":"), sort_keys=True),
        "properties_json": json.dumps(properties, ensure_ascii=False, separators=(",", ":"), sort_keys=True, default=str),
        "item_links_json": json.dumps(item.get("_links") or {}, ensure_ascii=False, separators=(",", ":"), sort_keys=True, default=str),
        # Preserve the complete API item so new or uncommon Planet fields are
        # not lost merely because they were not anticipated in this script.
        "item_json": json.dumps(item, ensure_ascii=False, separators=(",", ":"), sort_keys=True, default=str),
    }
    return row


def atomic_write(frame: pd.DataFrame, path: Path, columns: list[str]) -> None:
    """Write a CSV atomically so interruption cannot masquerade as success."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.reindex(columns=columns).to_csv(temporary, index=False)
    temporary.replace(path)


def load_manifest(path: Path) -> pd.DataFrame:
    """Load the window manifest or return an empty compatible table."""
    if not path.exists():
        return pd.DataFrame(columns=MANIFEST_COLUMNS)
    frame = pd.read_csv(path, dtype=str).fillna("")
    missing = set(MANIFEST_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"Existing search manifest is missing columns: {sorted(missing)}")
    return frame[MANIFEST_COLUMNS]


def search_window(planet: Planet, aoi: dict, start: datetime, end: datetime, args: argparse.Namespace) -> list[dict]:
    """Retrieve a complete window, retrying transient API failures."""
    search_filter = data_filter.and_filter(
        [
            data_filter.date_range_filter(field_name="acquired", gte=start, lt=end),
            data_filter.range_filter(field_name="cloud_cover", lt=args.max_cloud_cover),
            data_filter.permission_filter(),
        ]
    )
    transient = (TooManyRequests, ServerError, BadGateway)
    for attempt in range(args.max_window_retries + 1):
        try:
            return list(
                planet.data.search(
                    item_types=["PSScene"], geometry=aoi,
                    search_filter=search_filter, sort="acquired asc", limit=0,
                )
            )
        except transient:
            if attempt >= args.max_window_retries:
                raise
            delay = min(args.retry_base_delay * (2**attempt), 120.0)
            print(f"Transient Planet error; retrying in {delay:.1f}s", flush=True)
            time.sleep(delay)
    raise RuntimeError("Unreachable retry state")


def main() -> None:
    """Run a bounded, resumable, metadata-only batch."""
    args = parse_args()
    started = datetime.now(timezone.utc)
    log_path = PROJECT_ROOT / "data_source/data/planet_imagery/generated/logs" / f"search_planet_global_city_scenes_{started.strftime('%Y%m%dT%H%M%SZ')}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_lines = ["status=RUNNING", f"started_utc={started.isoformat()}"]
    try:
        start, end = parse_date(args.start_date), parse_date(args.end_date)
        if start >= end:
            raise ValueError("start-date must be earlier than end-date")
        if not 0.0 <= args.max_cloud_cover <= 1.0:
            raise ValueError("max-cloud-cover must be between 0 and 1")
        if not 0.0 <= args.min_aoi_coverage <= 100.0:
            raise ValueError("min-aoi-coverage must be between 0 and 100")
        if args.city_offset < 0 or args.city_limit < 0:
            raise ValueError("city-offset and city-limit cannot be negative")

        inventory = load_inventory(args.inventory)
        if args.city_slugs:
            requested = set(args.city_slugs)
            known = {city["city_slug"] for city in inventory}
            if requested - known:
                raise ValueError(f"Unknown city slugs: {sorted(requested - known)}")
            selected = [city for city in inventory if city["city_slug"] in requested]
        else:
            selected = inventory[args.city_offset:]
            if args.city_limit:
                selected = selected[:args.city_limit]
        if not selected:
            raise ValueError("The requested batch contains zero cities")

        missing_aois = [
            city["city_slug"] for city in selected
            if not (args.aoi_dir / f"{city['city_slug']}_5km.geojson").is_file()
        ]
        if missing_aois:
            raise FileNotFoundError(f"Missing AOIs for {len(missing_aois)} selected cities: {missing_aois[:10]}")

        api_key = load_api_key()
        session = Session(auth=Auth.from_key(api_key)) if api_key else Session(auth=Auth.from_user_default_session())
        planet = Planet(session=session)
        manifest_path = args.output_dir / "search_window_manifest.csv"
        manifest = load_manifest(manifest_path)
        completed = {
            (row.city_slug, row.window_start, row.window_end)
            for row in manifest.itertuples(index=False) if row.status == "success"
        }
        print(f"Metadata-only search for {len(selected):,} cities. No assets will be activated, ordered, or downloaded.")

        for city_number, city in enumerate(selected, start=1):
            city_slug = city["city_slug"]
            aoi = load_aoi(args.aoi_dir / f"{city_slug}_5km.geojson")
            city_output = args.output_dir / "by_city" / f"{city_slug}_planet_scenes.csv"
            city_frame = pd.read_csv(city_output) if city_output.exists() else pd.DataFrame(columns=OUTPUT_COLUMNS)
            for window_start, window_end in annual_windows(start, end):
                key = (city_slug, window_start.date().isoformat(), window_end.date().isoformat())
                if key in completed:
                    continue
                print(f"[{city_number}/{len(selected)}] {city_slug}: {key[1]} to {key[2]}", flush=True)
                attempted = datetime.now(timezone.utc).isoformat()
                try:
                    items = search_window(planet, aoi, window_start, window_end, args)
                    rows = []
                    for item in items:
                        row = item_to_row(item, city, aoi)
                        if float(row["aoi_coverage_percent"]) >= args.min_aoi_coverage:
                            rows.append(row)
                    if rows:
                        city_frame = pd.concat([city_frame, pd.DataFrame(rows)], ignore_index=True)
                        city_frame = city_frame.drop_duplicates(subset=["city_slug", "scene_id"], keep="last")
                        city_frame = city_frame.sort_values(["acquired", "scene_id"])
                    atomic_write(city_frame, city_output, OUTPUT_COLUMNS)
                    manifest = manifest[
                        ~((manifest["city_slug"] == key[0]) & (manifest["window_start"] == key[1]) & (manifest["window_end"] == key[2]))
                    ]
                    manifest.loc[len(manifest)] = [key[0], key[1], key[2], "success", str(len(rows)), attempted, ""]
                    atomic_write(manifest, manifest_path, MANIFEST_COLUMNS)
                    completed.add(key)
                except Exception as error:
                    manifest = manifest[
                        ~((manifest["city_slug"] == key[0]) & (manifest["window_start"] == key[1]) & (manifest["window_end"] == key[2]))
                    ]
                    manifest.loc[len(manifest)] = [key[0], key[1], key[2], "failed", "", attempted, f"{type(error).__name__}: {error}"]
                    atomic_write(manifest, manifest_path, MANIFEST_COLUMNS)
                    raise RuntimeError(f"Planet search failed for {city_slug}, {key[1]} to {key[2]}; partial outputs are preserved and marked failed") from error
                if args.request_pause:
                    time.sleep(args.request_pause)

        log_lines[0] = "status=SUCCESS"
        log_lines.extend([f"selected_city_count={len(selected)}", f"output_dir={args.output_dir.relative_to(PROJECT_ROOT)}"])
        log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
        print(f"SUCCESS: completed batch of {len(selected):,} cities. Log: {log_path}")
    except BaseException:
        log_lines[0] = "status=FAILED"
        log_lines.append(traceback.format_exc())
        log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
        print(f"FAILED: see {log_path}", file=sys.stderr)
        raise


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted: completed windows remain checkpointed; run status is not clean.", file=sys.stderr)
        sys.exit(130)
    except PlanetError:
        sys.exit(1)
