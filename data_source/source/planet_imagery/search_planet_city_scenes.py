"""
Search Planet City Scenes

Environment: data_source/source/planet_imagery/venv_planet_imagery

Requires (inputs from earlier stages):
    - README.md
    - data_source/data/city_aois/generated/city_buffers_5km_by_city/<city>_5km.geojson

Produces (outputs for later stages):
    - data_source/data/planet_imagery/generated/cities_scenes_results_planet.csv

Description:
    Reads the current city list from README.md, uses each city's 5km AOI buffer,
    and searches Planet PSScene metadata for imagery that covers at least 95%
    of the AOI. This script is metadata-only: it never activates, orders, or
    downloads imagery assets.

Usage:
    python3 data_source/source/planet_imagery/search_planet_city_scenes.py

Expected runtime: depends on Planet API pagination and selected date range
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import sys
import time
from zoneinfo import ZoneInfo

SCRIPT_DIR = Path(__file__).resolve().parent
VENV_DIR = SCRIPT_DIR / "venv_planet_imagery"
VENV_PYTHON = VENV_DIR / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
VENV_MARKER = "PLANET_IMAGERY_VENV_ACTIVE"


def relaunch_inside_venv() -> None:
    """
    Relaunch this script with the local virtual environment Python.

    This keeps the command simple for the user:
        python3 data_source/source/planet_imagery/search_planet_city_scenes.py

    If that command is run with the system Python, the script immediately
    re-executes itself with `venv_planet_imagery/bin/python` so the pinned
    dependencies in this folder are used.
    """
    if os.environ.get(VENV_MARKER) == "1":
        return

    current_python = Path(sys.executable).absolute()
    expected_python = VENV_PYTHON.absolute()

    if current_python == expected_python or Path(sys.prefix).absolute() == VENV_DIR.absolute():
        os.environ[VENV_MARKER] = "1"
        return

    if not VENV_PYTHON.exists():
        print("ERROR: Missing planet_imagery virtual environment.")
        print(f"Expected Python executable: {VENV_PYTHON}")
        print("Create it with:")
        print("  python3 -m venv data_source/source/planet_imagery/venv_planet_imagery")
        print("  data_source/source/planet_imagery/venv_planet_imagery/bin/python -m pip install -r data_source/source/planet_imagery/requirements.txt")
        sys.exit(1)

    env = os.environ.copy()
    env[VENV_MARKER] = "1"
    os.execve(str(VENV_PYTHON), [str(VENV_PYTHON), str(Path(__file__).resolve())] + sys.argv[1:], env)


relaunch_inside_venv()

import pandas as pd
from planet import Auth, Planet, Session, data_filter
from planet.exceptions import BadGateway, PlanetError, ServerError, TooManyRequests
from pyproj import Transformer
from shapely.geometry import shape
from shapely.ops import transform


CLOUD_COVER_LIMIT = 0.3
MIN_AOI_COVERAGE_PERCENT = 95.0
DEFAULT_LOOKBACK_YEARS = 12
EQUAL_AREA_CRS = "EPSG:6933"

# Bounding-box centroids can fall outside a city's civil timezone, especially
# for coastal or island AOIs. Explicit IANA zones make conversions auditable.
CITY_TIMEZONES = {
    "amsterdam": "Europe/Amsterdam",
    "barcelona": "Europe/Madrid",
    "birmingham": "Europe/London",
    "boston": "America/New_York",
    "buenos_aires": "America/Argentina/Buenos_Aires",
    "cape_town": "Africa/Johannesburg",
    "chicago": "America/Chicago",
    "copenhagen": "Europe/Copenhagen",
    "guadalajara": "America/Mexico_City",
    "helsinki": "Europe/Helsinki",
    "hong_kong": "Asia/Hong_Kong",
    "london": "Europe/London",
    "los_angeles": "America/Los_Angeles",
    "lyon": "Europe/Paris",
    "madrid": "Europe/Madrid",
    "manchester": "Europe/London",
    "marseille": "Europe/Paris",
    "montreal": "America/Toronto",
    "new_york_city": "America/New_York",
    "oslo": "Europe/Oslo",
    "paris": "Europe/Paris",
    "rotterdam": "Europe/Amsterdam",
    "san_francisco": "America/Los_Angeles",
    "sao_paulo": "America/Sao_Paulo",
    "seattle": "America/Los_Angeles",
    "utrecht": "Europe/Amsterdam",
    "valencia": "Europe/Madrid",
    "vancouver": "America/Vancouver",
    "zurich": "Europe/Zurich",
}

METADATA_FIELDS = [
    "cloud_cover",
    "clear_percent",
    "sun_elevation",
    "satellite_id",
    "instrument",
    "pixel_resolution",
    "shadow_percent",
    "snow_ice_percent",
    "heavy_haze_percent",
    "light_haze_percent",
    "quality_category",
]

OUTPUT_COLUMNS = [
    "city_slug",
    "city",
    "timezone",
    "id",
    "acquired",
    "acquired_local",
    *METADATA_FIELDS,
    "aoi_coverage_percent",
]


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(
        description=(
            "Search downloadable Planet PSScene metadata for current project "
            "cities using 5km city-specific AOI buffers. This command does not "
            "activate, order, or download assets."
        )
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
        help="Directory containing *_5km.geojson city AOI buffer files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root
        / "data_source"
        / "data"
        / "planet_imagery"
        / "generated"
        / "cities_scenes_results_planet.csv",
    )
    parser.add_argument(
        "--start-date",
        help=(
            "Inclusive UTC start date (YYYY-MM-DD). Defaults to exactly "
            f"{DEFAULT_LOOKBACK_YEARS} years before today."
        ),
    )
    parser.add_argument(
        "--end-date",
        help="Exclusive UTC end date (YYYY-MM-DD). Defaults to tomorrow UTC.",
    )
    parser.add_argument(
        "--city",
        action="append",
        dest="cities",
        help="Limit the run to a city slug. May be supplied more than once.",
    )
    parser.add_argument(
        "--max-results-per-window",
        type=int,
        default=0,
        help=(
            "Maximum results per city/year query. Zero retrieves every "
            "paginated result. Use a small value only for testing."
        ),
    )
    parser.add_argument(
        "--request-pause",
        type=float,
        default=0.25,
        help=(
            "Seconds to pause between annual search windows. Searches are "
            "sequential; the SDK additionally retries HTTP 429 responses."
        ),
    )
    parser.add_argument(
        "--max-window-retries",
        type=int,
        default=5,
        help="Retries for transient 429, 500, or 502 errors in one search window.",
    )
    parser.add_argument(
        "--retry-base-delay",
        type=float,
        default=5.0,
        help="Initial retry delay in seconds. Subsequent delays double.",
    )
    return parser.parse_args()


def slugify_city_name(value: str) -> str:
    return value.strip().lower().replace(" ", "_")


def current_city_slugs_from_readme(readme_path: Path) -> list[str]:
    """
    Read the active city list from the README Current Cities table.

    The README is the project-facing source of truth for the active sample.
    This parser intentionally reads only the table under `### Current Cities`
    so older city examples elsewhere in the README do not leak into the search.
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


def utc_midnight(value: str | None, default: datetime) -> datetime:
    if value is None:
        return default
    parsed = datetime.strptime(value, "%Y-%m-%d")
    return parsed.replace(tzinfo=timezone.utc)


def years_before(value: datetime, years: int) -> datetime:
    try:
        return value.replace(year=value.year - years)
    except ValueError:
        # February 29 becomes February 28 in a non-leap target year.
        return value.replace(year=value.year - years, day=28)


def annual_windows(start: datetime, end: datetime) -> Iterator[tuple[datetime, datetime]]:
    cursor = start
    while cursor < end:
        try:
            next_cursor = cursor.replace(year=cursor.year + 1)
        except ValueError:
            next_cursor = cursor.replace(year=cursor.year + 1, day=28)
        yield cursor, min(next_cursor, end)
        cursor = next_cursor


def load_aoi(path: Path) -> tuple[dict, str, str]:
    with path.open(encoding="utf-8") as handle:
        document = json.load(handle)

    if document["type"] == "FeatureCollection":
        features = document.get("features", [])
        if len(features) != 1:
            raise ValueError(f"Expected exactly one feature in {path}")
        feature = features[0]
    elif document["type"] == "Feature":
        feature = document
    else:
        feature = {"type": "Feature", "properties": {}, "geometry": document}

    properties = feature.get("properties") or {}
    city_slug = properties.get("city_slug") or path.name.removesuffix("_5km.geojson")
    city = (
        properties.get("city_name")
        or properties.get("city")
        or city_slug.replace("_", " ").title()
    )
    geometry = feature.get("geometry")
    if not geometry:
        raise ValueError(f"Missing geometry in {path}")
    return geometry, city_slug, city


def coverage_calculator(aoi_geojson: dict):
    transformer = Transformer.from_crs(
        "EPSG:4326", EQUAL_AREA_CRS, always_xy=True
    )
    project = transformer.transform
    aoi_equal_area = transform(project, shape(aoi_geojson))
    aoi_area = aoi_equal_area.area
    if aoi_area <= 0:
        raise ValueError("AOI has zero area")

    def calculate(scene_geojson: dict | None) -> float | None:
        if not scene_geojson:
            return None
        scene_equal_area = transform(project, shape(scene_geojson))
        coverage = aoi_equal_area.intersection(scene_equal_area).area / aoi_area
        return round(max(0.0, min(1.0, coverage)) * 100, 6)

    return calculate


def acquired_times(value: str, timezone_name: str) -> tuple[str, str]:
    acquired_utc = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if acquired_utc.tzinfo is None:
        acquired_utc = acquired_utc.replace(tzinfo=timezone.utc)
    acquired_utc = acquired_utc.astimezone(timezone.utc)
    acquired_local = acquired_utc.astimezone(ZoneInfo(timezone_name))
    return (
        acquired_utc.isoformat().replace("+00:00", "Z"),
        acquired_local.isoformat(),
    )


def item_to_row(
    item: dict,
    city_slug: str,
    city: str,
    timezone_name: str,
    calculate_coverage,
) -> dict:
    properties = item.get("properties") or {}
    acquired = properties.get("acquired")
    if not acquired:
        raise ValueError(f"Scene {item.get('id')} has no acquired timestamp")
    acquired_utc, acquired_local = acquired_times(acquired, timezone_name)

    row = {
        "city_slug": city_slug,
        "city": city,
        "timezone": timezone_name,
        "id": item.get("id"),
        "acquired": acquired_utc,
        "acquired_local": acquired_local,
        "aoi_coverage_percent": calculate_coverage(item.get("geometry")),
    }
    row.update({field: properties.get(field) for field in METADATA_FIELDS})
    return row


def write_checkpoint(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    if not frame.empty:
        frame = (
            frame.drop_duplicates(subset=["city_slug", "id"])
            .sort_values(["city_slug", "acquired", "id"])
            .reset_index(drop=True)
        )
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    frame.to_csv(temporary_path, index=False)
    temporary_path.replace(output_path)


def load_checkpoint(output_path: Path) -> list[dict]:
    if not output_path.exists():
        return []
    frame = pd.read_csv(output_path)
    missing = set(OUTPUT_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(
            f"Existing output has an incompatible schema; missing {sorted(missing)}"
        )
    return frame[OUTPUT_COLUMNS].where(pd.notna(frame), None).to_dict("records")


def search_window(
    planet: Planet,
    geometry: dict,
    search_filter: dict,
    city_slug: str,
    city: str,
    timezone_name: str,
    calculate_coverage,
    limit: int,
    max_retries: int,
    retry_base_delay: float,
) -> list[dict]:
    transient_errors = (TooManyRequests, ServerError, BadGateway)
    for attempt in range(max_retries + 1):
        try:
            items = planet.data.search(
                item_types=["PSScene"],
                geometry=geometry,
                search_filter=search_filter,
                sort="acquired asc",
                limit=limit,
            )
            # Keep attempt rows separate. If pagination fails, retry the whole
            # window without retaining an incomplete page sequence.
            rows = []
            for item in items:
                row = item_to_row(
                    item,
                    city_slug,
                    city,
                    timezone_name,
                    calculate_coverage,
                )
                coverage_percent = row["aoi_coverage_percent"]
                if (
                    coverage_percent is not None
                    and coverage_percent >= MIN_AOI_COVERAGE_PERCENT
                ):
                    rows.append(row)
            return rows
        except transient_errors as error:
            if attempt >= max_retries:
                raise
            delay = min(retry_base_delay * (2**attempt), 120.0)
            print(
                f"Transient Planet error ({type(error).__name__}). "
                f"Retry {attempt + 1}/{max_retries} in {delay:.1f}s.",
                flush=True,
            )
            time.sleep(delay)

    raise RuntimeError("Unreachable retry state")


def main() -> None:
    args = parse_args()
    api_key = os.environ.get("PL_API_KEY")

    now = datetime.now(timezone.utc)
    default_end = (
        now.replace(hour=0, minute=0, second=0, microsecond=0)
        + timedelta(days=1)
    )
    end = utc_midnight(args.end_date, default_end)
    start = utc_midnight(
        args.start_date, years_before(end, DEFAULT_LOOKBACK_YEARS)
    )
    if start >= end:
        raise SystemExit("start-date must be earlier than end-date")
    if args.max_results_per_window < 0:
        raise SystemExit("max-results-per-window cannot be negative")
    if args.max_window_retries < 0:
        raise SystemExit("max-window-retries cannot be negative")
    if args.retry_base_delay < 0:
        raise SystemExit("retry-base-delay cannot be negative")

    readme_path = Path(__file__).resolve().parents[3] / "README.md"
    current_city_slugs = current_city_slugs_from_readme(readme_path)
    aoi_files_by_slug = {
        path.name.removesuffix("_5km.geojson"): path
        for path in sorted(args.aoi_dir.glob("*_5km.geojson"))
    }

    if args.cities:
        requested = set(args.cities)
        unknown_requested = requested - set(current_city_slugs)
        if unknown_requested:
            raise SystemExit(
                f"City slug(s) not listed in README current cities: "
                f"{sorted(unknown_requested)}"
            )
        selected_city_slugs = [slug for slug in current_city_slugs if slug in requested]
    else:
        selected_city_slugs = current_city_slugs

    missing = [slug for slug in selected_city_slugs if slug not in aoi_files_by_slug]
    if missing:
        raise SystemExit(
            f"Missing 5km AOI file(s) in {args.aoi_dir}: "
            f"{[slug + '_5km.geojson' for slug in missing]}"
        )

    aoi_files = [aoi_files_by_slug[slug] for slug in selected_city_slugs]
    if args.cities:
        found = {path.name.removesuffix("_5km.geojson") for path in aoi_files}
        missing = requested - found
        if missing:
            raise SystemExit(f"Unknown city slug(s): {sorted(missing)}")
    if not aoi_files:
        raise SystemExit(f"No city 5km AOIs found in {args.aoi_dir}")

    print(
        f"Metadata-only Planet search: {len(aoi_files)} cities, "
        f"{start.date()} through {end.date()}, cloud_cover < "
        f"{CLOUD_COVER_LIMIT}, AOI coverage >= "
        f"{MIN_AOI_COVERAGE_PERCENT:.1f}%."
    )
    print("No asset activation, ordering, or downloading is performed.")

    if api_key:
        print("Authentication: PL_API_KEY environment variable.")
        session = Session(auth=Auth.from_key(api_key))
    else:
        print("Authentication: saved Planet OAuth profile.")
        session = Session(auth=Auth.from_user_default_session())
    planet = Planet(session=session)
    all_rows = load_checkpoint(args.output)
    if all_rows:
        print(
            f"Resuming with {len(all_rows):,} rows from {args.output}.",
            flush=True,
        )

    for city_number, aoi_path in enumerate(aoi_files, start=1):
        geometry, city_slug, city = load_aoi(aoi_path)
        timezone_name = CITY_TIMEZONES.get(city_slug)
        if timezone_name is None:
            raise KeyError(f"No IANA timezone configured for {city_slug}")
        calculate_coverage = coverage_calculator(geometry)
        city_rows: list[dict] = []

        for window_number, (window_start, window_end) in enumerate(
            annual_windows(start, end), start=1
        ):
            search_filter = data_filter.and_filter(
                [
                    data_filter.date_range_filter(
                        field_name="acquired",
                        gte=window_start,
                        lt=window_end,
                    ),
                    data_filter.range_filter(
                        field_name="cloud_cover",
                        lt=CLOUD_COVER_LIMIT,
                    ),
                    data_filter.permission_filter(),
                ]
            )
            print(
                f"[{city_number}/{len(aoi_files)}] {city_slug}: "
                f"{window_start.date()} to {window_end.date()}",
                flush=True,
            )
            try:
                window_rows = search_window(
                    planet=planet,
                    geometry=geometry,
                    search_filter=search_filter,
                    city_slug=city_slug,
                    city=city,
                    timezone_name=timezone_name,
                    calculate_coverage=calculate_coverage,
                    limit=args.max_results_per_window,
                    max_retries=args.max_window_retries,
                    retry_base_delay=args.retry_base_delay,
                )
                city_rows.extend(window_rows)
                write_checkpoint(all_rows + city_rows, args.output)
            except PlanetError as error:
                write_checkpoint(all_rows + city_rows, args.output)
                raise RuntimeError(
                    f"Planet search failed for {city_slug}, "
                    f"{window_start.date()} to {window_end.date()} after "
                    f"{args.max_window_retries} retries. Partial results were "
                    f"preserved in {args.output}."
                ) from error

            if args.request_pause > 0:
                time.sleep(args.request_pause)

        # A scene can appear in adjacent date windows but is retained once per
        # city. The output is atomically checkpointed after every window and city.
        unique_city_rows = {
            (row["city_slug"], row["id"]): row for row in city_rows
        }
        all_rows.extend(unique_city_rows.values())
        write_checkpoint(all_rows, args.output)
        print(
            f"{city_slug}: {len(unique_city_rows):,} unique scenes; "
            f"checkpointed {len(all_rows):,} rows.",
            flush=True,
        )

    write_checkpoint(all_rows, args.output)
    print(f"WROTE {args.output} ({len(all_rows):,} city-scene rows)")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted. The last completed-city checkpoint was preserved.")
        sys.exit(130)
