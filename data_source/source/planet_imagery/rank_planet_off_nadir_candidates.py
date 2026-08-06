"""
Rank PlanetScope Scenes For Off-Nadir Experiments

Environment: data_source/source/planet_imagery/venv_planet_imagery

Requires:
    - data_source/data/city_aois/generated/city_buffers_5km_by_city/
      <city>_5km.geojson
    - Planet authentication through PLANET_API.py or a saved Planet profile

Produces:
    - data_source/data/planet_imagery/generated/
      <city>_off_nadir_candidate_pool.csv
    - data_source/data/planet_imagery/generated/
      <city>_off_nadir_top10_scenes.csv

Description:
    Searches Planet metadata for scenes that cover a fixed city AOI. It ranks
    clean, standard-quality scenes using the reported view angle together with
    the distance and direction between the scene center and AOI center. The
    command is metadata-only: it never orders, activates, or downloads assets.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[3]
VENV_DIR = SCRIPT_DIR / "venv_planet_imagery"
VENV_PYTHON = VENV_DIR / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
VENV_MARKER = "PLANET_IMAGERY_VENV_ACTIVE"
EQUAL_AREA_CRS = "EPSG:6933"


def relaunch_inside_venv() -> None:
    """Relaunch with the task-specific environment when necessary."""
    if os.environ.get(VENV_MARKER) == "1":
        return
    if Path(sys.prefix).absolute() == VENV_DIR.absolute():
        os.environ[VENV_MARKER] = "1"
        return
    if not VENV_PYTHON.exists():
        raise SystemExit(f"Missing Planet environment: {VENV_PYTHON}")
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
from planet.exceptions import BadGateway, ServerError, TooManyRequests
from pyproj import Transformer
from shapely.geometry import shape
from shapely.ops import transform


METADATA_FIELDS = [
    "cloud_cover",
    "clear_percent",
    "sun_elevation",
    "sun_azimuth",
    "satellite_azimuth",
    "view_angle",
    "quality_category",
    "shadow_percent",
    "snow_ice_percent",
    "heavy_haze_percent",
    "light_haze_percent",
    "satellite_id",
    "instrument",
    "pixel_resolution",
]


def parse_args() -> argparse.Namespace:
    """Define a reproducible one-city candidate search."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city", default="new_york_city")
    parser.add_argument("--start-date", default="2016-01-01")
    parser.add_argument(
        "--end-date",
        default=(datetime.now(timezone.utc) + timedelta(days=1)).date().isoformat(),
        help="Exclusive UTC date.",
    )
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--max-cloud-cover", type=float, default=0.0)
    parser.add_argument("--min-aoi-coverage", type=float, default=99.999)
    parser.add_argument("--min-boundary-clearance-km", type=float, default=0.5)
    parser.add_argument("--quality-category", default="standard")
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--retry-base-delay", type=float, default=5.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data_source/data/planet_imagery/generated",
    )
    return parser.parse_args()


def load_planet_api_key() -> str | None:
    """Read the ignored local credential without printing its contents."""
    credential_path = SCRIPT_DIR / "PLANET_API.py"
    if credential_path.exists():
        spec = importlib.util.spec_from_file_location("planet_credentials", credential_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not load credentials from {credential_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        value = getattr(module, "PL_API_KEY", None)
        if value:
            return str(value)
    return os.environ.get("PL_API_KEY")


def load_aoi(path: Path) -> dict:
    """Read the single AOI geometry stored in a GeoJSON document."""
    with path.open(encoding="utf-8") as handle:
        document = json.load(handle)
    if document["type"] == "FeatureCollection":
        features = document.get("features", [])
        if len(features) != 1:
            raise ValueError(f"Expected exactly one AOI feature in {path}")
        geometry = features[0].get("geometry")
    elif document["type"] == "Feature":
        geometry = document.get("geometry")
    else:
        geometry = document
    if not geometry:
        raise ValueError(f"No AOI geometry found in {path}")
    return geometry


def parse_utc_date(value: str) -> datetime:
    """Convert YYYY-MM-DD text into a timezone-aware UTC datetime."""
    return datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def cardinal_direction(bearing: float) -> str:
    """Convert a clockwise bearing into one of eight readable directions."""
    labels = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    return labels[int((bearing + 22.5) // 45) % 8]


def geometry_calculator(aoi_geojson: dict):
    """Prepare equal-area calculations shared by every candidate scene."""
    transformer = Transformer.from_crs("EPSG:4326", EQUAL_AREA_CRS, always_xy=True)
    project = transformer.transform
    aoi_wgs84 = shape(aoi_geojson)
    aoi_projected = transform(project, aoi_wgs84)
    if aoi_projected.area <= 0:
        raise ValueError("AOI has zero projected area")
    aoi_center = aoi_projected.centroid

    def calculate(scene_geojson: dict | None) -> dict:
        if not scene_geojson:
            raise ValueError("Planet item is missing its scene footprint geometry")
        scene_wgs84 = shape(scene_geojson)
        scene_projected = transform(project, scene_wgs84)
        scene_center = scene_projected.centroid
        dx = scene_center.x - aoi_center.x
        dy = scene_center.y - aoi_center.y
        offset_m = math.hypot(dx, dy)
        bearing = (math.degrees(math.atan2(dx, dy)) + 360.0) % 360.0
        coverage = (
            aoi_projected.intersection(scene_projected).area
            / aoi_projected.area
            * 100.0
        )
        clearance_m = 0.0
        if scene_projected.covers(aoi_projected):
            clearance_m = scene_projected.boundary.distance(aoi_projected.boundary)
        return {
            "aoi_coverage_percent": min(100.0, max(0.0, coverage)),
            "scene_center_offset_km": offset_m / 1000.0,
            "offset_bearing_degrees": bearing,
            "offset_direction": cardinal_direction(bearing),
            "aoi_boundary_clearance_km": clearance_m / 1000.0,
            "aoi_centroid_lon": aoi_wgs84.centroid.x,
            "aoi_centroid_lat": aoi_wgs84.centroid.y,
            "scene_centroid_lon": scene_wgs84.centroid.x,
            "scene_centroid_lat": scene_wgs84.centroid.y,
        }

    return calculate


def search_items(
    planet: Planet,
    aoi: dict,
    search_filter: dict,
    max_retries: int,
    retry_base_delay: float,
) -> list[dict]:
    """Retrieve the complete paginated item set, retrying transient failures."""
    transient_errors = (TooManyRequests, ServerError, BadGateway)
    for attempt in range(max_retries + 1):
        try:
            return list(
                planet.data.search(
                    item_types=["PSScene"],
                    geometry=aoi,
                    search_filter=search_filter,
                    sort="acquired asc",
                    limit=0,
                )
            )
        except transient_errors as error:
            if attempt >= max_retries:
                raise
            delay = min(retry_base_delay * (2**attempt), 120.0)
            print(
                f"Transient Planet error {type(error).__name__}; retrying in "
                f"{delay:.1f} seconds.",
                flush=True,
            )
            time.sleep(delay)
    raise RuntimeError("Unreachable retry state")


def normalized(series: pd.Series) -> pd.Series:
    """Min-max normalize a metric, returning zero if it has no variation."""
    values = pd.to_numeric(series, errors="coerce")
    minimum = values.min()
    maximum = values.max()
    if pd.isna(minimum) or pd.isna(maximum) or maximum <= minimum:
        return pd.Series(0.0, index=series.index)
    return (values - minimum) / (maximum - minimum)


def rank_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    """Combine direct viewing geometry with clean, safe AOI placement."""
    ranked = frame.copy()
    ranked["off_nadir_score"] = (
        0.60 * normalized(ranked["view_angle"])
        + 0.25 * normalized(ranked["scene_center_offset_km"])
        + 0.10 * normalized(ranked["clear_percent"].fillna(0.0))
        + 0.05 * normalized(ranked["aoi_boundary_clearance_km"])
    )
    ranked = ranked.sort_values(
        ["off_nadir_score", "view_angle", "scene_center_offset_km", "acquired"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    ranked.insert(0, "candidate_rank", range(1, len(ranked) + 1))
    return ranked


def main() -> None:
    """Query one city, validate strict filters, and save the ranked outputs."""
    args = parse_args()
    if args.top_n <= 0:
        raise SystemExit("--top-n must be positive")
    start = parse_utc_date(args.start_date)
    end = parse_utc_date(args.end_date)
    if start >= end:
        raise SystemExit("--start-date must be earlier than --end-date")

    aoi_path = (
        PROJECT_ROOT
        / "data_source/data/city_aois/generated/city_buffers_5km_by_city"
        / f"{args.city}_5km.geojson"
    )
    if not aoi_path.exists():
        raise SystemExit(f"Missing city AOI: {aoi_path}")
    aoi = load_aoi(aoi_path)
    calculate_geometry = geometry_calculator(aoi)

    api_key = load_planet_api_key()
    session = (
        Session(auth=Auth.from_key(api_key))
        if api_key
        else Session(auth=Auth.from_user_default_session())
    )
    planet = Planet(session=session)
    # The API narrows the large inventory by date, cloud, permission, and AOI
    # intersection. Exact quality and geometric rules are checked locally.
    search_filter = data_filter.and_filter(
        [
            data_filter.date_range_filter("acquired", gte=start, lt=end),
            data_filter.range_filter(
                "cloud_cover", lt=args.max_cloud_cover + 1e-9
            ),
            data_filter.permission_filter(),
        ]
    )
    print(
        f"Querying Planet metadata for {args.city}: {start.date()} to {end.date()}.",
        flush=True,
    )
    print("This command cannot order or download imagery.", flush=True)
    items = search_items(
        planet, aoi, search_filter, args.max_retries, args.retry_base_delay
    )
    print(f"Planet returned {len(items):,} intersecting clean scenes.", flush=True)

    rows = []
    for item in items:
        properties = item.get("properties") or {}
        row = {"city_slug": args.city, "id": item.get("id")}
        row["acquired"] = properties.get("acquired")
        row.update({field: properties.get(field) for field in METADATA_FIELDS})
        row.update(calculate_geometry(item.get("geometry")))
        rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("Planet returned no candidate scenes")

    numeric_columns = [
        "cloud_cover",
        "clear_percent",
        "view_angle",
        "aoi_coverage_percent",
        "aoi_boundary_clearance_km",
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    eligible = frame.loc[
        frame["view_angle"].notna()
        & (frame["cloud_cover"] <= args.max_cloud_cover + 1e-12)
        & (frame["aoi_coverage_percent"] >= args.min_aoi_coverage)
        & (frame["aoi_boundary_clearance_km"] >= args.min_boundary_clearance_km)
        & (frame["quality_category"] == args.quality_category)
    ].copy()
    if len(eligible) < args.top_n:
        raise RuntimeError(
            f"Only {len(eligible)} scenes passed strict filters; requested "
            f"{args.top_n}. Relax a documented threshold and rerun."
        )
    ranked = rank_candidates(eligible)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pool_path = args.output_dir / f"{args.city}_off_nadir_candidate_pool.csv"
    top_path = args.output_dir / f"{args.city}_off_nadir_top{args.top_n}_scenes.csv"
    ranked.to_csv(pool_path, index=False)
    ranked.head(args.top_n).to_csv(top_path, index=False)
    print(f"Eligible candidate pool: {len(ranked):,}")
    print(f"Saved full ranking: {pool_path.relative_to(PROJECT_ROOT)}")
    print(f"Saved top {args.top_n}: {top_path.relative_to(PROJECT_ROOT)}")
    print(
        ranked.head(args.top_n)[
            [
                "candidate_rank",
                "id",
                "acquired",
                "view_angle",
                "scene_center_offset_km",
                "offset_direction",
                "aoi_boundary_clearance_km",
                "off_nadir_score",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
