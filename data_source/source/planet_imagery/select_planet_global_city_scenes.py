"""
Select nine PlanetScope scenes per global WUP city for model training.

Environment: data_source/source/planet_imagery/venv_planet_imagery

Requires (inputs from earlier stages):
    - data_source/data/city_aois/generated/
      wup2018_cities_over_300k_2018.csv
    - data_source/data/planet_imagery/generated/
      global_city_scene_metadata/by_city/<city_slug>_planet_scenes.csv
    - Saved Planet OAuth authentication, unless --skip-asset-check is used
      for an explicitly unverified offline test.

Produces (outputs for later review and ordering):
    - data_source/data/planet_imagery/generated/global_scene_selection/
      selected_global_planet_city_scenes.csv
    - data_source/data/planet_imagery/generated/global_scene_selection/
      global_scene_selection_city_summary.csv
    - data_source/data/planet_imagery/generated/global_scene_selection/
      global_scene_selection_shortfalls.csv
    - data_source/data/planet_imagery/generated/global_scene_selection/
      planet_scene_asset_availability.csv
    - data_source/data/planet_imagery/generated/global_scene_selection/
      by_city/<city_slug>_selected_planet_scenes.csv
    - data_source/data/planet_imagery/generated/logs/
      select_planet_global_city_scenes_<UTC timestamp>.log

Description:
    Reads every city's previously queried Planet metadata and selects up to
    nine standard-quality solstice-window scenes. Northern Hemisphere summer
    is June-July and winter is December-January; these labels reverse in the
    Southern Hemisphere. The selector first maximizes distinct acquisition
    years, then fills shortfalls with repeated years while balancing two
    scenes per cardinal direction, a five/four summer-winter split, diverse
    sun elevations, high off-nadir view angles, and strict image quality.

    Planet asset metadata is queried only to require RGB+NIR surface
    reflectance. ortho_analytic_8b_sr is preferred and
    ortho_analytic_4b_sr is the documented fallback. Asset results are cached
    atomically and reused on rerun.

Safety:
    This script NEVER activates assets, creates orders, or downloads imagery.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import traceback
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
VENV_DIR = SCRIPT_DIR / "venv_planet_imagery"
VENV_PYTHON = VENV_DIR / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
VENV_MARKER = "PLANET_IMAGERY_VENV_ACTIVE"


def relaunch_inside_venv() -> None:
    """Relaunch with the task-specific environment before third-party imports."""
    if os.environ.get(VENV_MARKER) == "1" or Path(sys.prefix).absolute() == VENV_DIR.absolute():
        return
    if not VENV_PYTHON.exists():
        raise SystemExit(
            "ERROR: Missing Planet imagery environment. Recreate it from "
            "data_source/source/planet_imagery/requirements.txt."
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
from planet import Auth, Session
from planet.clients import DataClient


TARGET_SCENES = 9
TARGET_DIRECTION_COUNT = 2
TARGET_SEASON_COUNTS = {"summer": 5, "winter": 4}
DIRECTION_SLOT_ORDER = ["N", "S", "E", "W", "N", "S", "E", "W", "ANY"]
SEASON_SLOT_ORDER = [
    "summer", "winter", "summer", "winter", "summer",
    "winter", "summer", "winter", "summer",
]
PREFERRED_ASSET_TYPES = ["ortho_analytic_8b_sr", "ortho_analytic_4b_sr"]
NUMERIC_COLUMNS = [
    "aoi_coverage_percent", "cloud_cover", "sun_elevation", "view_angle",
    "scene_centroid_bearing_degrees", "aoi_centroid_latitude",
    "shadow_percent", "snow_ice_percent", "heavy_haze_percent",
    "light_haze_percent",
]
SELECTION_COLUMNS = [
    "selection_rank", "selection_hemisphere", "selection_local_season",
    "selection_calendar_year", "selection_cardinal_direction",
    "selection_direction_target", "selection_season_target",
    "selection_filter_tier", "selection_filter_tier_name",
    "selection_filter_relaxed", "selection_year_repeated",
    "selection_direction_target_met", "selection_season_target_met",
    "selection_sun_diversity_gain_degrees", "selection_view_angle",
    "available_asset_types", "selected_asset_type", "asset_is_8band",
    "asset_fallback_used", "asset_check_status", "selection_score_components",
]
ASSET_CACHE_COLUMNS = [
    "scene_id", "available_asset_types", "selected_asset_type",
    "asset_check_status", "asset_checked_utc", "asset_check_error",
]
SUMMARY_COLUMNS = [
    "wup_urbancode", "city_slug", "city_name", "country", "hemisphere",
    "input_scene_count", "standard_solstice_candidate_count",
    "asset_compatible_candidate_count", "selected_scene_count",
    "distinct_selected_years", "repeated_year_scene_count",
    "summer_scene_count", "winter_scene_count", "north_scene_count",
    "south_scene_count", "east_scene_count", "west_scene_count",
    "strict_scene_count", "relaxed_scene_count", "eight_band_scene_count",
    "four_band_scene_count", "minimum_selected_sun_difference_degrees",
    "selected_sun_elevation_range_degrees", "selection_complete",
    "shortfall_reason",
]


def parse_args() -> argparse.Namespace:
    """Define bounded and resumable global selection options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inventory", type=Path,
        default=PROJECT_ROOT / "data_source/data/city_aois/generated/wup2018_cities_over_300k_2018.csv",
    )
    parser.add_argument(
        "--metadata-dir", type=Path,
        default=PROJECT_ROOT / "data_source/data/planet_imagery/generated/global_city_scene_metadata/by_city",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "data_source/data/planet_imagery/generated/global_scene_selection",
    )
    parser.add_argument("--city-slug", action="append", dest="city_slugs")
    parser.add_argument("--city-offset", type=int, default=0)
    parser.add_argument(
        "--city-limit", type=int, default=25,
        help="Cities in this run; zero processes every remaining city.",
    )
    parser.add_argument(
        "--asset-candidates-per-city", type=int, default=60,
        help="Maximum metadata-ranked candidates whose asset lists are checked per city.",
    )
    parser.add_argument(
        "--asset-check-concurrency", type=int, default=4,
        help="Maximum concurrent Planet asset-list requests.",
    )
    parser.add_argument(
        "--skip-asset-check", action="store_true",
        help=(
            "Offline test only. Treat candidates as unverified 8-band scenes. "
            "Outputs are clearly flagged and must not be ordered."
        ),
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Recompute selected city outputs instead of resuming completed ones.",
    )
    return parser.parse_args()


def resolve_project_path(path: Path) -> Path:
    """Resolve relative paths from the repository and reject outside writes."""
    resolved = path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()
    if not resolved.is_relative_to(PROJECT_ROOT):
        raise ValueError(f"Path is outside the project repository: {resolved}")
    return resolved


def atomic_write_csv(frame: pd.DataFrame, path: Path, columns: list[str] | None = None) -> None:
    """Atomically replace a CSV so interruption cannot create a false success."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    output = frame.reindex(columns=columns) if columns is not None else frame
    output.to_csv(temporary, index=False)
    temporary.replace(path)


def load_inventory(path: Path) -> pd.DataFrame:
    """Read and validate the complete WUP city inventory."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing required WUP city inventory: {path}")
    frame = pd.read_csv(path, dtype={"wup_urbancode": str, "city_slug": str})
    required = {"wup_urbancode", "city_slug", "city_name", "country"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"City inventory is missing columns: {sorted(missing)}")
    if frame.empty or frame["city_slug"].duplicated().any():
        raise ValueError("City inventory is empty or contains duplicate city_slug values")
    return frame


def select_inventory_batch(inventory: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    """Select explicit cities or a stable offset batch."""
    if args.city_slugs:
        requested = set(args.city_slugs)
        known = set(inventory["city_slug"])
        if requested.difference(known):
            raise ValueError(f"Unknown city slugs: {sorted(requested.difference(known))}")
        return inventory[inventory["city_slug"].isin(requested)].copy()
    selected = inventory.iloc[args.city_offset:].copy()
    if args.city_limit:
        selected = selected.head(args.city_limit).copy()
    if selected.empty:
        raise ValueError("The requested city batch is empty")
    return selected


def load_city_metadata(path: Path, expected_city_slug: str) -> pd.DataFrame:
    """Read one large metadata CSV without discarding its raw JSON columns."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing city Planet metadata CSV: {path}")
    frame = pd.read_csv(path, low_memory=False)
    required = {
        "wup_urbancode", "city_slug", "city_name", "country", "scene_id",
        "acquired", "cloud_cover", "quality_category", "sun_elevation",
        "view_angle", "aoi_coverage_percent", "aoi_centroid_latitude",
        "scene_centroid_bearing_degrees",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {sorted(missing)}")
    if frame.empty:
        return frame
    slugs = set(frame["city_slug"].dropna().astype(str))
    if slugs != {expected_city_slug}:
        raise ValueError(f"{path.name} contains unexpected city slugs: {sorted(slugs)}")
    if frame["scene_id"].astype(str).duplicated().any():
        raise ValueError(f"{path.name} contains duplicate scene_id values")
    frame = frame.copy()
    for column in NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    essential_numeric = [
        "aoi_coverage_percent", "cloud_cover", "sun_elevation", "view_angle",
        "aoi_centroid_latitude", "scene_centroid_bearing_degrees",
    ]
    if frame[essential_numeric].isna().any().any():
        bad = frame[essential_numeric].isna().sum()
        raise ValueError(
            f"{path.name} has missing/non-numeric selection metadata: "
            f"{bad[bad > 0].to_dict()}"
        )
    frame["acquired_dt"] = pd.to_datetime(frame["acquired"], utc=True, errors="raise")
    frame["selection_calendar_year"] = frame["acquired_dt"].dt.year.astype(int)
    frame["acquired_month"] = frame["acquired_dt"].dt.month.astype(int)
    return frame


def hemisphere_and_season(latitude: float, month: int) -> tuple[str, str | None]:
    """Assign local solstice season using the AOI's hemisphere."""
    hemisphere = "northern" if latitude >= 0 else "southern"
    if month in {6, 7}:
        return hemisphere, "summer" if hemisphere == "northern" else "winter"
    if month in {12, 1}:
        return hemisphere, "winter" if hemisphere == "northern" else "summer"
    return hemisphere, None


def cardinal_direction(bearing: float) -> str:
    """Map a clockwise bearing to four documented 90-degree sectors."""
    bearing = bearing % 360.0
    if bearing >= 315.0 or bearing < 45.0:
        return "N"
    if bearing < 135.0:
        return "E"
    if bearing < 225.0:
        return "S"
    return "W"


def filter_tier(coverage: float, cloud: float) -> tuple[int, str]:
    """Apply the approved strict-to-relaxed coverage/cloud hierarchy."""
    if coverage >= 99.999999 and cloud <= 1e-12:
        return 0, "100pct_coverage_0pct_cloud"
    if coverage >= 99.5 and cloud <= 1e-12:
        return 1, "at_least_99_5pct_coverage_0pct_cloud"
    if coverage >= 99.5 and cloud <= 0.01:
        return 2, "at_least_99_5pct_coverage_at_most_1pct_cloud"
    return 3, "at_least_95pct_coverage_lowest_available_cloud"


def prepare_candidates(frame: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Keep standard solstice scenes and derive all auditable selection fields."""
    if frame.empty:
        return frame.copy(), "unknown"
    latitude_values = frame["aoi_centroid_latitude"].round(8).dropna().unique()
    if len(latitude_values) != 1:
        raise ValueError(f"City metadata has inconsistent AOI latitudes: {latitude_values.tolist()}")
    latitude = float(latitude_values[0])
    hemisphere = "northern" if latitude >= 0 else "southern"
    candidates = frame[frame["quality_category"].astype(str).str.lower() == "standard"].copy()
    season_values = candidates["acquired_month"].map(
        lambda month: hemisphere_and_season(latitude, int(month))[1]
    )
    candidates["selection_local_season"] = season_values
    candidates = candidates[candidates["selection_local_season"].notna()].copy()
    candidates["selection_hemisphere"] = hemisphere
    candidates["selection_cardinal_direction"] = candidates[
        "scene_centroid_bearing_degrees"
    ].map(cardinal_direction)
    tiers = candidates.apply(
        lambda row: filter_tier(float(row["aoi_coverage_percent"]), float(row["cloud_cover"])),
        axis=1,
    )
    candidates["selection_filter_tier"] = [value[0] for value in tiers]
    candidates["selection_filter_tier_name"] = [value[1] for value in tiers]
    candidates["selection_filter_relaxed"] = candidates["selection_filter_tier"] > 0
    candidates["haze_shadow_snow_penalty"] = candidates[
        ["shadow_percent", "snow_ice_percent", "heavy_haze_percent", "light_haze_percent"]
    ].fillna(0.0).sum(axis=1)
    return candidates, hemisphere


def metadata_shortlist(candidates: pd.DataFrame, limit: int) -> pd.DataFrame:
    """Retain broad year/season/direction coverage before asset API checks."""
    if len(candidates) <= limit:
        return candidates.copy()
    ranked = candidates.sort_values(
        [
            "selection_filter_tier", "cloud_cover", "aoi_coverage_percent",
            "view_angle", "haze_shadow_snow_penalty", "scene_id",
        ],
        ascending=[True, True, False, False, True, True],
    )
    protected_groups = ranked.groupby(
        ["selection_calendar_year", "selection_local_season", "selection_cardinal_direction"],
        sort=True,
        group_keys=False,
    ).head(2)
    # Also protect rare seasonal and directional candidates. Without these
    # guards, a metadata-rich winter can consume the shortlist before asset
    # checks and incorrectly make a feasible summer/direction target appear
    # impossible.
    protected_seasons = ranked.groupby(
        "selection_local_season", sort=True, group_keys=False
    ).head(max(TARGET_SEASON_COUNTS.values()))
    protected_directions = ranked.groupby(
        "selection_cardinal_direction", sort=True, group_keys=False
    ).head(TARGET_DIRECTION_COUNT)
    protected = pd.concat(
        [protected_groups, protected_seasons, protected_directions],
        ignore_index=True,
    ).drop_duplicates("scene_id")
    remaining = ranked[~ranked["scene_id"].isin(protected["scene_id"])]
    result = pd.concat([protected, remaining], ignore_index=True)
    return result.drop_duplicates("scene_id").head(limit).copy()


def load_asset_cache(path: Path) -> pd.DataFrame:
    """Load prior successful asset checks for resumable selection."""
    if not path.exists():
        return pd.DataFrame(columns=ASSET_CACHE_COLUMNS)
    frame = pd.read_csv(path, dtype=str).fillna("")
    missing = set(ASSET_CACHE_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"Asset cache is missing columns: {sorted(missing)}")
    return frame[ASSET_CACHE_COLUMNS].drop_duplicates("scene_id", keep="last")


def choose_surface_reflectance_asset(assets: set[str]) -> str:
    """Prefer 8-band SR and use 4-band SR only as the approved fallback."""
    for asset_type in PREFERRED_ASSET_TYPES:
        if asset_type in assets:
            return asset_type
    return ""


async def check_candidate_assets(
    candidates: pd.DataFrame,
    cache: pd.DataFrame,
    cache_path: Path,
    concurrency: int,
    skip_asset_check: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Check or reuse Planet asset lists without activating any asset."""
    if candidates.empty:
        return candidates.copy(), cache
    cache_by_id = {
        str(row["scene_id"]): row.to_dict()
        for _, row in cache[cache["asset_check_status"] == "success"].iterrows()
    }
    needed = [scene_id for scene_id in candidates["scene_id"].astype(str) if scene_id not in cache_by_id]

    if skip_asset_check:
        checked_on = datetime.now(timezone.utc).isoformat()
        for scene_id in needed:
            cache_by_id[scene_id] = {
                "scene_id": scene_id,
                "available_asset_types": "UNVERIFIED_OFFLINE_TEST",
                "selected_asset_type": "ortho_analytic_8b_sr",
                "asset_check_status": "unverified_offline_test",
                "asset_checked_utc": checked_on,
                "asset_check_error": "Asset API check explicitly skipped; do not order this output.",
            }
    elif needed:
        semaphore = asyncio.Semaphore(concurrency)
        auth = Auth.from_user_default_session()

        async with Session(auth=auth) as session:
            client = DataClient(session)

            async def check_one(scene_id: str) -> dict[str, str]:
                async with semaphore:
                    try:
                        assets = await client.list_item_assets("PSScene", scene_id)
                        names = set(assets)
                        selected = choose_surface_reflectance_asset(names)
                        return {
                            "scene_id": scene_id,
                            "available_asset_types": ",".join(sorted(names)),
                            "selected_asset_type": selected,
                            "asset_check_status": "success",
                            "asset_checked_utc": datetime.now(timezone.utc).isoformat(),
                            "asset_check_error": "",
                        }
                    except Exception as error:
                        return {
                            "scene_id": scene_id,
                            "available_asset_types": "",
                            "selected_asset_type": "",
                            "asset_check_status": "failed",
                            "asset_checked_utc": datetime.now(timezone.utc).isoformat(),
                            "asset_check_error": f"{type(error).__name__}: {error}",
                        }

            results = await asyncio.gather(*(check_one(scene_id) for scene_id in needed))
        failed = [row for row in results if row["asset_check_status"] == "failed"]
        updated = pd.concat([cache, pd.DataFrame(results)], ignore_index=True)
        updated = updated.drop_duplicates("scene_id", keep="last")
        atomic_write_csv(updated, cache_path, ASSET_CACHE_COLUMNS)
        if failed:
            examples = [f"{row['scene_id']}: {row['asset_check_error']}" for row in failed[:5]]
            raise RuntimeError(
                f"Planet asset checks failed for {len(failed)} scenes. "
                f"No partial city selection will be accepted. Examples: {examples}"
            )
        cache = updated
        cache_by_id.update({row["scene_id"]: row for row in results})

    if skip_asset_check:
        updated = pd.concat([cache, pd.DataFrame(cache_by_id.values())], ignore_index=True)
        cache = updated.drop_duplicates("scene_id", keep="last")

    asset_table = pd.DataFrame([cache_by_id[str(scene_id)] for scene_id in candidates["scene_id"]])
    merged = candidates.merge(asset_table, on="scene_id", how="left", validate="one_to_one")
    compatible = merged[merged["selected_asset_type"].astype(str).isin(PREFERRED_ASSET_TYPES)].copy()
    compatible["asset_is_8band"] = compatible["selected_asset_type"] == "ortho_analytic_8b_sr"
    compatible["asset_fallback_used"] = compatible["selected_asset_type"] == "ortho_analytic_4b_sr"
    return compatible, cache


def sun_diversity_gain(candidate: pd.Series, selected: list[pd.Series]) -> float:
    """Measure the new scene's minimum sun-elevation separation."""
    if not selected:
        return 90.0
    value = float(candidate["sun_elevation"])
    return min(abs(value - float(row["sun_elevation"])) for row in selected)


def candidate_key(
    candidate: pd.Series,
    selected: list[pd.Series],
    selected_years: set[int],
    direction_counts: dict[str, int],
    season_counts: dict[str, int],
    direction_target: str,
    season_target: str,
    require_new_year: bool,
) -> tuple[Any, ...]:
    """Return a deterministic lexicographic score for one selection slot."""
    year = int(candidate["selection_calendar_year"])
    direction = str(candidate["selection_cardinal_direction"])
    season = str(candidate["selection_local_season"])
    new_year = year not in selected_years
    direction_needed = direction_counts.get(direction, 0) < TARGET_DIRECTION_COUNT
    direction_match = direction_target != "ANY" and direction == direction_target
    season_needed = season_counts.get(season, 0) < TARGET_SEASON_COUNTS[season]
    season_match = season == season_target
    diversity = sun_diversity_gain(candidate, selected)
    # Seasonal balance precedes direction balance. This prevents a city with
    # geographically one-sided summer coverage from receiving nine winter
    # scenes merely because winter scenes better fill cardinal quotas.
    season_priority = 0 if season_match and season_needed else (1 if season_needed else 2)
    if direction_target == "ANY":
        direction_priority = 0 if direction_needed else 1
    else:
        direction_priority = 0 if direction_match and direction_needed else (1 if direction_needed else 2)
    return (
        0 if (new_year or not require_new_year) else 1,
        season_priority,
        direction_priority,
        int(candidate["selection_filter_tier"]),
        -round(diversity, 8),
        0 if bool(candidate["asset_is_8band"]) else 1,
        -abs(float(candidate["view_angle"])),
        float(candidate["cloud_cover"]),
        -float(candidate["aoi_coverage_percent"]),
        float(candidate["haze_shadow_snow_penalty"]),
        str(candidate["scene_id"]),
    )


def select_scenes(candidates: pd.DataFrame) -> pd.DataFrame:
    """Greedily satisfy distinct-year, direction, season, and diversity goals."""
    if candidates.empty:
        return candidates.copy()
    selected: list[pd.Series] = []
    remaining = candidates.copy()
    selected_years: set[int] = set()
    direction_counts = {direction: 0 for direction in ["N", "S", "E", "W"]}
    season_counts = {season: 0 for season in ["summer", "winter"]}
    available_years = set(remaining["selection_calendar_year"].astype(int))

    for rank in range(1, TARGET_SCENES + 1):
        if remaining.empty:
            break
        direction_target = DIRECTION_SLOT_ORDER[rank - 1]
        season_target = SEASON_SLOT_ORDER[rank - 1]
        unselected_years_exist = bool(available_years.difference(selected_years))
        if unselected_years_exist:
            pool = remaining[
                ~remaining["selection_calendar_year"].astype(int).isin(selected_years)
            ].copy()
        else:
            pool = remaining.copy()
        if pool.empty:
            pool = remaining.copy()
            unselected_years_exist = False

        keyed = []
        for index, candidate in pool.iterrows():
            key = candidate_key(
                candidate, selected, selected_years, direction_counts,
                season_counts, direction_target, season_target,
                require_new_year=unselected_years_exist,
            )
            keyed.append((key, index))
        _, best_index = min(keyed, key=lambda value: value[0])
        best = remaining.loc[best_index].copy()
        year = int(best["selection_calendar_year"])
        direction = str(best["selection_cardinal_direction"])
        season = str(best["selection_local_season"])
        diversity_gain = sun_diversity_gain(best, selected)
        best["selection_rank"] = rank
        best["selection_direction_target"] = direction_target
        best["selection_season_target"] = season_target
        best["selection_year_repeated"] = year in selected_years
        best["selection_direction_target_met"] = direction_target == "ANY" or direction == direction_target
        best["selection_season_target_met"] = season == season_target
        best["selection_sun_diversity_gain_degrees"] = round(diversity_gain, 6)
        best["selection_view_angle"] = float(best["view_angle"])
        best["selection_score_components"] = json.dumps(
            {
                "new_year": year not in selected_years,
                "direction_target": direction_target,
                "direction": direction,
                "season_target": season_target,
                "season": season,
                "filter_tier": int(best["selection_filter_tier"]),
                "sun_diversity_gain_degrees": round(diversity_gain, 6),
                "asset_rank": 0 if bool(best["asset_is_8band"]) else 1,
                "absolute_view_angle": abs(float(best["view_angle"])),
            },
            sort_keys=True,
        )
        selected.append(best)
        selected_years.add(year)
        direction_counts[direction] += 1
        season_counts[season] += 1
        remaining = remaining.drop(index=best_index)

    return pd.DataFrame(selected)


def minimum_pairwise_difference(values: list[float]) -> float | None:
    """Return the smallest pairwise difference or missing for fewer than two."""
    if len(values) < 2:
        return None
    ordered = sorted(values)
    return min(b - a for a, b in zip(ordered, ordered[1:]))


def summarize_city(
    city_row: pd.Series,
    hemisphere: str,
    input_count: int,
    candidate_count: int,
    compatible_count: int,
    selected: pd.DataFrame,
) -> dict[str, Any]:
    """Build one auditable city-level result row."""
    directions = selected.get("selection_cardinal_direction", pd.Series(dtype=str)).value_counts()
    seasons = selected.get("selection_local_season", pd.Series(dtype=str)).value_counts()
    years = selected.get("selection_calendar_year", pd.Series(dtype=int))
    suns = selected.get("sun_elevation", pd.Series(dtype=float)).astype(float).tolist()
    selected_count = len(selected)
    reasons = []
    if selected_count < TARGET_SCENES:
        reasons.append(f"only_{selected_count}_asset_compatible_standard_solstice_scenes_selected")
    if years.nunique() < min(TARGET_SCENES, candidate_count):
        reasons.append("insufficient_distinct_years_repeated_years_used")
    for direction in ["N", "S", "E", "W"]:
        if int(directions.get(direction, 0)) < TARGET_DIRECTION_COUNT:
            reasons.append(f"direction_{direction.lower()}_below_target")
    if int(seasons.get("summer", 0)) < 5 or int(seasons.get("winter", 0)) < 4:
        reasons.append("five_four_season_balance_not_met")
    if not selected.empty and bool(selected["asset_check_status"].eq("unverified_offline_test").any()):
        reasons.append("asset_availability_unverified_offline_test")
    return {
        "wup_urbancode": city_row["wup_urbancode"],
        "city_slug": city_row["city_slug"],
        "city_name": city_row["city_name"],
        "country": city_row["country"],
        "hemisphere": hemisphere,
        "input_scene_count": input_count,
        "standard_solstice_candidate_count": candidate_count,
        "asset_compatible_candidate_count": compatible_count,
        "selected_scene_count": selected_count,
        "distinct_selected_years": int(years.nunique()) if selected_count else 0,
        "repeated_year_scene_count": int(selected.get("selection_year_repeated", pd.Series(dtype=bool)).sum()),
        "summer_scene_count": int(seasons.get("summer", 0)),
        "winter_scene_count": int(seasons.get("winter", 0)),
        "north_scene_count": int(directions.get("N", 0)),
        "south_scene_count": int(directions.get("S", 0)),
        "east_scene_count": int(directions.get("E", 0)),
        "west_scene_count": int(directions.get("W", 0)),
        "strict_scene_count": int((selected.get("selection_filter_tier", pd.Series(dtype=int)) == 0).sum()),
        "relaxed_scene_count": int((selected.get("selection_filter_tier", pd.Series(dtype=int)) > 0).sum()),
        "eight_band_scene_count": int(selected.get("asset_is_8band", pd.Series(dtype=bool)).sum()),
        "four_band_scene_count": int(selected.get("asset_fallback_used", pd.Series(dtype=bool)).sum()),
        "minimum_selected_sun_difference_degrees": minimum_pairwise_difference(suns),
        "selected_sun_elevation_range_degrees": max(suns) - min(suns) if suns else None,
        "selection_complete": selected_count == TARGET_SCENES,
        "shortfall_reason": ";".join(dict.fromkeys(reasons)),
    }


def rebuild_combined_outputs(output_dir: Path) -> None:
    """Rebuild combined selection and summary tables from completed city files."""
    city_files = sorted((output_dir / "by_city").glob("*_selected_planet_scenes.csv"))
    selection_frames = [pd.read_csv(path, low_memory=False) for path in city_files]
    selection_frames = [frame for frame in selection_frames if not frame.empty]
    combined = (
        pd.concat(selection_frames, ignore_index=True)
        if selection_frames
        else pd.DataFrame(columns=SELECTION_COLUMNS)
    )
    if not combined.empty:
        combined = combined.sort_values(["city_slug", "selection_rank", "scene_id"])
    atomic_write_csv(combined, output_dir / "selected_global_planet_city_scenes.csv")

    summary_files = sorted((output_dir / "by_city_summary").glob("*_selection_summary.csv"))
    summaries = [pd.read_csv(path) for path in summary_files]
    summary = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame(columns=SUMMARY_COLUMNS)
    if not summary.empty:
        summary = summary.sort_values("city_slug")
    atomic_write_csv(summary, output_dir / "global_scene_selection_city_summary.csv", SUMMARY_COLUMNS)
    shortfalls = summary[
        (summary["selection_complete"].astype(str).str.lower() != "true")
        | (summary["shortfall_reason"].fillna("").astype(str) != "")
    ].copy()
    atomic_write_csv(shortfalls, output_dir / "global_scene_selection_shortfalls.csv", SUMMARY_COLUMNS)


async def main() -> None:
    """Process one resumable city batch and rebuild cross-city outputs."""
    args = parse_args()
    started = datetime.now(timezone.utc)
    log_path = PROJECT_ROOT / "data_source/data/planet_imagery/generated/logs" / f"select_planet_global_city_scenes_{started.strftime('%Y%m%dT%H%M%SZ')}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_lines = ["status=RUNNING", f"started_utc={started.isoformat()}"]
    try:
        args.inventory = resolve_project_path(args.inventory)
        args.metadata_dir = resolve_project_path(args.metadata_dir)
        args.output_dir = resolve_project_path(args.output_dir)
        if args.city_offset < 0 or args.city_limit < 0:
            raise ValueError("city-offset and city-limit cannot be negative")
        if args.asset_candidates_per_city < TARGET_SCENES:
            raise ValueError(f"asset-candidates-per-city must be at least {TARGET_SCENES}")
        if args.asset_check_concurrency < 1:
            raise ValueError("asset-check-concurrency must be at least one")

        inventory = load_inventory(args.inventory)
        selected_cities = select_inventory_batch(inventory, args)
        missing_metadata = [
            slug for slug in selected_cities["city_slug"]
            if not (args.metadata_dir / f"{slug}_planet_scenes.csv").is_file()
        ]
        if missing_metadata:
            raise FileNotFoundError(
                f"Missing metadata files for {len(missing_metadata)} selected cities: "
                f"{missing_metadata[:20]}"
            )

        args.output_dir.mkdir(parents=True, exist_ok=True)
        asset_cache_path = args.output_dir / "planet_scene_asset_availability.csv"
        asset_cache = load_asset_cache(asset_cache_path)
        completed = 0
        skipped = 0
        print(
            f"Selection-only run for {len(selected_cities)} cities. "
            "No assets will be activated, ordered, or downloaded.",
            flush=True,
        )

        for number, (_, city_row) in enumerate(selected_cities.iterrows(), start=1):
            city_slug = str(city_row["city_slug"])
            city_output = args.output_dir / "by_city" / f"{city_slug}_selected_planet_scenes.csv"
            summary_output = args.output_dir / "by_city_summary" / f"{city_slug}_selection_summary.csv"
            if city_output.exists() and summary_output.exists() and not args.overwrite:
                print(f"[{number}/{len(selected_cities)}] {city_slug}: skipped completed output", flush=True)
                skipped += 1
                continue
            print(f"[{number}/{len(selected_cities)}] {city_slug}: selecting scenes", flush=True)
            metadata = load_city_metadata(
                args.metadata_dir / f"{city_slug}_planet_scenes.csv", city_slug
            )
            candidates, hemisphere = prepare_candidates(metadata)
            if hemisphere == "unknown":
                inventory_latitude = pd.to_numeric(city_row.get("latitude"), errors="coerce")
                if pd.isna(inventory_latitude):
                    raise ValueError(
                        f"{city_slug} has no scene rows and no valid inventory latitude "
                        "for hemisphere assignment"
                    )
                hemisphere = "northern" if float(inventory_latitude) >= 0 else "southern"
            shortlist = metadata_shortlist(candidates, args.asset_candidates_per_city)
            compatible, asset_cache = await check_candidate_assets(
                shortlist, asset_cache, asset_cache_path,
                args.asset_check_concurrency, args.skip_asset_check,
            )
            if args.skip_asset_check:
                atomic_write_csv(asset_cache, asset_cache_path, ASSET_CACHE_COLUMNS)
            selected = select_scenes(compatible)
            output_columns = [
                column for column in metadata.columns
                if column not in {"acquired_dt", "acquired_month"}
            ] + [column for column in SELECTION_COLUMNS if column not in metadata.columns]
            atomic_write_csv(selected, city_output, output_columns)
            summary_row = summarize_city(
                city_row, hemisphere, len(metadata), len(candidates),
                len(compatible), selected,
            )
            atomic_write_csv(pd.DataFrame([summary_row]), summary_output, SUMMARY_COLUMNS)
            completed += 1
            print(
                f"  selected={len(selected)} distinct_years={summary_row['distinct_selected_years']} "
                f"shortfalls={summary_row['shortfall_reason'] or 'none'}",
                flush=True,
            )

        rebuild_combined_outputs(args.output_dir)
        log_lines[0] = "status=SUCCESS"
        log_lines.extend(
            [
                f"selected_city_batch_count={len(selected_cities)}",
                f"completed_city_count={completed}",
                f"skipped_city_count={skipped}",
                f"skip_asset_check={args.skip_asset_check}",
                f"output_dir={args.output_dir.relative_to(PROJECT_ROOT)}",
            ]
        )
        log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
        print(f"SUCCESS: selection batch completed. Log: {log_path}", flush=True)
    except BaseException:
        log_lines[0] = "status=FAILED"
        log_lines.append(traceback.format_exc())
        log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
        print(f"FAILED: see {log_path}", file=sys.stderr, flush=True)
        raise


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Interrupted: completed city outputs and asset checks remain resumable.", file=sys.stderr)
        sys.exit(130)
