#!/usr/bin/env python3
"""Select eight PlanetScope scenes total per training/open-LiDAR city.

The input population is the 94-city intersection between the Planet training
split and cities with ready open LiDAR.  This script reads already-discovered
Planet metadata and checks asset listings only.  It cannot activate assets,
create orders, or download imagery or LiDAR.

Temporal policy:
  * Prefer scenes from explicitly documented LiDAR acquisition years.
  * Fill remaining slots from the nearest available post-LiDAR years.
  * If necessary to reach eight, use clearly flagged pre-LiDAR solstice scenes.
  * Use clearly flagged standard-quality non-solstice scenes only as the final
    fallback. A shortfall means fewer than eight compatible scenes in total.
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys
import traceback
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
VENV_DIR = SCRIPT_DIR / "venv_planet_imagery"
VENV_PYTHON = VENV_DIR / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
VENV_MARKER = "PLANET_IMAGERY_VENV_ACTIVE"


def relaunch_inside_venv() -> None:
    """Relaunch this script inside the existing Planet imagery environment."""
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

# Reuse the tested parsing, quality tiers, season definitions, asset checks,
# cache format, and atomic writer from the global selector.  The venv marker
# above prevents that module's own relaunch guard from re-executing it.
import select_planet_global_city_scenes as global_selector


TARGET_SCENES_PER_YEAR = 8
TARGET_SCENES_PER_CITY = 8
TARGET_DIRECTION_COUNT = 2
TARGET_SEASON_COUNT = 4
DIRECTION_SLOT_ORDER = ["N", "S", "E", "W", "N", "S", "E", "W"]
SEASON_SLOT_ORDER = ["summer", "winter"] * 4
PLANET_METADATA_FIRST_YEAR = 2016

ADDED_SELECTION_COLUMNS = [
    "lidar_acquisition_years", "lidar_acquisition_start_date",
    "lidar_acquisition_end_date", "lidar_acquisition_date_precision",
    "lidar_acquisition_date_source", "lidar_target_year",
    "lidar_year_selection_rank", "lidar_temporal_rule",
]
SUMMARY_COLUMNS = [
    "wup_urbancode", "city_slug", "city_name", "country",
    "lidar_acquisition_years", "lidar_acquisition_date_precision",
    "lidar_target_year", "temporal_eligibility_status", "temporal_rule",
    "input_scene_count", "same_year_scene_count",
    "standard_solstice_candidate_count", "asset_compatible_candidate_count",
    "selected_scene_count", "summer_scene_count", "winter_scene_count",
    "north_scene_count", "south_scene_count", "east_scene_count",
    "west_scene_count", "strict_scene_count", "relaxed_scene_count",
    "eight_band_scene_count", "four_band_scene_count",
    "minimum_selected_sun_difference_degrees",
    "selected_sun_elevation_range_degrees", "selection_complete",
    "shortfall_reason",
]
ELIGIBILITY_COLUMNS = [
    "wup_urbancode", "city_slug", "city_name", "country",
    "lidar_acquisition_years", "lidar_acquisition_start_date",
    "lidar_acquisition_end_date", "lidar_acquisition_date_precision",
    "eligible_lidar_years", "excluded_lidar_years",
    "temporal_eligibility_status", "temporal_rule", "temporal_notes",
]
CITY_TOTAL_SUMMARY_COLUMNS = [
    "wup_urbancode", "city_slug", "city_name", "country",
    "lidar_acquisition_years", "lidar_acquisition_start_date",
    "lidar_acquisition_end_date", "lidar_acquisition_date_precision",
    "input_scene_count", "standard_scene_count",
    "standard_solstice_scene_count", "asset_compatible_candidate_count",
    "selected_scene_count", "acquisition_year_scene_count",
    "post_lidar_scene_count", "pre_lidar_fallback_scene_count",
    "non_solstice_fallback_scene_count", "summer_scene_count",
    "winter_scene_count", "other_month_scene_count", "north_scene_count",
    "south_scene_count", "east_scene_count", "west_scene_count",
    "strict_scene_count", "relaxed_scene_count", "eight_band_scene_count",
    "four_band_scene_count", "minimum_selected_sun_difference_degrees",
    "selected_sun_elevation_range_degrees", "selection_complete",
    "shortfall_reason",
]

# Only range types whose endpoints actually describe a bounded flight or
# project collection are expanded.  Broader coverage programmes are excluded.
ENDPOINT_RANGE_PRECISIONS = {
    "exact_official_project_range",
    "official_flight_lot_range",
}
BROAD_RANGE_PRECISIONS = {
    "official_campaign_range",
    "official_catalog_series_range",
    "official_national_campaign_range",
}


def parse_args() -> argparse.Namespace:
    """Define safe, resumable arguments for the 94-city selector."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training-lidar-cities", type=Path,
        default=PROJECT_ROOT / (
            "data_source/data/height_labels/generated/training_open_lidar/"
            "training_cities_with_open_lidar.csv"
        ),
    )
    parser.add_argument(
        "--metadata-dir", type=Path,
        default=PROJECT_ROOT / (
            "data_source/data/planet_imagery/generated/"
            "global_city_scene_metadata/by_city"
        ),
    )
    parser.add_argument(
        "--combined-metadata", type=Path,
        help=(
            "Optional single CSV containing all queried scenes for the 94 cities. "
            "When supplied, this replaces reads from --metadata-dir."
        ),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_ROOT / (
            "data_source/data/planet_imagery/generated/"
            "training_lidar_year_scene_selection"
        ),
    )
    parser.add_argument("--city-slug", action="append", dest="city_slugs")
    parser.add_argument("--city-offset", type=int, default=0)
    parser.add_argument(
        "--city-limit", type=int, default=10,
        help="Cities to process in this run; zero processes every remaining city.",
    )
    parser.add_argument(
        "--asset-candidates-per-city", "--asset-candidates-per-city-year",
        dest="asset_candidates_per_city", type=int, default=120,
        help=(
            "Maximum ranked candidates whose Planet asset lists are checked per city. "
            "The older --asset-candidates-per-city-year name remains an alias."
        ),
    )
    parser.add_argument("--asset-check-concurrency", type=int, default=4)
    parser.add_argument(
        "--skip-asset-check", action="store_true",
        help="Offline test only; outputs are marked unverified and must not be ordered.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_project_path(path: Path) -> Path:
    """Resolve relative paths from the repository and reject outside paths."""
    return global_selector.resolve_project_path(path)


def load_training_cities(path: Path) -> pd.DataFrame:
    """Read and strictly validate the expected 94-city enriched input."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing 94-city training LiDAR file: {path}")
    frame = pd.read_csv(path, dtype={"wup_urbancode": str, "city_slug": str}).fillna("")
    required = {
        "wup_urbancode", "city_slug", "city_name", "country",
        "lidar_acquisition_years", "lidar_acquisition_start_date",
        "lidar_acquisition_end_date", "lidar_acquisition_date_precision",
        "lidar_acquisition_date_source",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Training LiDAR file is missing columns: {sorted(missing)}")
    if len(frame) != 94 or frame["city_slug"].duplicated().any():
        raise ValueError(
            "Training LiDAR input must contain exactly 94 unique city_slug rows; "
            f"found {len(frame)} rows and {frame['city_slug'].nunique()} unique slugs"
        )
    return frame


def select_city_batch(frame: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    """Select requested cities or a stable bounded offset batch."""
    if args.city_slugs:
        requested = set(args.city_slugs)
        unknown = requested.difference(frame["city_slug"])
        if unknown:
            raise ValueError(f"Unknown 94-city LiDAR slugs: {sorted(unknown)}")
        return frame[frame["city_slug"].isin(requested)].copy()
    selected = frame.iloc[args.city_offset:].copy()
    if args.city_limit:
        selected = selected.head(args.city_limit).copy()
    if selected.empty:
        raise ValueError("The requested city batch is empty")
    return selected


def prepare_metadata_frame(
    frame: pd.DataFrame,
    expected_city_slug: str,
    source_label: str,
) -> pd.DataFrame:
    """Validate and normalize one city's rows from a combined metadata table."""
    required = {
        "wup_urbancode", "city_slug", "city_name", "country", "scene_id",
        "acquired", "cloud_cover", "quality_category", "sun_elevation",
        "view_angle", "aoi_coverage_percent", "aoi_centroid_latitude",
        "scene_centroid_bearing_degrees",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{source_label} is missing required columns: {sorted(missing)}")
    frame = frame.copy()
    if frame.empty:
        return frame
    slugs = set(frame["city_slug"].dropna().astype(str))
    if slugs != {expected_city_slug}:
        raise ValueError(f"{source_label} contains unexpected city slugs: {sorted(slugs)}")
    if frame["scene_id"].astype(str).duplicated().any():
        raise ValueError(f"{source_label} contains duplicate scene_id values")
    for column in global_selector.NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    essential_numeric = [
        "aoi_coverage_percent", "cloud_cover", "sun_elevation", "view_angle",
        "aoi_centroid_latitude", "scene_centroid_bearing_degrees",
    ]
    if frame[essential_numeric].isna().any().any():
        bad = frame[essential_numeric].isna().sum()
        raise ValueError(
            f"{source_label} has missing/non-numeric selection metadata: "
            f"{bad[bad > 0].to_dict()}"
        )
    frame["acquired_dt"] = pd.to_datetime(frame["acquired"], utc=True, errors="raise")
    frame["selection_calendar_year"] = frame["acquired_dt"].dt.year.astype(int)
    frame["acquired_month"] = frame["acquired_dt"].dt.month.astype(int)
    return frame


def load_combined_metadata(
    path: Path,
    expected_city_slugs: set[str],
) -> dict[str, pd.DataFrame]:
    """Load and validate one combined 94-city file once, grouped by city.

    The file is read once rather than rescanned for each city. This is faster
    on Windows and avoids creating another set of temporary per-city files.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Missing combined Planet metadata CSV: {path}")
    frame = pd.read_csv(path, low_memory=False, dtype={"city_slug": str, "scene_id": str})
    if frame.empty:
        raise ValueError(f"Combined Planet metadata CSV is empty: {path}")
    required = {"city_slug", "scene_id"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Combined metadata is missing columns: {sorted(missing)}")
    if frame[["city_slug", "scene_id"]].isna().any().any():
        raise ValueError("Combined metadata contains blank city_slug or scene_id values")
    if frame[["city_slug", "scene_id"]].duplicated().any():
        raise ValueError("Combined metadata contains duplicate city_slug/scene_id pairs")
    actual_city_slugs = set(frame["city_slug"].astype(str))
    missing_cities = sorted(expected_city_slugs.difference(actual_city_slugs))
    extra_cities = sorted(actual_city_slugs.difference(expected_city_slugs))
    if missing_cities or extra_cities:
        raise ValueError(
            "Combined metadata city set does not equal the 94-city input. "
            f"Missing={missing_cities[:20]}; extra={extra_cities[:20]}"
        )
    grouped = {}
    for city_slug, city_frame in frame.groupby("city_slug", sort=False):
        slug = str(city_slug)
        grouped[slug] = prepare_metadata_frame(
            city_frame, slug, f"{path.name}:{slug}"
        )
    return grouped


def interpret_lidar_years(row: pd.Series) -> dict[str, Any]:
    """Apply the approved exact-year, endpoint-range, and pre-2016 rules."""
    value = str(row["lidar_acquisition_years"]).strip()
    precision = str(row["lidar_acquisition_date_precision"]).strip()
    eligible: list[int] = []
    excluded: list[int] = []
    notes: list[str] = []

    if ";" in value:
        years = [int(part) for part in value.split(";") if re.fullmatch(r"\d{4}", part)]
        rule = "semicolon_separated_exact_years"
    elif re.fullmatch(r"\d{4}", value):
        years = [int(value)]
        rule = "single_exact_or_documented_year"
    elif re.fullmatch(r"\d{4}-\d{4}", value) and precision in ENDPOINT_RANGE_PRECISIONS:
        start, end = (int(part) for part in value.split("-"))
        years = sorted({start, end})
        rule = "documented_two_year_range_endpoints"
    elif re.fullmatch(r"\d{4}-\d{4}", value) and precision in BROAD_RANGE_PRECISIONS:
        years = []
        rule = "broad_range_not_expanded"
        notes.append("A broad campaign/catalogue range does not prove acquisition in each year.")
    else:
        years = []
        rule = "unresolved_acquisition_year_format"
        notes.append("Acquisition-year value could not be converted under the approved temporal rules.")

    for year in sorted(set(years)):
        if year >= PLANET_METADATA_FIRST_YEAR:
            eligible.append(year)
        else:
            excluded.append(year)
    if excluded:
        notes.append(
            "Pre-2016 LiDAR year(s) have no matching year in the existing Planet metadata period: "
            + ";".join(map(str, excluded))
        )
    if eligible:
        status = "eligible"
    elif rule == "broad_range_not_expanded":
        status = "requires_exact_lidar_tile_year"
    elif excluded:
        status = "no_same_year_planetscope_metadata"
    else:
        status = "requires_acquisition_year_review"
    return {
        "eligible_years": eligible,
        "excluded_years": excluded,
        "status": status,
        "rule": rule,
        "notes": " ".join(notes),
    }


def eligibility_row(city: pd.Series, interpretation: dict[str, Any]) -> dict[str, Any]:
    """Create one transparent temporal-policy audit row per input city."""
    return {
        "wup_urbancode": city["wup_urbancode"],
        "city_slug": city["city_slug"],
        "city_name": city["city_name"],
        "country": city["country"],
        "lidar_acquisition_years": city["lidar_acquisition_years"],
        "lidar_acquisition_start_date": city["lidar_acquisition_start_date"],
        "lidar_acquisition_end_date": city["lidar_acquisition_end_date"],
        "lidar_acquisition_date_precision": city["lidar_acquisition_date_precision"],
        "eligible_lidar_years": ";".join(map(str, interpretation["eligible_years"])),
        "excluded_lidar_years": ";".join(map(str, interpretation["excluded_years"])),
        "temporal_eligibility_status": interpretation["status"],
        "temporal_rule": interpretation["rule"],
        "temporal_notes": interpretation["notes"],
    }


def candidate_key(
    candidate: pd.Series,
    selected: list[pd.Series],
    direction_counts: dict[str, int],
    season_counts: dict[str, int],
    direction_target: str,
    season_target: str,
) -> tuple[Any, ...]:
    """Rank one same-year candidate using all approved global filters."""
    direction = str(candidate["selection_cardinal_direction"])
    season = str(candidate["selection_local_season"])
    direction_needed = direction_counts.get(direction, 0) < TARGET_DIRECTION_COUNT
    season_needed = season_counts.get(season, 0) < TARGET_SEASON_COUNT
    season_priority = 0 if season == season_target and season_needed else (1 if season_needed else 2)
    direction_priority = 0 if direction == direction_target and direction_needed else (1 if direction_needed else 2)
    diversity = global_selector.sun_diversity_gain(candidate, selected)
    return (
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


def select_same_year_scenes(candidates: pd.DataFrame, lidar_year: int, rule: str) -> pd.DataFrame:
    """Select up to eight scenes with 4/4 seasons and 2/2/2/2 directions."""
    selected: list[pd.Series] = []
    remaining = candidates.copy()
    direction_counts = {direction: 0 for direction in ["N", "S", "E", "W"]}
    season_counts = {season: 0 for season in ["summer", "winter"]}
    for rank in range(1, TARGET_SCENES_PER_YEAR + 1):
        if remaining.empty:
            break
        direction_target = DIRECTION_SLOT_ORDER[rank - 1]
        season_target = SEASON_SLOT_ORDER[rank - 1]
        keyed = [
            (
                candidate_key(
                    candidate, selected, direction_counts, season_counts,
                    direction_target, season_target,
                ),
                index,
            )
            for index, candidate in remaining.iterrows()
        ]
        _, best_index = min(keyed, key=lambda item: item[0])
        best = remaining.loc[best_index].copy()
        direction = str(best["selection_cardinal_direction"])
        season = str(best["selection_local_season"])
        diversity = global_selector.sun_diversity_gain(best, selected)
        best["selection_rank"] = rank
        best["lidar_year_selection_rank"] = rank
        best["lidar_target_year"] = lidar_year
        best["lidar_temporal_rule"] = rule
        best["selection_direction_target"] = direction_target
        best["selection_season_target"] = season_target
        best["selection_year_repeated"] = rank > 1
        best["selection_direction_target_met"] = direction == direction_target
        best["selection_season_target_met"] = season == season_target
        best["selection_sun_diversity_gain_degrees"] = round(diversity, 6)
        best["selection_view_angle"] = float(best["view_angle"])
        best["selection_score_components"] = json.dumps({
            "same_lidar_year_required": True,
            "lidar_target_year": lidar_year,
            "season_target": season_target,
            "season": season,
            "direction_target": direction_target,
            "direction": direction,
            "filter_tier": int(best["selection_filter_tier"]),
            "sun_diversity_gain_degrees": round(diversity, 6),
            "asset_rank": 0 if bool(best["asset_is_8band"]) else 1,
            "absolute_view_angle": abs(float(best["view_angle"])),
        }, sort_keys=True)
        selected.append(best)
        direction_counts[direction] += 1
        season_counts[season] += 1
        remaining = remaining.drop(index=best_index)
    return pd.DataFrame(selected)


def summary_row(
    city: pd.Series,
    lidar_year: int,
    interpretation: dict[str, Any],
    input_count: int,
    same_year_count: int,
    candidate_count: int,
    compatible_count: int,
    selected: pd.DataFrame,
) -> dict[str, Any]:
    """Summarize one city-year attempt and expose every unmet target."""
    directions = selected.get("selection_cardinal_direction", pd.Series(dtype=str)).value_counts()
    seasons = selected.get("selection_local_season", pd.Series(dtype=str)).value_counts()
    suns = selected.get("sun_elevation", pd.Series(dtype=float)).astype(float).tolist()
    reasons = []
    if len(selected) < TARGET_SCENES_PER_YEAR:
        reasons.append(f"only_{len(selected)}_eligible_same_year_scenes_selected")
    for direction in ["N", "S", "E", "W"]:
        if int(directions.get(direction, 0)) < TARGET_DIRECTION_COUNT:
            reasons.append(f"direction_{direction.lower()}_below_target")
    if int(seasons.get("summer", 0)) < 4 or int(seasons.get("winter", 0)) < 4:
        reasons.append("four_four_season_balance_not_met")
    if not selected.empty and bool(selected["asset_check_status"].eq("unverified_offline_test").any()):
        reasons.append("asset_availability_unverified_offline_test")
    return {
        "wup_urbancode": city["wup_urbancode"], "city_slug": city["city_slug"],
        "city_name": city["city_name"], "country": city["country"],
        "lidar_acquisition_years": city["lidar_acquisition_years"],
        "lidar_acquisition_date_precision": city["lidar_acquisition_date_precision"],
        "lidar_target_year": lidar_year,
        "temporal_eligibility_status": interpretation["status"],
        "temporal_rule": interpretation["rule"],
        "input_scene_count": input_count, "same_year_scene_count": same_year_count,
        "standard_solstice_candidate_count": candidate_count,
        "asset_compatible_candidate_count": compatible_count,
        "selected_scene_count": len(selected),
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
        "minimum_selected_sun_difference_degrees": global_selector.minimum_pairwise_difference(suns),
        "selected_sun_elevation_range_degrees": max(suns) - min(suns) if suns else None,
        "selection_complete": len(selected) == TARGET_SCENES_PER_YEAR,
        "shortfall_reason": ";".join(dict.fromkeys(reasons)),
    }


def rebuild_outputs(output_dir: Path) -> None:
    """Rebuild the combined selected-scene file and audits from completed files."""
    scene_files = sorted((output_dir / "by_city_year").glob("*_selected_planet_scenes.csv"))
    frames = [pd.read_csv(path, low_memory=False) for path in scene_files]
    frames = [frame for frame in frames if not frame.empty]
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not combined.empty:
        combined = combined.sort_values(["city_slug", "lidar_target_year", "lidar_year_selection_rank", "scene_id"])
        if combined[["city_slug", "lidar_target_year", "scene_id"]].duplicated().any():
            raise RuntimeError("Combined output contains duplicate city/year/scene rows")
    global_selector.atomic_write_csv(
        combined, output_dir / "selected_training_lidar_year_planet_scenes.csv"
    )

    summary_files = sorted((output_dir / "by_city_year_summary").glob("*_selection_summary.csv"))
    summaries = [pd.read_csv(path) for path in summary_files]
    summary = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame(columns=SUMMARY_COLUMNS)
    if not summary.empty:
        summary = summary.sort_values(["city_slug", "lidar_target_year"])
    global_selector.atomic_write_csv(
        summary, output_dir / "training_lidar_year_scene_selection_summary.csv", SUMMARY_COLUMNS
    )
    shortfalls = summary[
        (summary["selection_complete"].astype(str).str.lower() != "true")
        | summary["shortfall_reason"].fillna("").ne("")
    ].copy()
    global_selector.atomic_write_csv(
        shortfalls, output_dir / "training_lidar_year_scene_selection_shortfalls.csv", SUMMARY_COLUMNS
    )


def documented_acquisition_years(city: pd.Series) -> set[int]:
    """Return only explicitly documented years, without expanding broad ranges."""
    value = str(city["lidar_acquisition_years"]).strip()
    precision = str(city["lidar_acquisition_date_precision"]).strip()
    if ";" in value:
        return {int(part) for part in value.split(";") if re.fullmatch(r"\d{4}", part)}
    if re.fullmatch(r"\d{4}", value):
        return {int(value)}
    if re.fullmatch(r"\d{4}-\d{4}", value) and precision in ENDPOINT_RANGE_PRECISIONS:
        return {int(part) for part in value.split("-")}
    return set()


def prepare_city_total_candidates(metadata: pd.DataFrame, city: pd.Series) -> pd.DataFrame:
    """Prepare standard scenes and assign an auditable temporal fallback tier.

    Tier order is strict: solstice scenes in an exact acquisition year, then
    solstice scenes after LiDAR, then solstice scenes before LiDAR. Standard
    non-solstice scenes repeat that temporal order only as the last mechanism
    needed to reach eight total scenes.
    """
    if metadata.empty:
        return metadata.copy()
    latitude = float(metadata["aoi_centroid_latitude"].iloc[0])
    hemisphere = "northern" if latitude >= 0 else "southern"
    candidates = metadata[
        metadata["quality_category"].astype(str).str.lower().eq("standard")
    ].copy()
    candidates["selection_hemisphere"] = hemisphere
    candidates["selection_local_season"] = candidates["acquired_month"].map(
        lambda month: global_selector.hemisphere_and_season(latitude, int(month))[1] or "other"
    )
    candidates["selection_is_solstice"] = candidates["selection_local_season"].isin(
        ["summer", "winter"]
    )
    candidates["selection_cardinal_direction"] = candidates[
        "scene_centroid_bearing_degrees"
    ].map(global_selector.cardinal_direction)
    tiers = candidates.apply(
        lambda row: global_selector.filter_tier(
            float(row["aoi_coverage_percent"]), float(row["cloud_cover"])
        ),
        axis=1,
    )
    candidates["selection_filter_tier"] = [value[0] for value in tiers]
    candidates["selection_filter_tier_name"] = [value[1] for value in tiers]
    candidates["selection_filter_relaxed"] = candidates["selection_filter_tier"] > 0
    candidates["haze_shadow_snow_penalty"] = candidates[
        ["shadow_percent", "snow_ice_percent", "heavy_haze_percent", "light_haze_percent"]
    ].fillna(0.0).sum(axis=1)

    exact_years = documented_acquisition_years(city)
    lidar_end_year = int(str(city["lidar_acquisition_end_date"])[:4])

    def temporal_values(row: pd.Series) -> tuple[int, str, int]:
        year = int(row["selection_calendar_year"])
        if year in exact_years:
            relation_rank, relation = 0, "acquisition_year"
            distance = 0
        elif year > lidar_end_year:
            relation_rank, relation = 1, "post_lidar"
            distance = year - lidar_end_year
        else:
            relation_rank, relation = 2, "pre_lidar_fallback"
            distance = lidar_end_year - year
        # All solstice tiers precede all non-solstice tiers. This preserves the
        # original seasonal rule whenever at least eight such scenes exist.
        fallback_level = relation_rank + (0 if bool(row["selection_is_solstice"]) else 3)
        return fallback_level, relation, distance

    temporal = candidates.apply(temporal_values, axis=1)
    candidates["selection_temporal_fallback_level"] = [value[0] for value in temporal]
    candidates["selection_temporal_relation"] = [value[1] for value in temporal]
    candidates["selection_year_distance_from_lidar"] = [value[2] for value in temporal]
    return candidates


def city_total_shortlist(candidates: pd.DataFrame, limit: int) -> pd.DataFrame:
    """Keep enough candidates from every fallback tier before asset checks."""
    if len(candidates) <= limit:
        return candidates.copy()
    ranked = candidates.sort_values(
        [
            "selection_temporal_fallback_level", "selection_year_distance_from_lidar",
            "selection_filter_tier", "cloud_cover", "aoi_coverage_percent",
            "view_angle", "haze_shadow_snow_penalty", "scene_id",
        ],
        ascending=[True, True, True, True, False, False, True, True],
    )
    protected = ranked.groupby(
        ["selection_temporal_fallback_level", "selection_local_season", "selection_cardinal_direction"],
        sort=True, group_keys=False,
    ).head(2)
    remaining = ranked[~ranked["scene_id"].isin(protected["scene_id"])]
    return pd.concat([protected, remaining], ignore_index=True).drop_duplicates("scene_id").head(limit)


def city_total_candidate_key(
    candidate: pd.Series,
    selected: list[pd.Series],
    direction_counts: dict[str, int],
    season_counts: dict[str, int],
    direction_target: str,
    season_target: str,
) -> tuple[Any, ...]:
    """Rank a candidate while enforcing the approved temporal fallback order."""
    direction = str(candidate["selection_cardinal_direction"])
    season = str(candidate["selection_local_season"])
    direction_needed = direction_counts.get(direction, 0) < TARGET_DIRECTION_COUNT
    season_needed = season_counts.get(season, 0) < TARGET_SEASON_COUNT
    season_priority = 0 if season == season_target and season_needed else (1 if season_needed else 2)
    direction_priority = 0 if direction == direction_target and direction_needed else (1 if direction_needed else 2)
    diversity = global_selector.sun_diversity_gain(candidate, selected)
    return (
        int(candidate["selection_temporal_fallback_level"]),
        int(candidate["selection_year_distance_from_lidar"]),
        season_priority, direction_priority,
        int(candidate["selection_filter_tier"]), -round(diversity, 8),
        0 if bool(candidate["asset_is_8band"]) else 1,
        -abs(float(candidate["view_angle"])), float(candidate["cloud_cover"]),
        -float(candidate["aoi_coverage_percent"]),
        float(candidate["haze_shadow_snow_penalty"]), str(candidate["scene_id"]),
    )


def select_city_total_scenes(candidates: pd.DataFrame, city: pd.Series) -> pd.DataFrame:
    """Select exactly eight scenes total when eight compatible scenes exist."""
    selected: list[pd.Series] = []
    remaining = candidates.copy()
    direction_counts = {direction: 0 for direction in ["N", "S", "E", "W"]}
    season_counts = {season: 0 for season in ["summer", "winter"]}
    for rank in range(1, TARGET_SCENES_PER_CITY + 1):
        if remaining.empty:
            break
        direction_target = DIRECTION_SLOT_ORDER[rank - 1]
        season_target = SEASON_SLOT_ORDER[rank - 1]
        keyed = [
            (
                city_total_candidate_key(
                    candidate, selected, direction_counts, season_counts,
                    direction_target, season_target,
                ),
                index,
            )
            for index, candidate in remaining.iterrows()
        ]
        _, best_index = min(keyed, key=lambda item: item[0])
        best = remaining.loc[best_index].copy()
        direction = str(best["selection_cardinal_direction"])
        season = str(best["selection_local_season"])
        diversity = global_selector.sun_diversity_gain(best, selected)
        best["selection_rank"] = rank
        best["lidar_year_selection_rank"] = rank
        best["lidar_target_year"] = "CITY_TOTAL"
        best["lidar_temporal_rule"] = "acquisition_year_then_post_lidar_then_flagged_fallbacks"
        best["selection_direction_target"] = direction_target
        best["selection_season_target"] = season_target
        best["selection_year_repeated"] = any(
            int(row["selection_calendar_year"]) == int(best["selection_calendar_year"])
            for row in selected
        )
        best["selection_direction_target_met"] = direction == direction_target
        best["selection_season_target_met"] = season == season_target
        best["selection_sun_diversity_gain_degrees"] = round(diversity, 6)
        best["selection_view_angle"] = float(best["view_angle"])
        best["selection_score_components"] = json.dumps({
            "temporal_relation": best["selection_temporal_relation"],
            "temporal_fallback_level": int(best["selection_temporal_fallback_level"]),
            "year_distance_from_lidar": int(best["selection_year_distance_from_lidar"]),
            "solstice_window": bool(best["selection_is_solstice"]),
            "season_target": season_target, "season": season,
            "direction_target": direction_target, "direction": direction,
            "filter_tier": int(best["selection_filter_tier"]),
            "sun_diversity_gain_degrees": round(diversity, 6),
            "asset_rank": 0 if bool(best["asset_is_8band"]) else 1,
            "absolute_view_angle": abs(float(best["view_angle"])),
        }, sort_keys=True)
        selected.append(best)
        direction_counts[direction] += 1
        if season in season_counts:
            season_counts[season] += 1
        remaining = remaining.drop(index=best_index)
    selected_frame = pd.DataFrame(selected)
    for field in [
        "lidar_acquisition_years", "lidar_acquisition_start_date",
        "lidar_acquisition_end_date", "lidar_acquisition_date_precision",
        "lidar_acquisition_date_source",
    ]:
        selected_frame[field] = city[field]
    return selected_frame


def city_total_summary(
    city: pd.Series,
    metadata: pd.DataFrame,
    candidates: pd.DataFrame,
    compatible: pd.DataFrame,
    selected: pd.DataFrame,
) -> dict[str, Any]:
    """Create one city-level summary; shortfall now means fewer than eight."""
    directions = selected.get("selection_cardinal_direction", pd.Series(dtype=str)).value_counts()
    seasons = selected.get("selection_local_season", pd.Series(dtype=str)).value_counts()
    relations = selected.get("selection_temporal_relation", pd.Series(dtype=str)).value_counts()
    suns = selected.get("sun_elevation", pd.Series(dtype=float)).astype(float).tolist()
    shortfall = "" if len(selected) == 8 else f"only_{len(selected)}_compatible_scenes_selected"
    return {
        "wup_urbancode": city["wup_urbancode"], "city_slug": city["city_slug"],
        "city_name": city["city_name"], "country": city["country"],
        "lidar_acquisition_years": city["lidar_acquisition_years"],
        "lidar_acquisition_start_date": city["lidar_acquisition_start_date"],
        "lidar_acquisition_end_date": city["lidar_acquisition_end_date"],
        "lidar_acquisition_date_precision": city["lidar_acquisition_date_precision"],
        "input_scene_count": len(metadata), "standard_scene_count": len(candidates),
        "standard_solstice_scene_count": int(candidates["selection_is_solstice"].sum()),
        "asset_compatible_candidate_count": len(compatible),
        "selected_scene_count": len(selected),
        "acquisition_year_scene_count": int(relations.get("acquisition_year", 0)),
        "post_lidar_scene_count": int(relations.get("post_lidar", 0)),
        "pre_lidar_fallback_scene_count": int(relations.get("pre_lidar_fallback", 0)),
        "non_solstice_fallback_scene_count": int((~selected.get("selection_is_solstice", pd.Series(dtype=bool))).sum()),
        "summer_scene_count": int(seasons.get("summer", 0)),
        "winter_scene_count": int(seasons.get("winter", 0)),
        "other_month_scene_count": int(seasons.get("other", 0)),
        "north_scene_count": int(directions.get("N", 0)), "south_scene_count": int(directions.get("S", 0)),
        "east_scene_count": int(directions.get("E", 0)), "west_scene_count": int(directions.get("W", 0)),
        "strict_scene_count": int((selected.get("selection_filter_tier", pd.Series(dtype=int)) == 0).sum()),
        "relaxed_scene_count": int((selected.get("selection_filter_tier", pd.Series(dtype=int)) > 0).sum()),
        "eight_band_scene_count": int(selected.get("asset_is_8band", pd.Series(dtype=bool)).sum()),
        "four_band_scene_count": int(selected.get("asset_fallback_used", pd.Series(dtype=bool)).sum()),
        "minimum_selected_sun_difference_degrees": global_selector.minimum_pairwise_difference(suns),
        "selected_sun_elevation_range_degrees": max(suns) - min(suns) if suns else None,
        "selection_complete": len(selected) == 8, "shortfall_reason": shortfall,
    }


def rebuild_city_total_outputs(output_dir: Path) -> None:
    """Rebuild combined city-total selections, summaries, and true shortfalls."""
    scene_files = sorted((output_dir / "by_city").glob("*_selected_planet_scenes.csv"))
    frames = [pd.read_csv(path, low_memory=False) for path in scene_files]
    frames = [frame for frame in frames if not frame.empty]
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not combined.empty:
        combined = combined.sort_values(["city_slug", "selection_rank", "scene_id"])
        if combined[["city_slug", "scene_id"]].duplicated().any():
            raise RuntimeError("Combined city-total output contains duplicate city/scene pairs")
    global_selector.atomic_write_csv(
        combined, output_dir / "selected_training_lidar_city_planet_scenes.csv"
    )
    summary_files = sorted((output_dir / "by_city_summary").glob("*_selection_summary.csv"))
    summaries = [pd.read_csv(path) for path in summary_files]
    summary = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame(columns=CITY_TOTAL_SUMMARY_COLUMNS)
    if not summary.empty:
        summary = summary.sort_values("city_slug")
    global_selector.atomic_write_csv(
        summary, output_dir / "training_lidar_city_scene_selection_summary.csv",
        CITY_TOTAL_SUMMARY_COLUMNS,
    )
    shortfalls = summary[summary["selected_scene_count"].astype(int) < 8].copy()
    global_selector.atomic_write_csv(
        shortfalls, output_dir / "training_lidar_city_scene_selection_shortfalls.csv",
        CITY_TOTAL_SUMMARY_COLUMNS,
    )


async def main() -> None:
    """Process a bounded 94-city batch and rebuild combined outputs."""
    args = parse_args()
    started = datetime.now(timezone.utc)
    log_path = PROJECT_ROOT / (
        "data_source/data/planet_imagery/generated/logs/"
        f"select_planet_scenes_for_training_lidar_years_{started.strftime('%Y%m%dT%H%M%SZ')}.log"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_lines = ["status=RUNNING", f"started_utc={started.isoformat()}"]
    try:
        args.training_lidar_cities = resolve_project_path(args.training_lidar_cities)
        args.metadata_dir = resolve_project_path(args.metadata_dir)
        if args.combined_metadata is not None:
            args.combined_metadata = resolve_project_path(args.combined_metadata)
        args.output_dir = resolve_project_path(args.output_dir)
        if args.city_offset < 0 or args.city_limit < 0:
            raise ValueError("city-offset and city-limit cannot be negative")
        if args.asset_candidates_per_city < TARGET_SCENES_PER_CITY:
            raise ValueError("asset-candidates-per-city must be at least 8")
        if args.asset_check_concurrency < 1:
            raise ValueError("asset-check-concurrency must be at least one")

        all_cities = load_training_cities(args.training_lidar_cities)
        selected_cities = select_city_batch(all_cities, args)
        args.output_dir.mkdir(parents=True, exist_ok=True)

        combined_metadata_by_city = None
        if args.combined_metadata is not None:
            print(f"Loading combined metadata once: {args.combined_metadata}", flush=True)
            combined_metadata_by_city = load_combined_metadata(
                args.combined_metadata,
                set(all_cities["city_slug"].astype(str)),
            )
            print(
                f"Validated combined metadata: {len(combined_metadata_by_city)} cities, "
                f"{sum(len(frame) for frame in combined_metadata_by_city.values()):,} scene rows",
                flush=True,
            )

        # Always write the full 94-city temporal audit, even for a small test
        # batch, so excluded ranges and pre-2016 years remain visible.
        eligibility = []
        for _, city in all_cities.iterrows():
            eligibility.append(eligibility_row(city, interpret_lidar_years(city)))
        global_selector.atomic_write_csv(
            pd.DataFrame(eligibility),
            args.output_dir / "training_lidar_year_eligibility.csv",
            ELIGIBILITY_COLUMNS,
        )

        missing_metadata = [] if combined_metadata_by_city is not None else [
            str(city["city_slug"])
            for _, city in selected_cities.iterrows()
            if not (args.metadata_dir / f"{city['city_slug']}_planet_scenes.csv").is_file()
        ]
        if missing_metadata:
            raise FileNotFoundError(
                f"Missing Planet metadata for {len(missing_metadata)} eligible cities: "
                f"{missing_metadata[:20]}"
            )

        asset_cache_path = args.output_dir / "planet_scene_asset_availability.csv"
        asset_cache = global_selector.load_asset_cache(asset_cache_path)
        completed = skipped = 0
        print(
            f"Eight-total-scene selection run for {len(selected_cities)} of 94 cities. "
            "No assets will be activated, ordered, or downloaded.", flush=True,
        )
        for number, (_, city) in enumerate(selected_cities.iterrows(), start=1):
            slug = str(city["city_slug"])
            city_output = args.output_dir / "by_city" / f"{slug}_selected_planet_scenes.csv"
            summary_output = args.output_dir / "by_city_summary" / f"{slug}_selection_summary.csv"
            if city_output.exists() and summary_output.exists() and not args.overwrite:
                print(f"[{number}/{len(selected_cities)}] {slug}: skipped completed output", flush=True)
                skipped += 1
                continue
            if combined_metadata_by_city is not None:
                metadata = combined_metadata_by_city[slug].copy()
            else:
                metadata = global_selector.load_city_metadata(
                    args.metadata_dir / f"{slug}_planet_scenes.csv", slug
                )
            print(f"[{number}/{len(selected_cities)}] {slug}: selecting 8 total scenes", flush=True)
            candidates = prepare_city_total_candidates(metadata, city)
            shortlist = city_total_shortlist(candidates, args.asset_candidates_per_city)
            compatible, asset_cache = await global_selector.check_candidate_assets(
                shortlist, asset_cache, asset_cache_path,
                args.asset_check_concurrency, args.skip_asset_check,
            )
            if args.skip_asset_check:
                global_selector.atomic_write_csv(
                    asset_cache, asset_cache_path, global_selector.ASSET_CACHE_COLUMNS
                )
            selected = select_city_total_scenes(compatible, city)
            derived_columns = [
                "selection_is_solstice", "selection_temporal_fallback_level",
                "selection_temporal_relation", "selection_year_distance_from_lidar",
            ]
            output_columns = [
                column for column in metadata.columns
                if column not in {"acquired_dt", "acquired_month"}
            ] + [
                column for column in global_selector.SELECTION_COLUMNS + ADDED_SELECTION_COLUMNS + derived_columns
                if column not in metadata.columns
            ]
            global_selector.atomic_write_csv(selected, city_output, output_columns)
            summary = city_total_summary(city, metadata, candidates, compatible, selected)
            global_selector.atomic_write_csv(
                pd.DataFrame([summary]), summary_output, CITY_TOTAL_SUMMARY_COLUMNS
            )
            completed += 1
            print(
                f"  selected={len(selected)} post_lidar={summary['post_lidar_scene_count']} "
                f"pre_lidar_fallback={summary['pre_lidar_fallback_scene_count']} "
                f"non_solstice_fallback={summary['non_solstice_fallback_scene_count']} "
                f"shortfall={summary['shortfall_reason'] or 'none'}",
                flush=True,
            )

        rebuild_city_total_outputs(args.output_dir)
        log_lines[0] = "status=SUCCESS"
        log_lines.extend([
            f"selected_city_batch_count={len(selected_cities)}",
            f"completed_city_count={completed}",
            f"skipped_city_count={skipped}",
            f"skip_asset_check={args.skip_asset_check}",
            f"combined_metadata={args.combined_metadata or ''}",
            f"output_dir={args.output_dir.relative_to(PROJECT_ROOT)}",
        ])
        log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
        print(f"SUCCESS: eight-total-scene selection completed. Log: {log_path}", flush=True)
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
        print("Interrupted: completed city outputs remain resumable.", file=sys.stderr)
        sys.exit(130)
