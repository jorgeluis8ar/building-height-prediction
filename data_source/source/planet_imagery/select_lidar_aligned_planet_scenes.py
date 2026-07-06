"""
Select LiDAR-Aligned Planet Scenes for NYC and LA

Environment: data_source/source/planet_imagery/venv_planet_imagery

Requires:
    - data_source/data/height_labels/generated/usgs_3dep_projects.csv
    - data_source/data/planet_imagery/generated/cities_scenes_results_planet.csv

Produces:
    - data_source/data/planet_imagery/generated/selected_lidar_aligned_planet_scenes.csv
    - data_source/data/planet_imagery/generated/lidar_capture_summary_for_planet_selection.csv

Description:
    For Los Angeles and New York City, summarize the USGS 3DEP LiDAR capture
    window and select one winter and one summer Planet scene from the existing
    scene-search results. The selection first looks for strict, clean scenes
    inside the LiDAR collection window. If the Planet inventory has no scene
    inside that window, it selects the nearest strict, clean post-LiDAR scene
    and records the gap.

Usage:
    python3 data_source/source/planet_imagery/select_lidar_aligned_planet_scenes.py
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
VENV_DIR = SCRIPT_DIR / "venv_planet_imagery"
VENV_PYTHON = VENV_DIR / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
VENV_MARKER = "PLANET_IMAGERY_VENV_ACTIVE"


def relaunch_inside_venv() -> None:
    """Restart this script inside the task-local virtual environment."""
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


SEASONS = {
    "winter_jan_dec": {1, 12},
    "summer_jun_jul": {6, 7},
}
DEFAULT_CITIES = ["los_angeles", "new_york_city"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select winter/summer Planet scenes aligned to USGS LiDAR capture windows."
    )
    parser.add_argument(
        "--projects",
        type=Path,
        default=PROJECT_ROOT / "data_source" / "data" / "height_labels" / "generated" / "usgs_3dep_projects.csv",
        help="USGS 3DEP project summary CSV with collect_start and collect_end.",
    )
    parser.add_argument(
        "--scenes",
        type=Path,
        default=PROJECT_ROOT / "data_source" / "data" / "planet_imagery" / "generated" / "cities_scenes_results_planet.csv",
        help="Planet scene metadata CSV created by search_planet_city_scenes.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data_source" / "data" / "planet_imagery" / "generated" / "selected_lidar_aligned_planet_scenes.csv",
        help="Selected scene CSV that can be passed to order_selected_planet_scenes.py.",
    )
    parser.add_argument(
        "--capture-summary",
        type=Path,
        default=PROJECT_ROOT / "data_source" / "data" / "planet_imagery" / "generated" / "lidar_capture_summary_for_planet_selection.csv",
        help="Small CSV summarizing the LiDAR capture windows used for scene selection.",
    )
    parser.add_argument(
        "--city",
        action="append",
        dest="cities",
        default=None,
        help="City slug to process. Defaults to los_angeles and new_york_city.",
    )
    return parser.parse_args()


def require_columns(table: pd.DataFrame, required: set[str], label: str) -> None:
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"{label} is missing required columns: {sorted(missing)}")


def load_projects(path: Path, cities: list[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing USGS project CSV: {path}")
    projects = pd.read_csv(path)
    require_columns(
        projects,
        {
            "city_slug",
            "project_directory",
            "workunit",
            "collect_start",
            "collect_end",
            "quality_level",
            "selected_tile_count",
            "aoi_coverage_percent",
        },
        "USGS project CSV",
    )
    projects = projects[projects["city_slug"].isin(cities)].copy()
    if projects.empty:
        raise ValueError(f"No USGS project rows found for cities: {cities}")
    projects["collect_start_dt"] = pd.to_datetime(projects["collect_start"], format="mixed", utc=True)
    projects["collect_end_dt"] = pd.to_datetime(projects["collect_end"], format="mixed", utc=True)
    return projects


def load_scenes(path: Path, cities: list[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing Planet scene CSV: {path}")
    scenes = pd.read_csv(path)
    require_columns(
        scenes,
        {
            "city_slug",
            "id",
            "acquired",
            "cloud_cover",
            "aoi_coverage_percent",
            "quality_category",
        },
        "Planet scene CSV",
    )
    scenes = scenes[scenes["city_slug"].isin(cities)].copy()
    if scenes.empty:
        raise ValueError(f"No Planet scene rows found for cities: {cities}")
    scenes["acquired_dt"] = pd.to_datetime(scenes["acquired"], format="mixed", utc=True)
    scenes["acquired_date"] = scenes["acquired_dt"].dt.date.astype(str)
    scenes["acquired_year"] = scenes["acquired_dt"].dt.year
    scenes["acquired_month"] = scenes["acquired_dt"].dt.month
    return scenes


def summarize_lidar_capture(projects: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for city_slug, city_projects in projects.groupby("city_slug", sort=False):
        capture_start = city_projects["collect_start_dt"].min()
        capture_end = city_projects["collect_end_dt"].max()
        midpoint = capture_start + (capture_end - capture_start) / 2
        rows.append(
            {
                "city_slug": city_slug,
                "lidar_collect_start": capture_start.date().isoformat(),
                "lidar_collect_end": capture_end.date().isoformat(),
                "lidar_collect_midpoint": midpoint.date().isoformat(),
                "lidar_project_count": len(city_projects),
                "lidar_project_directories": ";".join(city_projects["project_directory"].astype(str)),
                "lidar_workunits": ";".join(city_projects["workunit"].astype(str)),
                "lidar_quality_levels": ";".join(sorted(city_projects["quality_level"].astype(str).unique())),
                "lidar_selected_tile_count": int(city_projects["selected_tile_count"].sum()),
                "lidar_aoi_coverage_percent_max": float(city_projects["aoi_coverage_percent"].max()),
            }
        )
    return pd.DataFrame(rows)


def strict_clean_candidates(scenes: pd.DataFrame, season_months: set[int]) -> pd.DataFrame:
    """
    Keep scenes that are suitable for training imagery.

    These are deliberately conservative: zero cloud, full AOI coverage, and
    Planet's standard quality category. If this filter becomes too strict for
    another city, relax it explicitly rather than silently changing it here.
    """
    return scenes[
        scenes["acquired_month"].isin(season_months)
        & scenes["cloud_cover"].eq(0)
        & scenes["aoi_coverage_percent"].eq(100)
        & scenes["quality_category"].eq("standard")
    ].copy()


def add_selection_scores(
    candidates: pd.DataFrame,
    capture_start: pd.Timestamp,
    capture_end: pd.Timestamp,
    capture_midpoint: pd.Timestamp,
) -> pd.DataFrame:
    scored = candidates.copy()
    scored["inside_lidar_window"] = scored["acquired_dt"].between(capture_start, capture_end)
    scored["days_from_lidar_midpoint"] = (scored["acquired_dt"] - capture_midpoint).abs().dt.days
    scored["days_after_lidar_end"] = (scored["acquired_dt"] - capture_end).dt.days
    scored["days_before_lidar_start"] = (capture_start - scored["acquired_dt"]).dt.days
    scored["haze_shadow_snow_sum"] = 0.0
    for column in ["shadow_percent", "snow_ice_percent", "heavy_haze_percent", "light_haze_percent"]:
        if column in scored.columns:
            scored["haze_shadow_snow_sum"] += scored[column].fillna(0).astype(float)
    if "clear_percent" in scored.columns:
        scored["clear_percent_sort"] = scored["clear_percent"].fillna(-1).astype(float)
    else:
        scored["clear_percent_sort"] = -1.0
    if "sun_elevation" in scored.columns:
        scored["sun_elevation_sort"] = scored["sun_elevation"].fillna(-999).astype(float)
    else:
        scored["sun_elevation_sort"] = -999.0
    return scored


def choose_scene(
    city_scenes: pd.DataFrame,
    season: str,
    capture_start: pd.Timestamp,
    capture_end: pd.Timestamp,
    capture_midpoint: pd.Timestamp,
) -> pd.Series:
    candidates = strict_clean_candidates(city_scenes, SEASONS[season])
    if candidates.empty:
        raise ValueError(
            f"No strict clean Planet candidates for {city_scenes['city_slug'].iloc[0]} {season}"
        )

    scored = add_selection_scores(candidates, capture_start, capture_end, capture_midpoint)
    inside = scored[scored["inside_lidar_window"]].copy()
    if not inside.empty:
        selected = inside.sort_values(
            [
                "days_from_lidar_midpoint",
                "haze_shadow_snow_sum",
                "cloud_cover",
                "clear_percent_sort",
                "sun_elevation_sort",
            ],
            ascending=[True, True, True, False, False],
        ).iloc[0].copy()
        selected["scene_lidar_relation"] = "inside_lidar_collection_window"
        selected["selection_rule"] = (
            "Strict zero-cloud, full-AOI, standard-quality scene inside LiDAR "
            "collection window; ranked by distance to LiDAR midpoint and clean-scene metadata."
        )
        return selected

    after = scored[scored["days_after_lidar_end"].ge(0)].copy()
    if not after.empty:
        selected = after.sort_values(
            [
                "days_after_lidar_end",
                "haze_shadow_snow_sum",
                "cloud_cover",
                "clear_percent_sort",
                "sun_elevation_sort",
            ],
            ascending=[True, True, True, False, False],
        ).iloc[0].copy()
        selected["scene_lidar_relation"] = "after_lidar_collection_window"
        selected["selection_rule"] = (
            "No strict clean scene exists inside the LiDAR collection window in "
            "the local Planet list; selected nearest strict clean post-LiDAR scene."
        )
        return selected

    selected = scored.sort_values(
        [
            "days_before_lidar_start",
            "haze_shadow_snow_sum",
            "cloud_cover",
            "clear_percent_sort",
            "sun_elevation_sort",
        ],
        ascending=[True, True, True, False, False],
    ).iloc[0].copy()
    selected["scene_lidar_relation"] = "before_lidar_collection_window"
    selected["selection_rule"] = (
        "No strict clean scene exists inside or after the LiDAR collection window "
        "in the local Planet list; selected nearest strict clean pre-LiDAR scene."
    )
    return selected


def select_scenes(scenes: pd.DataFrame, capture_summary: pd.DataFrame) -> pd.DataFrame:
    selected_rows: list[pd.Series] = []
    for _, capture in capture_summary.iterrows():
        city_slug = capture["city_slug"]
        city_scenes = scenes[scenes["city_slug"].eq(city_slug)].copy()
        capture_start = pd.Timestamp(capture["lidar_collect_start"], tz="UTC")
        capture_end = pd.Timestamp(capture["lidar_collect_end"], tz="UTC")
        capture_midpoint = pd.Timestamp(capture["lidar_collect_midpoint"], tz="UTC")
        for season in SEASONS:
            selected = choose_scene(city_scenes, season, capture_start, capture_end, capture_midpoint)
            selected["selection_season"] = season
            selected["lidar_collect_start"] = capture["lidar_collect_start"]
            selected["lidar_collect_end"] = capture["lidar_collect_end"]
            selected["lidar_collect_midpoint"] = capture["lidar_collect_midpoint"]
            selected["lidar_project_directories"] = capture["lidar_project_directories"]
            selected["lidar_workunits"] = capture["lidar_workunits"]
            selected["lidar_selected_tile_count"] = capture["lidar_selected_tile_count"]
            selected["selected_asset_type"] = ""
            selected["selected_asset_type_reason"] = (
                "Asset availability will be checked by order_selected_planet_scenes.py."
            )
            selected_rows.append(selected)
    selected = pd.DataFrame(selected_rows)
    front_columns = [
        "city_slug",
        "city",
        "selection_season",
        "id",
        "acquired",
        "acquired_local",
        "acquired_date",
        "cloud_cover",
        "clear_percent",
        "sun_elevation",
        "quality_category",
        "aoi_coverage_percent",
        "scene_lidar_relation",
        "days_from_lidar_midpoint",
        "days_after_lidar_end",
        "days_before_lidar_start",
        "lidar_collect_start",
        "lidar_collect_end",
        "lidar_collect_midpoint",
        "lidar_project_directories",
        "lidar_workunits",
        "lidar_selected_tile_count",
        "selection_rule",
        "selected_asset_type",
        "selected_asset_type_reason",
    ]
    existing_front_columns = [column for column in front_columns if column in selected.columns]
    other_columns = [column for column in selected.columns if column not in existing_front_columns]
    return selected[existing_front_columns + other_columns]


def write_csv(path: Path, table: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    table.to_csv(temporary_path, index=False)
    temporary_path.replace(path)


def main() -> None:
    args = parse_args()
    cities = args.cities or DEFAULT_CITIES
    projects = load_projects(args.projects, cities)
    scenes = load_scenes(args.scenes, cities)
    capture_summary = summarize_lidar_capture(projects)
    selected = select_scenes(scenes, capture_summary)
    write_csv(args.capture_summary, capture_summary)
    write_csv(args.output, selected)

    print(f"WROTE {args.capture_summary} ({len(capture_summary)} city rows)")
    print(f"WROTE {args.output} ({len(selected)} selected scenes)")
    print(
        selected[
            [
                "city_slug",
                "selection_season",
                "id",
                "acquired_date",
                "scene_lidar_relation",
                "days_from_lidar_midpoint",
                "days_after_lidar_end",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
