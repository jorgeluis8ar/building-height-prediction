#!/usr/bin/env python3
"""Select intermediate PlanetScope scenes that diversify sun elevation.

This script is for the NYC/LA HTC-DC Net multi-scene experiment.  It starts
from the two LiDAR-aligned scenes already used in the 6-channel model, then
looks for two additional strict-clean scenes between those dates.  The chosen
pair is the one that gives the four-scene set the widest and most diverse sun
elevation values.

The selected-scene output is intentionally compatible with
order_selected_planet_scenes.py.  Always dry-run the order script before
submitting Planet orders.
"""

from __future__ import annotations

import argparse
from itertools import combinations
import os
from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
VENV_DIR = SCRIPT_DIR / "venv_planet_imagery"
VENV_PYTHON = VENV_DIR / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
VENV_MARKER = "PLANET_IMAGERY_VENV_ACTIVE"


def relaunch_inside_venv() -> None:
    """Restart this script inside the task-local Planet imagery environment."""
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
        print(
            "  data_source/source/planet_imagery/venv_planet_imagery/bin/python "
            "-m pip install -r data_source/source/planet_imagery/requirements.txt"
        )
        sys.exit(1)

    env = os.environ.copy()
    env[VENV_MARKER] = "1"
    os.execve(str(VENV_PYTHON), [str(VENV_PYTHON), str(Path(__file__).resolve())] + sys.argv[1:], env)


relaunch_inside_venv()

import numpy as np
import pandas as pd


DEFAULT_CITIES = ["los_angeles", "new_york_city"]
OUTPUT_COLUMNS_FRONT = [
    "city_slug",
    "city",
    "selection_season",
    "scene_role",
    "scene_order",
    "id",
    "acquired",
    "acquired_local",
    "acquired_date",
    "sun_elevation",
    "clear_percent",
    "cloud_cover",
    "aoi_coverage_percent",
    "quality_category",
    "selection_rule",
    "selected_asset_type",
    "selected_asset_type_reason",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenes",
        type=Path,
        default=PROJECT_ROOT / "data_source/data/planet_imagery/generated/cities_scenes_results_planet.csv",
        help="Planet scene metadata CSV created by search_planet_city_scenes.py.",
    )
    parser.add_argument(
        "--existing-scenes",
        type=Path,
        default=PROJECT_ROOT
        / "data_source/data/planet_imagery/generated/selected_lidar_aligned_planet_scenes.csv",
        help="The two-scene LiDAR-aligned selection currently used by the 6-channel model.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT
        / "data_source/data/planet_imagery/generated/selected_intermediate_planet_scenes.csv",
        help="Order-compatible CSV with the two newly selected scenes per city.",
    )
    parser.add_argument(
        "--review-output",
        type=Path,
        default=PROJECT_ROOT
        / "data_source/data/planet_imagery/generated/intermediate_sun_elevation_scene_review.csv",
        help="Review CSV containing existing and new scenes plus selection diagnostics.",
    )
    parser.add_argument(
        "--city",
        action="append",
        dest="cities",
        default=None,
        help="City slug to process. Defaults to los_angeles and new_york_city.",
    )
    parser.add_argument(
        "--selection-objective",
        choices=["sun-elevation-diversity"],
        default="sun-elevation-diversity",
        help="Selection objective. Kept explicit so future objectives are auditable.",
    )
    return parser.parse_args()


def require_columns(table: pd.DataFrame, required: set[str], label: str) -> None:
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"{label} is missing required columns: {sorted(missing)}")


def resolve_project_path(path: Path) -> Path:
    resolved = path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()
    if not resolved.is_relative_to(PROJECT_ROOT):
        raise ValueError(f"Path is outside the project repository: {resolved}")
    return resolved


def load_scenes(path: Path, cities: list[str]) -> pd.DataFrame:
    path = resolve_project_path(path)
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
            "sun_elevation",
        },
        "Planet scene CSV",
    )
    scenes = scenes[scenes["city_slug"].isin(cities)].copy()
    if scenes.empty:
        raise ValueError(f"No Planet scene rows found for cities: {cities}")
    scenes["acquired_dt"] = pd.to_datetime(scenes["acquired"], format="mixed", utc=True)
    scenes["acquired_date"] = scenes["acquired_dt"].dt.date.astype(str)
    scenes["cloud_cover"] = scenes["cloud_cover"].astype(float)
    scenes["aoi_coverage_percent"] = scenes["aoi_coverage_percent"].astype(float)
    scenes["sun_elevation"] = scenes["sun_elevation"].astype(float)
    if "clear_percent" in scenes.columns:
        scenes["clear_percent"] = scenes["clear_percent"].astype(float)
    else:
        scenes["clear_percent"] = np.nan
    for column in ["shadow_percent", "snow_ice_percent", "heavy_haze_percent", "light_haze_percent"]:
        if column in scenes.columns:
            scenes[column] = scenes[column].fillna(0).astype(float)
        else:
            scenes[column] = 0.0
    scenes["haze_shadow_snow_sum"] = (
        scenes["shadow_percent"]
        + scenes["snow_ice_percent"]
        + scenes["heavy_haze_percent"]
        + scenes["light_haze_percent"]
    )
    return scenes


def load_existing_selection(path: Path, cities: list[str]) -> pd.DataFrame:
    path = resolve_project_path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing existing scene selection CSV: {path}")
    existing = pd.read_csv(path)
    require_columns(
        existing,
        {"city_slug", "id", "acquired", "selection_season", "sun_elevation"},
        "Existing selected scene CSV",
    )
    existing = existing[existing["city_slug"].isin(cities)].copy()
    existing["acquired_dt"] = pd.to_datetime(existing["acquired"], format="mixed", utc=True)
    existing["acquired_date"] = existing["acquired_dt"].dt.date.astype(str)
    existing["sun_elevation"] = existing["sun_elevation"].astype(float)
    return existing


def strict_intermediate_candidates(city_scenes: pd.DataFrame, existing_city: pd.DataFrame) -> pd.DataFrame:
    """Keep strict-clean scenes between the two currently selected scene dates."""
    if len(existing_city) != 2:
        city_slug = existing_city["city_slug"].iloc[0] if not existing_city.empty else "unknown"
        raise ValueError(f"Expected exactly two existing selected scenes for {city_slug}; found {len(existing_city)}")

    window_start = existing_city["acquired_dt"].min()
    window_end = existing_city["acquired_dt"].max()
    existing_ids = set(existing_city["id"].astype(str))
    candidates = city_scenes[
        city_scenes["acquired_dt"].gt(window_start)
        & city_scenes["acquired_dt"].lt(window_end)
        & city_scenes["cloud_cover"].eq(0)
        & city_scenes["aoi_coverage_percent"].eq(100)
        & city_scenes["quality_category"].eq("standard")
        & ~city_scenes["id"].astype(str).isin(existing_ids)
    ].copy()
    return candidates


def pair_score(pair: tuple[pd.Series, pd.Series], existing_city: pd.DataFrame) -> dict:
    """Score a candidate pair by sun-elevation diversity, then clean-scene metadata."""
    all_sun = np.array(
        existing_city["sun_elevation"].astype(float).tolist()
        + [float(pair[0]["sun_elevation"]), float(pair[1]["sun_elevation"])],
        dtype="float64",
    )
    all_dates = pd.to_datetime(
        existing_city["acquired_dt"].tolist() + [pair[0]["acquired_dt"], pair[1]["acquired_dt"]],
        utc=True,
    ).sort_values()
    date_gaps = np.diff(all_dates.view("int64")) / (24 * 60 * 60 * 1e9)
    return {
        "sun_elevation_range": float(np.max(all_sun) - np.min(all_sun)),
        "sun_elevation_std": float(np.std(all_sun)),
        "sun_elevation_min_gap": float(np.min(np.diff(np.sort(all_sun)))),
        "pair_clear_percent_mean": float(np.nanmean([pair[0]["clear_percent"], pair[1]["clear_percent"]])),
        "pair_haze_shadow_snow_sum": float(pair[0]["haze_shadow_snow_sum"] + pair[1]["haze_shadow_snow_sum"]),
        "date_min_gap_days": float(np.min(date_gaps)) if len(date_gaps) else 0.0,
    }


def choose_pair(city_scenes: pd.DataFrame, existing_city: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    candidates = strict_intermediate_candidates(city_scenes, existing_city)
    city_slug = existing_city["city_slug"].iloc[0]
    if len(candidates) < 2:
        raise ValueError(
            f"Need at least two strict-clean intermediate scenes for {city_slug}; found {len(candidates)}."
        )

    scored_pairs = []
    candidate_rows = [row for _, row in candidates.iterrows()]
    for first, second in combinations(candidate_rows, 2):
        score = pair_score((first, second), existing_city)
        score["first_id"] = first["id"]
        score["second_id"] = second["id"]
        scored_pairs.append((score, first, second))

    scored_pairs.sort(
        key=lambda item: (
            item[0]["sun_elevation_range"],
            item[0]["sun_elevation_std"],
            item[0]["sun_elevation_min_gap"],
            item[0]["pair_clear_percent_mean"],
            item[0]["date_min_gap_days"],
            -item[0]["pair_haze_shadow_snow_sum"],
        ),
        reverse=True,
    )
    best_score, first, second = scored_pairs[0]
    chosen = pd.DataFrame([first, second]).sort_values("acquired_dt").copy()
    return chosen, best_score


def season_label_for_new_scene(row: pd.Series, order: int) -> str:
    """Create an order-safe label that keeps intermediate scenes distinct."""
    date_text = str(row["acquired_date"]).replace("-", "")
    return f"intermediate_sun_elevation_{order}_{date_text}"


def build_outputs(scenes: pd.DataFrame, existing: pd.DataFrame, cities: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_rows = []
    review_rows = []
    for city_slug in cities:
        city_scenes = scenes[scenes["city_slug"].eq(city_slug)].copy()
        existing_city = existing[existing["city_slug"].eq(city_slug)].sort_values("acquired_dt").copy()
        chosen, score = choose_pair(city_scenes, existing_city)

        existing_review = existing_city.copy()
        existing_review["scene_role"] = "existing_6ch_scene"
        existing_review["scene_order"] = range(1, len(existing_review) + 1)
        existing_review["selection_rule"] = "Existing LiDAR-aligned scene already used in the 6-channel model."
        for _, row in existing_review.iterrows():
            review_rows.append(row.to_dict() | score)

        for order, (_, row) in enumerate(chosen.iterrows(), start=1):
            out = row.copy()
            out["selection_season"] = season_label_for_new_scene(out, order)
            out["scene_role"] = "new_intermediate_scene"
            out["scene_order"] = order
            out["selection_rule"] = (
                "Strict zero-cloud, full-AOI, standard-quality intermediate scene selected "
                "to maximize sun-elevation diversity across the four-scene city set."
            )
            out["selected_asset_type"] = ""
            out["selected_asset_type_reason"] = "Asset availability will be checked by order_selected_planet_scenes.py."
            selected_rows.append(out)
            review_rows.append(out.to_dict() | score)

    selected = pd.DataFrame(selected_rows)
    review = pd.DataFrame(review_rows)
    selected = reorder_columns(selected, OUTPUT_COLUMNS_FRONT)
    review = reorder_columns(review, OUTPUT_COLUMNS_FRONT + list(score_columns()))
    return selected, review


def score_columns() -> list[str]:
    return [
        "sun_elevation_range",
        "sun_elevation_std",
        "sun_elevation_min_gap",
        "pair_clear_percent_mean",
        "pair_haze_shadow_snow_sum",
        "date_min_gap_days",
        "first_id",
        "second_id",
    ]


def reorder_columns(table: pd.DataFrame, front_columns: list[str]) -> pd.DataFrame:
    existing_front = [column for column in front_columns if column in table.columns]
    other_columns = [column for column in table.columns if column not in existing_front]
    return table[existing_front + other_columns]


def write_csv(path: Path, table: pd.DataFrame) -> None:
    path = resolve_project_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    table.to_csv(temporary_path, index=False)
    temporary_path.replace(path)


def main() -> None:
    args = parse_args()
    cities = args.cities or DEFAULT_CITIES
    scenes = load_scenes(args.scenes, cities)
    existing = load_existing_selection(args.existing_scenes, cities)
    selected, review = build_outputs(scenes, existing, cities)
    write_csv(args.output, selected)
    write_csv(args.review_output, review)

    print(f"WROTE {resolve_project_path(args.output)} ({len(selected)} new scenes)")
    print(f"WROTE {resolve_project_path(args.review_output)} ({len(review)} review rows)")
    print(
        review[
            [
                "city_slug",
                "scene_role",
                "scene_order",
                "id",
                "acquired_date",
                "sun_elevation",
                "clear_percent",
                "cloud_cover",
                "aoi_coverage_percent",
                "quality_category",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
