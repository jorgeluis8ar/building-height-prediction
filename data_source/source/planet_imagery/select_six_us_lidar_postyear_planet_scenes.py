#!/usr/bin/env python3
"""Select eight same-year Planet scenes after locally used LiDAR vintages.

This selector is intentionally limited to the six U.S. cities being added to
the HTC dataset.  It searches calendar years in ascending order and accepts
the first year with eight asset-compatible, standard-quality solstice scenes:
four summer, four winter, and two from each cardinal direction.
"""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
VENV_DIR = SCRIPT_DIR / "venv_planet_imagery"
VENV_PYTHON = VENV_DIR / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
VENV_MARKER = "PLANET_IMAGERY_VENV_ACTIVE"


def relaunch_inside_venv() -> None:
    if os.environ.get(VENV_MARKER) == "1" or Path(sys.prefix).absolute() == VENV_DIR.absolute():
        return
    if not VENV_PYTHON.is_file():
        raise SystemExit(f"Missing Planet environment: {VENV_PYTHON}")
    environment = os.environ.copy()
    environment[VENV_MARKER] = "1"
    os.execve(str(VENV_PYTHON), [str(VENV_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]], environment)


relaunch_inside_venv()

import pandas as pd

import select_planet_global_city_scenes as global_selector
import select_planet_scenes_for_training_lidar_years as lidar_selector


# These are the latest vintages visible in the locally downloaded LiDAR
# project paths. Antioch is deliberately excluded from this experiment.
CITY_LIDAR_REFERENCE_YEARS = {
    "new_york_newark_23083": 2014,
    "cincinnati_22957": 2007,
    "harrisburg_23008": 2008,
    "miami_23064": 2007,
    "philadelphia_23098": 2008,
    "trenton_23165": 2009,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--combined-metadata",
        type=Path,
        default=PROJECT_ROOT / (
            "data_source/data/planet_imagery/generated/training_lidar_year_scene_selection/"
            "all_94_training_lidar_city_scene_metadata.csv"
        ),
    )
    parser.add_argument(
        "--miami-metadata",
        type=Path,
        default=PROJECT_ROOT / (
            "data_source/data/planet_imagery/generated/miami_scene_refresh_20261021/"
            "by_city/miami_23064_planet_scenes.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / (
            "data_source/data/planet_imagery/generated/"
            "six_us_lidar_postyear_scene_selection_v1"
        ),
    )
    parser.add_argument("--asset-check-concurrency", type=int, default=4)
    parser.add_argument("--skip-asset-check", action="store_true")
    return parser.parse_args()


def accepted(selected: pd.DataFrame) -> tuple[bool, str]:
    if len(selected) != 8:
        return False, f"selected_{len(selected)}_of_8"
    seasons = selected["selection_local_season"].value_counts()
    directions = selected["selection_cardinal_direction"].value_counts()
    reasons = []
    if int(seasons.get("summer", 0)) != 4 or int(seasons.get("winter", 0)) != 4:
        reasons.append("season_balance_failed")
    for direction in ("N", "S", "E", "W"):
        if int(directions.get(direction, 0)) != 2:
            reasons.append(f"direction_{direction.lower()}_count_{int(directions.get(direction, 0))}")
    return not reasons, ";".join(reasons)


async def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    combined = pd.read_csv(args.combined_metadata.resolve(), low_memory=False)
    miami = pd.read_csv(args.miami_metadata.resolve(), low_memory=False)
    cache_path = output_dir / "planet_scene_asset_availability.csv"
    cache = global_selector.load_asset_cache(cache_path)
    selected_frames = []
    audits = []

    for city_slug, lidar_year in CITY_LIDAR_REFERENCE_YEARS.items():
        source = miami if city_slug == "miami_23064" else combined
        metadata = lidar_selector.prepare_metadata_frame(
            source[source["city_slug"].astype(str).eq(city_slug)], city_slug, city_slug
        )
        candidates, _ = global_selector.prepare_candidates(metadata)
        available_years = sorted(
            int(year) for year in candidates["selection_calendar_year"].unique()
            if int(year) > lidar_year
        )
        city_accepted = False
        for planet_year in available_years:
            year_candidates = candidates[
                candidates["selection_calendar_year"].astype(int).eq(planet_year)
            ].copy()
            compatible, cache = await global_selector.check_candidate_assets(
                year_candidates,
                cache,
                cache_path,
                args.asset_check_concurrency,
                args.skip_asset_check,
            )
            selected = lidar_selector.select_same_year_scenes(
                compatible, planet_year, "first_complete_calendar_year_after_local_lidar"
            )
            passed, reason = accepted(selected)
            audits.append({
                "city_slug": city_slug,
                "lidar_reference_year": lidar_year,
                "planet_year_tested": planet_year,
                "metadata_candidate_count": len(year_candidates),
                "asset_compatible_candidate_count": len(compatible),
                "selected_scene_count": len(selected),
                "accepted": passed,
                "rejection_reason": reason,
            })
            print(
                f"{city_slug} {planet_year}: metadata={len(year_candidates)} "
                f"compatible={len(compatible)} selected={len(selected)} accepted={passed}"
            )
            if passed:
                selected = selected.copy()
                selected["id"] = selected["scene_id"]
                selected["selection_season"] = selected["selection_local_season"]
                selected["lidar_reference_year"] = lidar_year
                selected["selected_planet_year"] = planet_year
                selected["years_after_lidar"] = planet_year - lidar_year
                selected_frames.append(selected)
                city_accepted = True
                break
        if not city_accepted:
            audits.append({
                "city_slug": city_slug,
                "lidar_reference_year": lidar_year,
                "planet_year_tested": "",
                "metadata_candidate_count": 0,
                "asset_compatible_candidate_count": 0,
                "selected_scene_count": 0,
                "accepted": False,
                "rejection_reason": "no_available_post_lidar_year_satisfied_all_conditions",
            })

    selected_output = pd.concat(selected_frames, ignore_index=True) if selected_frames else pd.DataFrame()
    global_selector.atomic_write_csv(
        selected_output, output_dir / "selected_six_us_lidar_postyear_planet_scenes.csv"
    )
    global_selector.atomic_write_csv(
        pd.DataFrame(audits), output_dir / "six_city_year_search_audit.csv"
    )
    accepted_cities = selected_output["city_slug"].nunique() if not selected_output.empty else 0
    print(f"Completed selections for {accepted_cities} of {len(CITY_LIDAR_REFERENCE_YEARS)} cities")


if __name__ == "__main__":
    asyncio.run(main())
