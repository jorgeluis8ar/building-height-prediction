"""
Select Planet City Scenes

Environment: data_source/source/planet_imagery/venv_planet_imagery

Requires (inputs from earlier stages):
    - README.md
    - data_source/data/planet_imagery/generated/cities_scenes_results_planet.csv
    - data_source/source/planet_imagery/building_footprint_source_dates.csv

Produces (outputs for later stages):
    - data_source/data/planet_imagery/generated/selected_planet_city_scenes.csv

Description:
    Selects two Planet PSScene rows per current city: one winter scene from
    January or December and one summer scene from June or July. The script
    requires zero cloud cover and 100% AOI coverage, then scores all eligible
    winter/summer pairs by closeness to the building-footprint source date,
    pair spacing, and quality metadata. It then checks Planet asset
    availability and assigns one common selected asset type per city pair.

Usage:
    python3 data_source/source/planet_imagery/select_planet_city_scenes.py

Expected runtime: < 1 minute
"""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path
import sys
from typing import Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
VENV_DIR = SCRIPT_DIR / "venv_planet_imagery"
VENV_PYTHON = VENV_DIR / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
VENV_MARKER = "PLANET_IMAGERY_VENV_ACTIVE"


def relaunch_inside_venv() -> None:
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
from planet import Auth, Session
from planet.clients import DataClient


WINTER_MONTHS = {1, 12}
SUMMER_MONTHS = {6, 7}
MAX_IDEAL_PAIR_GAP_DAYS = 183
DEFAULT_ASSET_CANDIDATES_PER_SEASON = 40
DEFAULT_ASSET_CHECK_CONCURRENCY = 8
PREFERRED_ASSET_TYPES = [
    "ortho_analytic_8b_sr",
    "ortho_analytic_8b",
    "ortho_analytic_4b_sr",
    "ortho_analytic_4b",
]
CITY_PAIR_OVERRIDES = {
    # Audited against Planet asset metadata on 2026-06-17. These cities had
    # earlier selected pairs without shared 8-band surface reflectance assets.
    "boston": {
        "winter_jan_dec": "20201206_145311_27_2264",
        "summer_jun_jul": "20210602_145757_43_2259",
        "selected_asset_type": "ortho_analytic_8b_sr",
    },
    "chicago": {
        "winter_jan_dec": "20201217_160252_53_2279",
        "summer_jun_jul": "20210613_155128_64_2439",
        "selected_asset_type": "ortho_analytic_8b_sr",
    },
    "helsinki": {
        "winter_jan_dec": "20230131_091551_75_2489",
        "summer_jun_jul": "20230610_092134_81_247a",
        "selected_asset_type": "ortho_analytic_8b_sr",
    },
    "los_angeles": {
        "winter_jan_dec": "20201206_175447_63_222f",
        "summer_jun_jul": "20210605_174638_17_241a",
        "selected_asset_type": "ortho_analytic_8b_sr",
    },
    "new_york_city": {
        "winter_jan_dec": "20211226_153959_13_241c",
        "summer_jun_jul": "20210628_150649_99_227e",
        "selected_asset_type": "ortho_analytic_8b_sr",
    },
    "oslo": {
        "winter_jan_dec": "20200127_103228_14_105a",
        "summer_jun_jul": "20200613_104333_77_105d",
        "selected_asset_type": "ortho_analytic_4b_sr",
    },
    "vancouver": {
        "winter_jan_dec": "20211231_183044_66_2264",
        "summer_jun_jul": "20210704_182002_63_2460",
        "selected_asset_type": "ortho_analytic_8b_sr",
    },
}


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(
        description="Select two zero-cloud, full-AOI Planet scenes per city."
    )
    parser.add_argument(
        "--scenes",
        type=Path,
        default=project_root / "data_source" / "data" / "planet_imagery" / "generated" / "cities_scenes_results_planet.csv",
        help="Planet metadata CSV created by search_planet_city_scenes.py.",
    )
    parser.add_argument(
        "--footprint-dates",
        type=Path,
        default=project_root / "data_source" / "source" / "planet_imagery" / "building_footprint_source_dates.csv",
        help="City footprint source-date catalog.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "data_source" / "data" / "planet_imagery" / "generated" / "selected_planet_city_scenes.csv",
        help="Selected two-scene-per-city output CSV.",
    )
    parser.add_argument(
        "--asset-type",
        action="append",
        dest="asset_types",
        help="Preferred download asset type. May be supplied more than once in priority order.",
    )
    parser.add_argument(
        "--asset-candidates-per-season",
        type=int,
        default=DEFAULT_ASSET_CANDIDATES_PER_SEASON,
        help=(
            "Number of winter and summer candidate scenes per city to check "
            "for asset availability. Larger values search harder for a shared "
            "ortho_analytic_8b_sr pair."
        ),
    )
    parser.add_argument(
        "--asset-check-concurrency",
        type=int,
        default=DEFAULT_ASSET_CHECK_CONCURRENCY,
        help="Maximum concurrent Planet asset-availability requests.",
    )
    return parser.parse_args()


def load_scene_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing Planet scene metadata CSV: {path}")
    scenes = pd.read_csv(path)
    required = {"city_slug", "id", "acquired", "acquired_local", "cloud_cover", "aoi_coverage_percent"}
    missing = required - set(scenes.columns)
    if missing:
        raise ValueError(f"Scene table is missing required columns: {sorted(missing)}")

    scenes["acquired_dt"] = pd.to_datetime(scenes["acquired"], format="mixed", utc=True)
    scenes["acquired_local_dt"] = pd.to_datetime(scenes["acquired_local"], format="mixed", utc=True)
    scenes["acquired_local_date"] = scenes["acquired_local_dt"].dt.date.astype(str)
    scenes["acquired_local_year"] = scenes["acquired_local_dt"].dt.year
    scenes["acquired_local_month"] = scenes["acquired_local_dt"].dt.month
    scenes["acquired_local_hour"] = scenes["acquired_local_dt"].dt.hour
    return scenes


def load_footprint_dates(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing footprint source-date CSV: {path}")
    dates = pd.read_csv(path)
    required = {"city_slug", "footprint_reference_date"}
    missing = required - set(dates.columns)
    if missing:
        raise ValueError(f"Footprint date table is missing required columns: {sorted(missing)}")
    dates["footprint_reference_dt"] = pd.to_datetime(
        dates["footprint_reference_date"], format="mixed", utc=True
    )
    return dates


def requirement_penalty(row: pd.Series) -> float:
    """
    Penalize deviations from the requested hard filters.

    Most cities have strict candidates. This penalty only matters when a city
    has no strict winter or summer scene, as happened for Helsinki winter.
    """
    cloud_penalty = max(0.0, float(row.get("cloud_cover") or 0.0)) * 10000
    coverage_penalty = max(0.0, 100.0 - float(row.get("aoi_coverage_percent") or 0.0)) * 10000
    return cloud_penalty + coverage_penalty


def strict_requirements_met(row: pd.Series) -> bool:
    return (
        float(row.get("cloud_cover") or 0.0) == 0.0
        and float(row.get("aoi_coverage_percent") or 0.0) == 100.0
    )


def score_pair(winter: pd.Series, summer: pd.Series, footprint_dt: pd.Timestamp) -> tuple:
    pair_gap_days = abs((summer["acquired_local_dt"] - winter["acquired_local_dt"]).days)
    midpoint = winter["acquired_local_dt"] + (summer["acquired_local_dt"] - winter["acquired_local_dt"]) / 2
    footprint_distance_days = abs((midpoint - footprint_dt).days)
    gap_penalty = max(0, pair_gap_days - MAX_IDEAL_PAIR_GAP_DAYS) * 1000
    quality_penalty = 0
    if winter.get("quality_category") != "standard":
        quality_penalty += 100
    if summer.get("quality_category") != "standard":
        quality_penalty += 100

    haze_shadow_snow = 0.0
    for row in (winter, summer):
        for field in ["shadow_percent", "snow_ice_percent", "heavy_haze_percent", "light_haze_percent"]:
            value = row.get(field)
            if pd.notna(value):
                haze_shadow_snow += float(value)

    sun_bonus = 0.0
    for row in (winter, summer):
        value = row.get("sun_elevation")
        if pd.notna(value):
            sun_bonus -= float(value) / 1000.0

    score = (
        footprint_distance_days
        + gap_penalty
        + quality_penalty
        + requirement_penalty(winter)
        + requirement_penalty(summer)
        + haze_shadow_snow
        + sun_bonus
    )
    return (
        score,
        footprint_distance_days,
        pair_gap_days,
        pair_gap_days <= MAX_IDEAL_PAIR_GAP_DAYS,
    )


def season_candidate_pool(
    city_scenes: pd.DataFrame,
    months: set[int],
    footprint_dt: pd.Timestamp,
    max_candidates: int,
) -> pd.DataFrame:
    """
    Return candidate scenes for one season.

    We keep the requested strict filters whenever they exist. When a city has no
    strict scene for that season, we use requested-month alternatives and flag
    the eventual selected row. The pool is then ranked by closeness to the
    footprint date and basic image quality before Planet asset availability is
    checked.
    """
    desired_months = city_scenes[city_scenes["acquired_local_month"].isin(months)].copy()
    strict = desired_months[
        (desired_months["cloud_cover"] == 0)
        & (desired_months["aoi_coverage_percent"] == 100)
    ].copy()
    pool = strict if not strict.empty else desired_months
    if pool.empty:
        return pool

    pool = pool.copy()
    pool["season_candidate_score"] = pool.apply(
        lambda row: (
            abs((row["acquired_local_dt"] - footprint_dt).days)
            + requirement_penalty(row)
            + (0 if row.get("quality_category") == "standard" else 100)
            - (float(row.get("sun_elevation")) / 1000.0 if pd.notna(row.get("sun_elevation")) else 0.0)
        ),
        axis=1,
    )
    return pool.sort_values(["season_candidate_score", "id"]).head(max_candidates)


def candidate_scene_ids_for_city(
    city_scenes: pd.DataFrame,
    footprint_row: pd.Series,
    max_candidates: int,
) -> list[str]:
    city_slug = footprint_row["city_slug"]
    footprint_dt = footprint_row["footprint_reference_dt"]
    winter = season_candidate_pool(city_scenes, WINTER_MONTHS, footprint_dt, max_candidates)
    summer = season_candidate_pool(city_scenes, SUMMER_MONTHS, footprint_dt, max_candidates)
    scene_ids = set(winter["id"].astype(str)) | set(summer["id"].astype(str))
    override = CITY_PAIR_OVERRIDES.get(city_slug)
    if override:
        scene_ids.update(
            [
                override["winter_jan_dec"],
                override["summer_jun_jul"],
            ]
        )
    return sorted(scene_ids)


def common_asset_choice(
    winter_assets: set[str],
    summer_assets: set[str],
    preferred_asset_types: list[str],
) -> tuple[int, str] | None:
    common_assets = winter_assets & summer_assets
    for index, asset_type in enumerate(preferred_asset_types):
        if asset_type in common_assets:
            return index, asset_type
    return None


def available_asset_string(scene_id: str, availability_by_scene: dict[str, set[str]]) -> str:
    return ",".join(sorted(availability_by_scene.get(scene_id, set())))


def output_rows_for_pair(
    winter_row: pd.Series,
    summer_row: pd.Series,
    footprint_row: pd.Series,
    availability_by_scene: dict[str, set[str]],
    selected_asset_type: str,
    pair_note: str,
) -> list[dict]:
    footprint_dt = footprint_row["footprint_reference_dt"]
    score, footprint_distance_days, pair_gap_days, within_ideal_gap = score_pair(
        winter_row, summer_row, footprint_dt
    )

    rows = []
    for season, row in [("winter_jan_dec", winter_row), ("summer_jun_jul", summer_row)]:
        scene_id = str(row["id"])
        available_assets = availability_by_scene.get(scene_id, set())
        result = row.drop(labels=["acquired_dt", "acquired_local_dt"], errors="ignore").to_dict()
        result.update(
            {
                "selection_season": season,
                "selection_strict_requirements_met": strict_requirements_met(row),
                "selection_relaxation_reason": ""
                if strict_requirements_met(row)
                else "No strict zero-cloud/100%-AOI scene was available for this city-season; closest requested-month scene selected for review.",
                "footprint_reference_date": footprint_row["footprint_reference_date"],
                "footprint_date_precision": footprint_row.get("date_precision", ""),
                "footprint_dataset_last_updated": footprint_row.get("dataset_last_updated", ""),
                "footprint_source_name": footprint_row.get("source_name", ""),
                "footprint_source_url": footprint_row.get("source_url", ""),
                "footprint_date_basis": footprint_row.get("date_basis", ""),
                "footprint_date_confidence": footprint_row.get("date_confidence", ""),
                "available_asset_types": available_asset_string(scene_id, availability_by_scene),
                "has_ortho_analytic_8b_sr": "ortho_analytic_8b_sr" in available_assets,
                "selected_asset_type": selected_asset_type,
                "selected_asset_type_reason": pair_note,
                "pair_score": round(float(score), 6),
                "pair_footprint_midpoint_distance_days": int(footprint_distance_days),
                "pair_gap_days": int(pair_gap_days),
                "pair_within_ideal_6_month_gap": bool(within_ideal_gap),
            }
        )
        rows.append(result)
    return rows


def selected_rows_for_city(
    city_scenes: pd.DataFrame,
    footprint_row: pd.Series,
    availability_by_scene: dict[str, set[str]],
    preferred_asset_types: list[str],
    max_candidates: int,
) -> list[dict]:
    city_slug = footprint_row["city_slug"]
    footprint_dt = footprint_row["footprint_reference_dt"]
    override = CITY_PAIR_OVERRIDES.get(city_slug)
    if override:
        winter = city_scenes[city_scenes["id"].astype(str) == override["winter_jan_dec"]]
        summer = city_scenes[city_scenes["id"].astype(str) == override["summer_jun_jul"]]
        if winter.empty or summer.empty:
            raise RuntimeError(f"Manual scene override for {city_slug} is missing from the scene table.")

        selected_asset_type = override["selected_asset_type"]
        for scene_id in [override["winter_jan_dec"], override["summer_jun_jul"]]:
            if selected_asset_type not in availability_by_scene.get(scene_id, set()):
                raise RuntimeError(
                    f"Manual scene override for {city_slug} expected {selected_asset_type} "
                    f"for {scene_id}, but Planet asset metadata did not list it."
                )

        pair_note = (
            "Manual audited override: both city scenes have ortho_analytic_8b_sr."
            if selected_asset_type == "ortho_analytic_8b_sr"
            else "Manual audited override: Oslo did not have a reviewed shared ortho_analytic_8b_sr pair; both scenes use ortho_analytic_4b_sr."
        )
        return output_rows_for_pair(
            winter.iloc[0],
            summer.iloc[0],
            footprint_row,
            availability_by_scene,
            selected_asset_type,
            pair_note,
        )

    winter = season_candidate_pool(city_scenes, WINTER_MONTHS, footprint_dt, max_candidates)
    summer = season_candidate_pool(city_scenes, SUMMER_MONTHS, footprint_dt, max_candidates)

    if winter.empty or summer.empty:
        raise ValueError(
            f"{city_slug} does not have both winter and summer scenes in the "
            "requested months."
        )

    best = None
    for _, winter_row in winter.iterrows():
        for _, summer_row in summer.iterrows():
            winter_id = str(winter_row["id"])
            summer_id = str(summer_row["id"])
            asset_choice = common_asset_choice(
                availability_by_scene.get(winter_id, set()),
                availability_by_scene.get(summer_id, set()),
                preferred_asset_types,
            )
            if asset_choice is None:
                continue
            asset_rank, selected_asset_type = asset_choice
            score, footprint_distance_days, pair_gap_days, within_ideal_gap = score_pair(
                winter_row, summer_row, footprint_dt
            )
            candidate = (
                asset_rank,
                score,
                footprint_distance_days,
                pair_gap_days,
                str(winter_row["id"]),
                str(summer_row["id"]),
                winter_row,
                summer_row,
                within_ideal_gap,
                selected_asset_type,
            )
            if best is None or candidate[:6] < best[:6]:
                best = candidate

    if best is None:
        raise RuntimeError(
            f"No scene pair for {city_slug} has a common preferred asset type "
            f"inside the top {max_candidates} candidate scenes per season."
        )

    (
        asset_rank,
        score,
        footprint_distance_days,
        pair_gap_days,
        _,
        _,
        winter_row,
        summer_row,
        within_ideal_gap,
        selected_asset_type,
    ) = best
    pair_note = (
        "Both city scenes have ortho_analytic_8b_sr."
        if selected_asset_type == "ortho_analytic_8b_sr"
        else "No reviewed city pair with shared ortho_analytic_8b_sr was found in the candidate pool; both scenes use the best shared fallback asset type."
    )
    return output_rows_for_pair(
        winter_row,
        summer_row,
        footprint_row,
        availability_by_scene,
        selected_asset_type,
        pair_note,
    )


def auth_from_environment() -> Auth:
    api_key = os.environ.get("PL_API_KEY")
    if api_key:
        print("Authentication: PL_API_KEY environment variable")
        return Auth.from_key(api_key)

    print("Authentication: saved Planet OAuth profile")
    return Auth.from_user_default_session()


async def fetch_asset_availability(
    scene_ids: Iterable[str],
    concurrency: int,
) -> dict[str, set[str]]:
    auth = auth_from_environment()
    availability_by_scene: dict[str, set[str]] = {}
    semaphore = asyncio.Semaphore(concurrency)

    async with Session(auth=auth) as session:
        data_client = DataClient(session)

        async def fetch_one(scene_id: str) -> None:
            async with semaphore:
                assets = await data_client.list_item_assets("PSScene", scene_id)
                availability_by_scene[scene_id] = set(assets.keys())

        await asyncio.gather(*(fetch_one(scene_id) for scene_id in sorted(set(scene_ids))))

    return availability_by_scene


async def async_main() -> None:
    args = parse_args()
    scenes = load_scene_table(args.scenes)
    footprint_dates = load_footprint_dates(args.footprint_dates)
    preferred_asset_types = args.asset_types or PREFERRED_ASSET_TYPES
    if args.asset_candidates_per_season <= 0:
        raise SystemExit("--asset-candidates-per-season must be positive")
    if args.asset_check_concurrency <= 0:
        raise SystemExit("--asset-check-concurrency must be positive")

    missing_dates = sorted(set(scenes["city_slug"]) - set(footprint_dates["city_slug"]))
    if missing_dates:
        raise SystemExit(f"Missing footprint source dates for city/cities: {missing_dates}")

    candidate_scene_ids: set[str] = set()
    for city_slug in footprint_dates["city_slug"]:
        city_scenes = scenes[scenes["city_slug"] == city_slug]
        if city_scenes.empty:
            raise SystemExit(f"No Planet scene rows found for {city_slug}")
        footprint_row = footprint_dates[footprint_dates["city_slug"] == city_slug].iloc[0]
        candidate_scene_ids.update(
            candidate_scene_ids_for_city(
                city_scenes,
                footprint_row,
                args.asset_candidates_per_season,
            )
        )

    print(
        f"Checking Planet asset availability for {len(candidate_scene_ids)} "
        f"candidate scenes with concurrency={args.asset_check_concurrency}.",
        flush=True,
    )
    availability_by_scene = await fetch_asset_availability(
        candidate_scene_ids,
        args.asset_check_concurrency,
    )

    selected_rows: list[dict] = []
    for city_slug in footprint_dates["city_slug"]:
        city_scenes = scenes[scenes["city_slug"] == city_slug]
        if city_scenes.empty:
            raise SystemExit(f"No Planet scene rows found for {city_slug}")
        footprint_row = footprint_dates[footprint_dates["city_slug"] == city_slug].iloc[0]
        selected_rows.extend(
            selected_rows_for_city(
                city_scenes,
                footprint_row,
                availability_by_scene,
                preferred_asset_types,
                args.asset_candidates_per_season,
            )
        )

    selected = pd.DataFrame(selected_rows)
    if len(selected) != len(footprint_dates) * 2:
        raise RuntimeError(
            f"Expected {len(footprint_dates) * 2} selected rows, got {len(selected)}"
        )
    duplicates = selected.duplicated(subset=["city_slug", "selection_season"]).sum()
    if duplicates:
        raise RuntimeError("Duplicate city/season rows in selected scene output")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = args.output.with_suffix(args.output.suffix + ".tmp")
    selected.to_csv(temporary_path, index=False)
    temporary_path.replace(args.output)

    print(f"WROTE {args.output} ({len(selected)} selected scene rows)")
    print(selected.groupby("city_slug")["id"].count().to_string())
    print("Selected asset type by city:")
    print(selected.drop_duplicates("city_slug").set_index("city_slug")["selected_asset_type"].to_string())


if __name__ == "__main__":
    asyncio.run(async_main())
