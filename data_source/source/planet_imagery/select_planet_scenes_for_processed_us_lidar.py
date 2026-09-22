#!/usr/bin/env python3
"""Select eight PlanetScope scenes for every completed U.S. LiDAR nDSM city.

The completed-city universe comes from the U.S. LiDAR-to-nDSM run manifest,
not from the broader open-LiDAR inventory.  One LiDAR acquisition year is
chosen reproducibly per city with a SHA-256 seeded draw.  Scene selection then
uses the same rules for every city: acquisition-year proximity, solstice-season
balance, scene-centroid direction balance, strict-to-relaxed cloud/coverage
tiers, sun-elevation diversity, RGB+NIR asset availability, 8-band preference,
view angle, and atmospheric-artifact penalties.

This script only reads metadata and Planet asset listings.  It cannot activate,
order, or download imagery.
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
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
DEFAULT_SEED = 20260921
TARGET_SCENES = 8
COMPLETED_STATUSES = {"ndsm_validated", "complete_lidar_retained", "complete_lidar_deleted"}


def relaunch_inside_venv() -> None:
    """Use the established Planet environment on both Windows and macOS."""
    if os.environ.get(VENV_MARKER) == "1" or Path(sys.prefix).absolute() == VENV_DIR.absolute():
        return
    if not VENV_PYTHON.exists():
        raise SystemExit(
            "ERROR: Missing Planet environment. Recreate it from "
            "data_source/source/planet_imagery/requirements.txt."
        )
    environment = os.environ.copy()
    environment[VENV_MARKER] = "1"
    if os.name == "nt":
        import subprocess

        completed = subprocess.run(
            [str(VENV_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]],
            env=environment,
            check=False,
        )
        raise SystemExit(completed.returncode)
    os.execve(
        str(VENV_PYTHON),
        [str(VENV_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]],
        environment,
    )


relaunch_inside_venv()

import pandas as pd

import select_planet_global_city_scenes as global_selector
import select_planet_scenes_for_training_lidar_years as lidar_selector


YEAR_COLUMNS = [
    "wup_urbancode", "city_slug", "city_name", "country", "split_group",
    "split_seed", "randomized_city_rank", "aoi_path", "ndsm_status",
    "ndsm_path", "lidar_collect_start", "lidar_collect_end",
    "available_lidar_years", "selected_lidar_year", "year_selection_seed",
    "year_selection_algorithm", "year_selection_score_sha256",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ndsm-manifest",
        type=Path,
        default=PROJECT_ROOT / (
            "data_source/data/height_labels/generated/us_training_planet_ndsm/"
            "us_lidar_to_planet_ndsm_manifest.csv"
        ),
    )
    parser.add_argument(
        "--city-inventory",
        type=Path,
        default=PROJECT_ROOT / (
            "data_source/data/height_labels/generated/training_open_lidar/"
            "training_cities_with_open_lidar.csv"
        ),
    )
    parser.add_argument(
        "--combined-metadata",
        type=Path,
        default=PROJECT_ROOT / (
            "data_source/data/planet_imagery/generated/"
            "training_lidar_year_scene_selection/"
            "all_94_training_lidar_city_scene_metadata.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / (
            "data_source/data/planet_imagery/generated/"
            "processed_us_lidar_scene_selection"
        ),
    )
    parser.add_argument("--year-seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--city-slug", action="append", dest="city_slugs")
    parser.add_argument("--city-offset", type=int, default=0)
    parser.add_argument("--city-limit", type=int, default=0)
    parser.add_argument("--asset-candidates-per-city", type=int, default=120)
    parser.add_argument("--asset-check-concurrency", type=int, default=4)
    parser.add_argument(
        "--skip-asset-check",
        action="store_true",
        help="Offline test only. Unverified output must never be ordered.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return global_selector.resolve_project_path(path)


def require_columns(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def years_from_collection(start_value: str, end_value: str) -> list[int]:
    """Return every calendar year touched by the validated collection interval."""
    start_match = re.search(r"(?:19|20)\d{2}", str(start_value))
    end_match = re.search(r"(?:19|20)\d{2}", str(end_value))
    if not start_match or not end_match:
        raise ValueError(
            f"Could not parse LiDAR collection years from start={start_value!r}, end={end_value!r}"
        )
    start_year, end_year = int(start_match.group()), int(end_match.group())
    if start_year > end_year or end_year - start_year > 20:
        raise ValueError(f"Implausible LiDAR collection interval: {start_year}-{end_year}")
    return list(range(start_year, end_year + 1))


def choose_year(city_slug: str, years: list[int], seed: int) -> tuple[int, str]:
    """Make a stable, row-order-independent pseudo-random year draw."""
    scored = []
    for year in sorted(set(years)):
        score = hashlib.sha256(f"{seed}:{city_slug}:{year}".encode("utf-8")).hexdigest()
        scored.append((score, year))
    score, year = min(scored)
    return year, score


def load_completed_cities(manifest_path: Path, inventory_path: Path, seed: int) -> pd.DataFrame:
    """Join validated nDSM rows to city metadata and choose one year per city."""
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing U.S. nDSM run manifest: {manifest_path}")
    if not inventory_path.is_file():
        raise FileNotFoundError(f"Missing training-city inventory: {inventory_path}")
    manifest = pd.read_csv(manifest_path, dtype=str).fillna("")
    inventory = pd.read_csv(inventory_path, dtype=str).fillna("")
    require_columns(
        manifest,
        {
            "city_slug", "status", "output_ndsm_path", "lidar_collect_start",
            "lidar_collect_end",
        },
        "nDSM manifest",
    )
    require_columns(
        inventory,
        {
            "wup_urbancode", "city_slug", "city_name", "country", "split_group",
            "split_seed", "randomized_city_rank", "aoi_path",
        },
        "city inventory",
    )
    completed = manifest[manifest["status"].isin(COMPLETED_STATUSES)].copy()
    if completed.empty:
        raise RuntimeError("The nDSM manifest contains no validated completed cities.")
    if completed["city_slug"].duplicated().any():
        raise ValueError("The nDSM manifest contains duplicate completed city rows.")

    # A status alone is insufficient: the actual generated raster must exist.
    missing_outputs = []
    for _, row in completed.iterrows():
        output = resolve(Path(row["output_ndsm_path"]))
        if not output.is_file():
            missing_outputs.append(f"{row['city_slug']}: {row['output_ndsm_path']}")
    if missing_outputs:
        raise FileNotFoundError(
            "Completed manifest rows reference missing nDSM files: " + "; ".join(missing_outputs)
        )

    # The run manifest also carries descriptive fields such as city_name. Keep
    # only processing fields before joining the authoritative city inventory;
    # otherwise pandas adds _x/_y suffixes and the expected metadata names no
    # longer exist. This is especially important for manifests produced by the
    # Windows orchestrator, which use the complete MANIFEST_COLUMNS schema.
    completed_processing = completed[
        [
            "city_slug", "status", "output_ndsm_path",
            "lidar_collect_start", "lidar_collect_end",
        ]
    ].copy()
    joined = completed_processing.merge(
        inventory[
            [
                "wup_urbancode", "city_slug", "city_name", "country", "split_group",
                "split_seed", "randomized_city_rank", "aoi_path",
            ]
        ],
        on="city_slug",
        how="left",
        validate="one_to_one",
    )
    if joined["city_name"].eq("").any() or joined["city_name"].isna().any():
        raise ValueError("At least one completed U.S. city is absent from the city inventory.")
    if set(joined["country"]) != {"United States of America"}:
        raise ValueError("Completed-city selection contains a non-U.S. city.")
    if set(joined["split_group"]) != {"training"}:
        raise ValueError("Completed-city selection contains a non-training city.")

    records = []
    for _, row in joined.iterrows():
        years = years_from_collection(row["lidar_collect_start"], row["lidar_collect_end"])
        selected_year, score = choose_year(str(row["city_slug"]), years, seed)
        records.append(
            {
                "wup_urbancode": row["wup_urbancode"],
                "city_slug": row["city_slug"],
                "city_name": row["city_name"],
                "country": row["country"],
                "split_group": row["split_group"],
                "split_seed": int(row["split_seed"]),
                "randomized_city_rank": int(row["randomized_city_rank"]),
                "aoi_path": row["aoi_path"],
                "ndsm_status": row["status"],
                "ndsm_path": row["output_ndsm_path"],
                "lidar_collect_start": row["lidar_collect_start"],
                "lidar_collect_end": row["lidar_collect_end"],
                "available_lidar_years": ";".join(map(str, years)),
                "selected_lidar_year": selected_year,
                "year_selection_seed": seed,
                "year_selection_algorithm": "minimum_sha256(seed:city_slug:year)",
                "year_selection_score_sha256": score,
            }
        )
    result = pd.DataFrame(records, columns=YEAR_COLUMNS)
    return result.sort_values(["randomized_city_rank", "city_slug"]).reset_index(drop=True)


def selector_city_row(row: pd.Series) -> pd.Series:
    """Present the chosen year through the tested LiDAR selector interface."""
    year = int(row["selected_lidar_year"])
    converted = row.copy()
    converted["lidar_acquisition_years"] = str(year)
    converted["lidar_acquisition_start_date"] = f"{year}-01-01"
    converted["lidar_acquisition_end_date"] = f"{year}-12-31"
    converted["lidar_acquisition_date_precision"] = "deterministic_draw_from_validated_collection_interval"
    converted["lidar_acquisition_date_source"] = "us_lidar_to_planet_ndsm_manifest.csv"
    return converted


def selection_summary(
    city: pd.Series,
    metadata: pd.DataFrame,
    candidates: pd.DataFrame,
    compatible: pd.DataFrame,
    selected: pd.DataFrame,
) -> dict[str, Any]:
    """Extend the established summary with the chosen-year audit fields."""
    base = lidar_selector.city_total_summary(city, metadata, candidates, compatible, selected)
    return {
        **base,
        "available_lidar_years": city["available_lidar_years"],
        "selected_lidar_year": city["selected_lidar_year"],
        "year_selection_seed": city["year_selection_seed"],
        "year_selection_score_sha256": city["year_selection_score_sha256"],
    }


def rebuild_combined_outputs(output_dir: Path, years: pd.DataFrame) -> None:
    scene_files = sorted((output_dir / "by_city").glob("*_selected_planet_scenes.csv"))
    scene_frames = [pd.read_csv(path, low_memory=False) for path in scene_files]
    scene_frames = [frame for frame in scene_frames if not frame.empty]
    selected = pd.concat(scene_frames, ignore_index=True) if scene_frames else pd.DataFrame()
    if not selected.empty:
        selected = selected.sort_values(["randomized_city_rank", "city_slug", "selection_rank"])
        if selected[["city_slug", "scene_id"]].duplicated().any():
            raise RuntimeError("Combined selection contains duplicate city/scene rows.")
    global_selector.atomic_write_csv(
        selected, output_dir / "selected_processed_us_lidar_planet_scenes.csv"
    )

    summary_files = sorted((output_dir / "by_city_summary").glob("*_selection_summary.csv"))
    summaries = [pd.read_csv(path) for path in summary_files]
    summary = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    if not summary.empty:
        summary = summary.sort_values(["city_slug"])
    global_selector.atomic_write_csv(
        summary, output_dir / "processed_us_lidar_scene_selection_summary.csv"
    )
    shortfalls = (
        summary[pd.to_numeric(summary["selected_scene_count"]) != TARGET_SCENES].copy()
        if not summary.empty
        else summary.copy()
    )
    global_selector.atomic_write_csv(
        shortfalls, output_dir / "processed_us_lidar_scene_selection_shortfalls.csv"
    )
    global_selector.atomic_write_csv(years, output_dir / "processed_us_lidar_city_years.csv", YEAR_COLUMNS)


async def main() -> None:
    args = parse_args()
    started = datetime.now(timezone.utc)
    output_dir = resolve(args.output_dir)
    log_path = output_dir / "logs" / (
        f"select_processed_us_lidar_planet_scenes_{started.strftime('%Y%m%dT%H%M%SZ')}.log"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_lines = ["status=RUNNING", f"started_utc={started.isoformat()}"]
    try:
        if args.city_offset < 0 or args.city_limit < 0:
            raise ValueError("city-offset and city-limit cannot be negative")
        if args.asset_candidates_per_city < TARGET_SCENES or args.asset_check_concurrency < 1:
            raise ValueError("Candidate limit must be at least 8 and concurrency at least 1")
        manifest_path = resolve(args.ndsm_manifest)
        inventory_path = resolve(args.city_inventory)
        metadata_path = resolve(args.combined_metadata)
        years = load_completed_cities(manifest_path, inventory_path, args.year_seed)
        global_selector.atomic_write_csv(years, output_dir / "processed_us_lidar_city_years.csv", YEAR_COLUMNS)

        requested = set(args.city_slugs or [])
        if requested:
            unknown = sorted(requested - set(years["city_slug"]))
            if unknown:
                raise ValueError(f"Requested cities are not completed U.S. nDSM cities: {unknown}")
            batch = years[years["city_slug"].isin(requested)].copy()
        else:
            batch = years.iloc[args.city_offset:].copy()
            if args.city_limit:
                batch = batch.head(args.city_limit).copy()
        if batch.empty:
            raise ValueError("Requested completed-city batch is empty")

        metadata_by_city = lidar_selector.load_combined_metadata(
            metadata_path, set(years["city_slug"].astype(str))
        )
        asset_cache_path = output_dir / "planet_scene_asset_availability.csv"
        asset_cache = global_selector.load_asset_cache(asset_cache_path)
        completed = skipped = 0
        print(
            f"Completed U.S. nDSM cities: {len(years)}; processing this batch: {len(batch)}",
            flush=True,
        )
        for number, (_, original_city) in enumerate(batch.iterrows(), start=1):
            slug = str(original_city["city_slug"])
            city = selector_city_row(original_city)
            city_output = output_dir / "by_city" / f"{slug}_selected_planet_scenes.csv"
            summary_output = output_dir / "by_city_summary" / f"{slug}_selection_summary.csv"
            if city_output.exists() and summary_output.exists() and not args.overwrite:
                print(f"[{number}/{len(batch)}] {slug}: skipped completed output", flush=True)
                skipped += 1
                continue
            metadata = metadata_by_city[slug].copy()
            print(
                f"[{number}/{len(batch)}] {slug}: selected LiDAR year "
                f"{city['selected_lidar_year']}; selecting 8 scenes",
                flush=True,
            )
            candidates = lidar_selector.prepare_city_total_candidates(metadata, city)
            shortlist = lidar_selector.city_total_shortlist(
                candidates, args.asset_candidates_per_city
            )
            compatible, asset_cache = await global_selector.check_candidate_assets(
                shortlist,
                asset_cache,
                asset_cache_path,
                args.asset_check_concurrency,
                args.skip_asset_check,
            )
            selected = lidar_selector.select_city_total_scenes(compatible, city)
            for field in YEAR_COLUMNS:
                if field in city:
                    selected[field] = city[field]
            if args.skip_asset_check:
                selected["asset_verification_status"] = "SKIPPED_NOT_ORDERABLE"
            output_columns = [
                column for column in selected.columns
                if column not in {"acquired_dt", "acquired_month"}
            ]
            global_selector.atomic_write_csv(selected, city_output, output_columns)
            summary = selection_summary(city, metadata, candidates, compatible, selected)
            global_selector.atomic_write_csv(pd.DataFrame([summary]), summary_output)
            completed += 1
            print(
                f"  selected={len(selected)} acquisition_year="
                f"{summary['acquisition_year_scene_count']} post_lidar="
                f"{summary['post_lidar_scene_count']} pre_lidar_fallback="
                f"{summary['pre_lidar_fallback_scene_count']}",
                flush=True,
            )

        rebuild_combined_outputs(output_dir, years)
        combined = pd.read_csv(output_dir / "selected_processed_us_lidar_planet_scenes.csv")
        counts = combined.groupby("city_slug")["scene_id"].nunique()
        bad = counts[counts != TARGET_SCENES]
        if not bad.empty:
            raise RuntimeError(
                "Final output is not orderable because completed city outputs do not "
                f"contain exactly eight unique scenes: {bad.to_dict()}"
            )
        if args.skip_asset_check:
            raise RuntimeError(
                "Offline asset checks were skipped. Outputs were written for review but are not orderable."
            )

        all_cities_complete = len(counts) == len(years)
        log_lines[0] = "status=SUCCESS" if all_cities_complete else "status=PARTIAL_SUCCESS"
        log_lines.extend(
            [
                f"completed_us_city_count={len(years)}",
                f"batch_completed={completed}",
                f"batch_skipped={skipped}",
                f"selected_scene_count={len(combined)}",
                f"year_selection_seed={args.year_seed}",
            ]
        )
        log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
        if all_cities_complete:
            print(f"SUCCESS: {len(years)} cities x 8 scenes are ready for order planning.", flush=True)
        else:
            print(
                f"PARTIAL SUCCESS: {len(counts)} of {len(years)} completed cities now have "
                "eight scenes. Run the remaining city batch before order planning.",
                flush=True,
            )
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
        raise SystemExit(130)
