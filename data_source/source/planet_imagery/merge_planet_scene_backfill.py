"""
Merge Planet Scene Backfill Results

Environment: data_source/source/planet_imagery/venv_planet_imagery

Requires:
    - data_source/data/planet_imagery/generated/cities_scenes_results_planet.csv
    - data_source/data/planet_imagery/generated/cities_scenes_results_planet_2010_2015_backfill.csv

Produces:
    - data_source/data/planet_imagery/generated/cities_scenes_results_planet.csv

Description:
    Append a targeted Planet metadata backfill to the main city-scene table,
    remove duplicate city/scene rows, sort consistently, and write the result
    atomically. This script only touches metadata CSVs; it does not activate,
    order, or download imagery assets.

Usage:
    python3 data_source/source/planet_imagery/merge_planet_scene_backfill.py
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


KEY_COLUMNS = ["city_slug", "id"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge a Planet scene metadata backfill into the main scene table."
    )
    parser.add_argument(
        "--main-scenes",
        type=Path,
        default=PROJECT_ROOT / "data_source" / "data" / "planet_imagery" / "generated" / "cities_scenes_results_planet.csv",
        help="Main Planet scene metadata CSV.",
    )
    parser.add_argument(
        "--backfill-scenes",
        type=Path,
        default=PROJECT_ROOT / "data_source" / "data" / "planet_imagery" / "generated" / "cities_scenes_results_planet_2010_2015_backfill.csv",
        help="Backfill Planet scene metadata CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data_source" / "data" / "planet_imagery" / "generated" / "cities_scenes_results_planet.csv",
        help="Merged output CSV. Defaults to overwriting the main scene table.",
    )
    return parser.parse_args()


def load_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    table = pd.read_csv(path)
    missing = set(KEY_COLUMNS) - set(table.columns)
    if missing:
        raise ValueError(f"{label} is missing required columns: {sorted(missing)}")
    return table


def write_csv(path: Path, table: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    table.to_csv(temporary_path, index=False)
    temporary_path.replace(path)


def main() -> None:
    args = parse_args()
    main_scenes = load_csv(args.main_scenes, "main scene table")
    backfill_scenes = load_csv(args.backfill_scenes, "backfill scene table")

    main_columns = list(main_scenes.columns)
    backfill_extra = [column for column in backfill_scenes.columns if column not in main_columns]
    if backfill_extra:
        raise ValueError(
            "Backfill has columns not present in the main scene table: "
            f"{backfill_extra}"
        )

    combined = pd.concat(
        [main_scenes, backfill_scenes[main_columns]],
        ignore_index=True,
    )
    before_dedup = len(combined)
    combined = (
        combined.drop_duplicates(subset=KEY_COLUMNS, keep="first")
        .sort_values(["city_slug", "acquired", "id"])
        .reset_index(drop=True)
    )
    added_rows = len(combined) - len(main_scenes)
    duplicate_rows = before_dedup - len(combined)
    write_csv(args.output, combined)

    print(f"Main rows before merge: {len(main_scenes):,}")
    print(f"Backfill rows: {len(backfill_scenes):,}")
    print(f"New rows added: {added_rows:,}")
    print(f"Duplicate rows skipped: {duplicate_rows:,}")
    print(f"WROTE {args.output} ({len(combined):,} rows)")


if __name__ == "__main__":
    main()
