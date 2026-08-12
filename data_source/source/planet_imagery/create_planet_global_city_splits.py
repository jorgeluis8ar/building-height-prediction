"""
Create reproducible city-level train, validation, and test scene manifests.

Environment: data_source/source/planet_imagery/venv_planet_imagery

This is a local manifest-building script. It never authenticates with Planet,
activates assets, creates orders, or downloads imagery.

The default split uses seed 419453 and partitions the 1,779 cities represented
in the selected-scenes file into exactly:
    - 711 training cities;
    - 711 validation cities; and
    - 357 testing cities.

Every scene from one city inherits the same group. A SHA-256 score derived
from ``seed:city_slug`` creates a stable permutation that is reproducible
across Python, operating-system, and Pandas versions.

Default outputs:
    data_source/data/planet_imagery/generated/global_scene_selection_split/
        planet_city_split_manifest.csv
        planet_scene_split_manifest.csv
        training_scene_order_input.csv
        training_scene_ids.csv
        validation_scene_ids.csv
        testing_scene_ids.csv
        planet_split_summary.csv
        logs/create_planet_global_city_splits_<UTC timestamp>.log
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import sys
import traceback


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
VENV_DIR = SCRIPT_DIR / "venv_planet_imagery"
VENV_PYTHON = VENV_DIR / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
VENV_MARKER = "PLANET_IMAGERY_VENV_ACTIVE"

DEFAULT_SEED = 419453
DEFAULT_TRAINING_CITIES = 711
DEFAULT_VALIDATION_CITIES = 711
DEFAULT_TESTING_CITIES = 357
EXPECTED_REPRESENTED_CITIES = 1779
EXPECTED_SCENES_PER_COMPLETE_CITY = 9

REQUIRED_SCENE_COLUMNS = {
    "city_slug",
    "city_name",
    "country",
    "scene_id",
    "selection_rank",
    "selection_local_season",
    "selected_asset_type",
}


def relaunch_inside_venv() -> None:
    """Relaunch inside the task environment before importing Pandas."""
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


def parse_args() -> argparse.Namespace:
    """Define explicit, reviewable split parameters and portable paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selected-scenes",
        type=Path,
        default=PROJECT_ROOT
        / (
            "data_source/data/planet_imagery/generated/global_scene_selection/"
            "selected_global_planet_city_scenes.csv"
        ),
    )
    parser.add_argument(
        "--aoi-dir",
        type=Path,
        default=PROJECT_ROOT
        / "data_source/data/city_aois/generated/wup2018_city_buffers_5km_by_city",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT
        / "data_source/data/planet_imagery/generated/global_scene_selection_split",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--training-cities", type=int, default=DEFAULT_TRAINING_CITIES)
    parser.add_argument("--validation-cities", type=int, default=DEFAULT_VALIDATION_CITIES)
    parser.add_argument("--testing-cities", type=int, default=DEFAULT_TESTING_CITIES)
    return parser.parse_args()


def resolve_project_path(path: Path, *, output: bool = False) -> Path:
    """Resolve relative paths from the repository and reject outside paths."""
    resolved = path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()
    if not resolved.is_relative_to(PROJECT_ROOT):
        role = "Output" if output else "Input"
        raise ValueError(f"{role} path is outside the project repository: {resolved}")
    return resolved


def require_file(path: Path, label: str) -> None:
    """Fail before processing if a required input is absent or empty."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"Empty {label}: {path}")


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    """Publish a complete CSV atomically instead of leaving partial output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, quoting=csv.QUOTE_MINIMAL)
    temporary.replace(path)


def stable_city_score(seed: int, city_slug: str) -> str:
    """Return a portable deterministic randomization score for one city."""
    return hashlib.sha256(f"{seed}:{city_slug}".encode("utf-8")).hexdigest()


def assign_group(rank: int, training: int, validation: int) -> str:
    """Translate a one-based randomized rank into its disjoint split."""
    if rank <= training:
        return "training"
    if rank <= training + validation:
        return "validation"
    return "testing"


def create_manifests(
    scenes: pd.DataFrame,
    aoi_dir: Path,
    seed: int,
    training_count: int,
    validation_count: int,
    testing_count: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create city, scene, and reconciliation manifests from the source CSV."""
    missing = sorted(REQUIRED_SCENE_COLUMNS - set(scenes.columns))
    if missing:
        raise ValueError(f"Selected-scenes CSV is missing required columns: {missing}")
    if scenes.empty:
        raise ValueError("Selected-scenes CSV contains no rows")
    if scenes[["city_slug", "scene_id"]].duplicated().any():
        raise ValueError("Selected-scenes CSV has duplicate city_slug/scene_id rows")

    city_counts = (
        scenes.groupby(["city_slug", "city_name", "country"], as_index=False, dropna=False)
        .agg(selected_scene_count=("scene_id", "size"))
    )
    represented_count = len(city_counts)
    requested_total = training_count + validation_count + testing_count
    if represented_count != EXPECTED_REPRESENTED_CITIES:
        raise ValueError(
            f"Expected {EXPECTED_REPRESENTED_CITIES:,} represented cities but found "
            f"{represented_count:,}. Review the input before changing split counts."
        )
    if requested_total != represented_count:
        raise ValueError(
            f"Split counts sum to {requested_total:,}, but the input represents "
            f"{represented_count:,} cities"
        )
    if min(training_count, validation_count, testing_count) <= 0:
        raise ValueError("Every split count must be positive")

    city_counts["split_seed"] = seed
    city_counts["randomization_algorithm"] = "sha256(seed:city_slug)_ascending"
    city_counts["randomization_score_sha256"] = city_counts["city_slug"].map(
        lambda slug: stable_city_score(seed, str(slug))
    )
    city_counts = city_counts.sort_values(
        ["randomization_score_sha256", "city_slug"], kind="mergesort"
    ).reset_index(drop=True)
    city_counts["randomized_city_rank"] = city_counts.index + 1
    city_counts["split_group"] = city_counts["randomized_city_rank"].map(
        lambda rank: assign_group(int(rank), training_count, validation_count)
    )
    city_counts["has_exactly_nine_scenes"] = (
        city_counts["selected_scene_count"] == EXPECTED_SCENES_PER_COMPLETE_CITY
    )
    city_counts["scene_shortfall_count"] = (
        EXPECTED_SCENES_PER_COMPLETE_CITY - city_counts["selected_scene_count"]
    ).clip(lower=0)
    city_counts["aoi_path"] = city_counts["city_slug"].map(
        lambda slug: str(
            (aoi_dir / f"{slug}_5km.geojson").relative_to(PROJECT_ROOT)
        ).replace("\\", "/")
    )
    missing_aois = [
        slug
        for slug in city_counts["city_slug"]
        if not (aoi_dir / f"{slug}_5km.geojson").is_file()
    ]
    if missing_aois:
        raise FileNotFoundError(f"Missing AOIs for split cities: {missing_aois[:10]}")

    scene_manifest = scenes.merge(
        city_counts[
            [
                "city_slug", "split_seed", "randomization_algorithm",
                "randomization_score_sha256", "randomized_city_rank", "split_group",
                "selected_scene_count", "has_exactly_nine_scenes",
                "scene_shortfall_count", "aoi_path",
            ]
        ],
        on="city_slug",
        how="left",
        validate="many_to_one",
        suffixes=("", "_city_manifest"),
    )
    if scene_manifest["split_group"].isna().any():
        raise ValueError("At least one scene did not receive a city split")

    # Compatibility aliases allow a future global order script to consume the
    # file without discarding the canonical global-selection column names.
    scene_manifest["id"] = scene_manifest["scene_id"]
    scene_manifest["selection_season"] = scene_manifest["selection_local_season"]
    scene_manifest = scene_manifest.sort_values(
        ["randomized_city_rank", "selection_rank", "scene_id"], kind="mergesort"
    ).reset_index(drop=True)

    summary = (
        city_counts.groupby("split_group", as_index=False)
        .agg(
            city_count=("city_slug", "size"),
            scene_count=("selected_scene_count", "sum"),
            cities_with_nine_scenes=("has_exactly_nine_scenes", "sum"),
            total_scene_shortfall=("scene_shortfall_count", "sum"),
        )
    )
    summary["seed"] = seed
    order = pd.CategoricalDtype(["training", "validation", "testing"], ordered=True)
    summary["split_group"] = summary["split_group"].astype(order)
    summary = summary.sort_values("split_group").reset_index(drop=True)
    summary["split_group"] = summary["split_group"].astype("string")
    return city_counts, scene_manifest, summary


class Tee:
    """Mirror honest run messages to the terminal and a dated log."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = path.open("w", encoding="utf-8")

    def write(self, message: str) -> None:
        print(message, flush=True)
        self._file.write(message + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()


def main() -> int:
    """Build all manifests and fail loudly if any reconciliation is false."""
    args = parse_args()
    scenes_path = resolve_project_path(args.selected_scenes)
    aoi_dir = resolve_project_path(args.aoi_dir)
    output_dir = resolve_project_path(args.output_dir, output=True)
    require_file(scenes_path, "combined selected-scenes CSV")
    if not aoi_dir.is_dir():
        raise FileNotFoundError(f"Missing global AOI directory: {aoi_dir}")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    logger = Tee(output_dir / "logs" / f"create_planet_global_city_splits_{timestamp}.log")
    try:
        logger.write("Planet global city split started")
        logger.write(f"Seed: {args.seed}")
        logger.write(
            f"Requested city counts: training={args.training_cities}, "
            f"validation={args.validation_cities}, testing={args.testing_cities}"
        )
        scenes = pd.read_csv(
            scenes_path,
            dtype={"city_slug": str, "scene_id": str, "strip_id": str},
        )
        city_manifest, scene_manifest, summary = create_manifests(
            scenes,
            aoi_dir,
            args.seed,
            args.training_cities,
            args.validation_cities,
            args.testing_cities,
        )

        atomic_write_csv(city_manifest, output_dir / "planet_city_split_manifest.csv")
        atomic_write_csv(scene_manifest, output_dir / "planet_scene_split_manifest.csv")
        atomic_write_csv(summary, output_dir / "planet_split_summary.csv")

        id_columns = [
            "split_group", "city_slug", "city_name", "country", "scene_id",
            "selection_rank", "selection_local_season", "selected_asset_type",
            "aoi_path", "split_seed", "randomized_city_rank",
            "has_exactly_nine_scenes", "scene_shortfall_count",
        ]
        for split_group in ["training", "validation", "testing"]:
            split_ids = scene_manifest.loc[scene_manifest["split_group"] == split_group, id_columns]
            atomic_write_csv(split_ids, output_dir / f"{split_group}_scene_ids.csv")
            logger.write(
                f"{split_group}: {split_ids['city_slug'].nunique():,} cities, "
                f"{len(split_ids):,} scene rows"
            )

        training_order = scene_manifest[scene_manifest["split_group"] == "training"].copy()
        atomic_write_csv(training_order, output_dir / "training_scene_order_input.csv")

        if len(scene_manifest) != len(scenes):
            raise AssertionError("Scene manifest row count does not match source")
        if scene_manifest[["city_slug", "scene_id"]].duplicated().any():
            raise AssertionError("Scene manifest introduced duplicate city/scene rows")
        observed_counts = city_manifest["split_group"].value_counts().to_dict()
        expected_counts = {
            "training": args.training_cities,
            "validation": args.validation_cities,
            "testing": args.testing_cities,
        }
        if observed_counts != expected_counts:
            raise AssertionError(
                f"Observed split counts {observed_counts} do not equal {expected_counts}"
            )
        if scene_manifest.groupby("city_slug")["split_group"].nunique().max() != 1:
            raise AssertionError("City leakage detected across split groups")

        logger.write(
            f"Cities with exactly nine scenes: "
            f"{int(city_manifest['has_exactly_nine_scenes'].sum()):,}"
        )
        logger.write(
            f"Cities with scene shortfalls retained and flagged: "
            f"{int((~city_manifest['has_exactly_nine_scenes']).sum()):,}"
        )
        logger.write(f"Wrote outputs under: {output_dir.relative_to(PROJECT_ROOT)}")
        logger.write("SUCCESS: every represented city and selected scene was assigned exactly once")
        return 0
    except Exception:
        logger.write("FAILED: split outputs are incomplete or invalid")
        logger.write(traceback.format_exc())
        return 1
    finally:
        logger.close()


if __name__ == "__main__":
    raise SystemExit(main())
