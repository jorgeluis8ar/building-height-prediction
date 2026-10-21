#!/usr/bin/env python3
"""Prepare scene and split manifests for the versioned US44-to-US50 extension."""

from __future__ import annotations

import argparse
from pathlib import Path
import os

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
NEW_ASSIGNMENTS = {
    "new_york_newark_23083": "train",
    "miami_23064": "train",
    "philadelphia_23098": "train",
    "cincinnati_22957": "val",
    "harrisburg_23008": "val",
    "trenton_23165": "val",
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--existing-scene-metadata", type=Path, required=True,
        help="Existing 44-city selected scene metadata CSV.",
    )
    parser.add_argument(
        "--new-scene-metadata", type=Path,
        default=Path(
            "data_source/data/planet_imagery/generated/"
            "six_us_lidar_postyear_scene_selection_v1/"
            "selected_six_us_lidar_postyear_planet_scenes.csv"
        ),
    )
    parser.add_argument(
        "--existing-city-split", type=Path, required=True,
        help="Validated US44 city_split_manifest.csv.",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(
            "data_source/data/ml_models/generated/htc_dc_net/"
            "us_50_extension_inputs_v1"
        ),
    )
    return parser.parse_args()


def project_path(path: Path) -> Path:
    resolved = path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()
    resolved.relative_to(PROJECT_ROOT)
    return resolved


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(str(temporary), str(path))


def main() -> None:
    args = arguments()
    existing_scenes = pd.read_csv(project_path(args.existing_scene_metadata), low_memory=False)
    new_scenes = pd.read_csv(project_path(args.new_scene_metadata), low_memory=False)
    existing_split = pd.read_csv(project_path(args.existing_city_split), dtype=str).fillna("")
    output_dir = project_path(args.output_dir)

    for label, frame in (("existing scenes", existing_scenes), ("new scenes", new_scenes)):
        required = {"city_slug", "scene_id"}
        missing = required - set(frame.columns)
        if missing:
            raise RuntimeError(f"{label} missing columns: {sorted(missing)}")
    if len(set(existing_scenes["city_slug"])) != 44:
        raise RuntimeError("Existing scene metadata must contain exactly 44 cities")
    if set(new_scenes["city_slug"]) != set(NEW_ASSIGNMENTS):
        raise RuntimeError("New scene metadata does not contain exactly the approved six cities")
    if (new_scenes.groupby("city_slug")["scene_id"].nunique() != 8).any():
        raise RuntimeError("Every added city must contain exactly eight unique scenes")
    if set(existing_scenes["city_slug"]) & set(NEW_ASSIGNMENTS):
        raise RuntimeError("An added city is already present in the US44 scene metadata")
    if set(existing_split["city_slug"]) != set(existing_scenes["city_slug"]):
        raise RuntimeError("Existing US44 scene and split city sets differ")
    if existing_split["city_slug"].duplicated().any():
        raise RuntimeError("Existing US44 split contains duplicate cities")

    # Preserve the original 44 assignments exactly and append explicit new ones.
    split_columns = list(existing_split.columns)
    additions = []
    for rank, (city, split) in enumerate(NEW_ASSIGNMENTS.items(), start=len(existing_split) + 1):
        row = {column: "" for column in split_columns}
        row.update(city_slug=city, split=split)
        if "randomized_rank" in row:
            row["randomized_rank"] = rank
        if "seed" in row:
            row["seed"] = "explicit_us50_extension_v1"
        additions.append(row)
    combined_split = pd.concat([existing_split, pd.DataFrame(additions)], ignore_index=True)
    counts = combined_split["split"].value_counts().to_dict()
    if counts != {"train": 22, "val": 22, "test": 6}:
        raise RuntimeError(f"Unexpected US50 split counts: {counts}")

    combined_scenes = pd.concat([existing_scenes, new_scenes], ignore_index=True, sort=False)
    if combined_scenes[["city_slug", "scene_id"]].duplicated().any():
        raise RuntimeError("Combined scene metadata contains duplicate city/scene rows")
    if len(set(combined_scenes["city_slug"])) != 50:
        raise RuntimeError("Combined scene metadata must contain exactly 50 cities")

    atomic_csv(combined_scenes, output_dir / "selected_processed_us_lidar_planet_scenes_us50.csv")
    atomic_csv(combined_split, output_dir / "city_split_manifest_us50.csv")
    atomic_csv(
        pd.DataFrame(
            [{"city_slug": city, "split": split, "assignment_rule": "explicit_user_assignment"}
             for city, split in NEW_ASSIGNMENTS.items()]
        ),
        output_dir / "six_city_extension_assignments.csv",
    )
    print(f"WROTE {output_dir} with 50 cities and split counts {counts}")


if __name__ == "__main__":
    main()
