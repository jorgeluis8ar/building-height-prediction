#!/usr/bin/env python3
"""Combine all per-city Planet metadata CSV files into one global CSV.

The script is local and API-free. It does not activate, order, or download any
Planet asset. It reads a city inventory to establish the expected file set,
validates every city file, and writes the combined output atomically so an
interrupted run can never look complete. The default inventory contains all
1,862 cities; an explicit expected count supports validated subsets.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
import sys
import traceback


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INVENTORY = REPOSITORY_ROOT / (
    "data_source/data/city_aois/generated/wup2018_cities_over_300k_2018.csv"
)
DEFAULT_INPUT_DIRECTORY = REPOSITORY_ROOT / (
    "data_source/data/planet_imagery/generated/global_city_scene_metadata/by_city"
)
DEFAULT_OUTPUT = REPOSITORY_ROOT / (
    "data_source/data/planet_imagery/generated/global_city_scene_metadata/"
    "all_global_planet_city_scene_metadata.csv"
)
DEFAULT_LOG_DIRECTORY = REPOSITORY_ROOT / (
    "data_source/data/planet_imagery/generated/logs"
)
DEFAULT_EXPECTED_CITY_COUNT = 1_862
PROVENANCE_FIELD = "source_metadata_file"


def parse_arguments() -> argparse.Namespace:
    """Parse optional paths and the explicit partial-run override."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--input-directory", type=Path, default=DEFAULT_INPUT_DIRECTORY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--expected-city-count",
        type=int,
        default=DEFAULT_EXPECTED_CITY_COUNT,
        help=(
            "Required number of unique cities in --inventory. Use 94 with the "
            "training/open-LiDAR city list."
        ),
    )
    parser.add_argument(
        "--allow-missing-city-files",
        action="store_true",
        help=(
            "Create an explicitly partial combined file when one or more "
            "expected city files are absent. The default is to fail."
        ),
    )
    return parser.parse_args()


def resolve_project_path(path: Path) -> Path:
    """Resolve paths from the repository and reject paths outside it."""
    resolved = path.resolve() if path.is_absolute() else (REPOSITORY_ROOT / path).resolve()
    if not resolved.is_relative_to(REPOSITORY_ROOT):
        raise ValueError(f"Path is outside the project repository: {resolved}")
    return resolved


def read_inventory(path: Path, expected_city_count: int) -> list[dict[str, str]]:
    """Read the authoritative city order and reject malformed inventories."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing city inventory: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"wup_urbancode", "city_slug", "city_name", "country"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise RuntimeError(f"Inventory is missing columns: {sorted(missing)}")
        rows = list(reader)
    slugs = [row["city_slug"].strip() for row in rows]
    if len(rows) != expected_city_count:
        raise RuntimeError(
            f"Expected {expected_city_count:,} inventory cities, found {len(rows):,}"
        )
    if any(not slug for slug in slugs) or len(slugs) != len(set(slugs)):
        raise RuntimeError("Inventory contains blank or duplicate city_slug values")
    return rows


def inspect_city_file(path: Path, expected_slug: str) -> tuple[list[str], int]:
    """Validate one file and return its header plus row count.

    Each scene ID must be unique within a city. The same Planet scene may
    legitimately appear in different city files, so uniqueness is assessed on
    the city/scene pair rather than globally on scene_id alone.
    """
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        if not fields or len(fields) != len(set(fields)):
            raise RuntimeError(f"Missing or duplicate CSV headers: {path}")
        required = {"city_slug", "scene_id", "acquired"}
        missing = required.difference(fields)
        if missing:
            raise RuntimeError(f"{path.name} is missing columns: {sorted(missing)}")
        scene_ids: set[str] = set()
        count = 0
        for line_number, row in enumerate(reader, start=2):
            slug = str(row.get("city_slug", "")).strip()
            scene_id = str(row.get("scene_id", "")).strip()
            if slug != expected_slug:
                raise RuntimeError(
                    f"{path.name}:{line_number} has city_slug={slug!r}; "
                    f"expected {expected_slug!r}"
                )
            if not scene_id:
                raise RuntimeError(f"{path.name}:{line_number} has a blank scene_id")
            if scene_id in scene_ids:
                raise RuntimeError(
                    f"{path.name}:{line_number} duplicates scene_id {scene_id}"
                )
            scene_ids.add(scene_id)
            count += 1
    return fields, count


def write_combined_csv(
    files: list[tuple[str, Path]],
    output_path: Path,
    output_fields: list[str],
) -> int:
    """Stream all records to an atomic temporary file without loading them all."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".partial")
    row_count = 0
    try:
        with temporary.open("w", encoding="utf-8", newline="") as output_handle:
            writer = csv.DictWriter(
                output_handle,
                fieldnames=output_fields,
                extrasaction="raise",
            )
            writer.writeheader()
            for expected_slug, path in files:
                with path.open("r", encoding="utf-8-sig", newline="") as input_handle:
                    reader = csv.DictReader(input_handle)
                    for row in reader:
                        # Validation already checked the slug and scene ID. The
                        # provenance column makes every combined row traceable.
                        row[PROVENANCE_FIELD] = path.name
                        writer.writerow(row)
                        row_count += 1
        temporary.replace(output_path)
    except BaseException:
        # A .partial file is visibly incomplete. Leave it for diagnosis rather
        # than replacing the last known-good combined output.
        raise
    return row_count


def main() -> None:
    """Validate the full file set, combine it, and write an honest run log."""
    arguments = parse_arguments()
    started = datetime.now(timezone.utc)
    log_path = DEFAULT_LOG_DIRECTORY / (
        f"combine_planet_global_city_scene_metadata_"
        f"{started.strftime('%Y%m%dT%H%M%SZ')}.log"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_lines = ["status=RUNNING", f"started_utc={started.isoformat()}"]
    try:
        inventory_path = resolve_project_path(arguments.inventory)
        input_directory = resolve_project_path(arguments.input_directory)
        output_path = resolve_project_path(arguments.output)
        if not input_directory.is_dir():
            raise FileNotFoundError(f"Missing metadata directory: {input_directory}")
        if output_path.parent == input_directory and output_path.name.endswith("_planet_scenes.csv"):
            raise ValueError(
                "The combined output name must not match the per-city *_planet_scenes.csv pattern"
            )

        if arguments.expected_city_count < 1:
            raise ValueError("expected-city-count must be at least one")
        inventory = read_inventory(inventory_path, arguments.expected_city_count)
        expected_files = [
            (row["city_slug"], input_directory / f"{row['city_slug']}_planet_scenes.csv")
            for row in inventory
        ]
        missing_files = [(slug, path) for slug, path in expected_files if not path.is_file()]
        if missing_files and not arguments.allow_missing_city_files:
            raise FileNotFoundError(
                f"Missing {len(missing_files):,} of {arguments.expected_city_count:,} expected city files. "
                f"Examples: {[slug for slug, _ in missing_files[:20]]}. "
                "Rerun the missing API queries or pass --allow-missing-city-files "
                "to create an explicitly partial combined file."
            )
        files = [(slug, path) for slug, path in expected_files if path.is_file()]
        if not files:
            raise RuntimeError("No per-city metadata files are available to combine")

        # First pass: validate all inputs and build a stable union of headers.
        # The first file's order is retained; fields seen only in later files
        # are appended in their original order.
        output_fields: list[str] = []
        expected_row_count = 0
        empty_city_count = 0
        for number, (slug, path) in enumerate(files, start=1):
            fields, row_count = inspect_city_file(path, slug)
            for field in fields:
                if field not in output_fields:
                    output_fields.append(field)
            expected_row_count += row_count
            empty_city_count += int(row_count == 0)
            if number % 100 == 0 or number == len(files):
                print(
                    f"Validated {number:,}/{len(files):,} city files; "
                    f"scene rows so far={expected_row_count:,}",
                    flush=True,
                )
        if PROVENANCE_FIELD in output_fields:
            raise RuntimeError(
                f"Input unexpectedly already contains reserved field {PROVENANCE_FIELD}"
            )
        output_fields.append(PROVENANCE_FIELD)

        print(
            f"Writing {expected_row_count:,} scene rows from {len(files):,} city files...",
            flush=True,
        )
        written_row_count = write_combined_csv(files, output_path, output_fields)
        if written_row_count != expected_row_count:
            raise RuntimeError(
                f"Row-count mismatch: validated {expected_row_count:,}, "
                f"wrote {written_row_count:,}"
            )

        completed = datetime.now(timezone.utc)
        completeness = "PARTIAL" if missing_files else "COMPLETE"
        log_lines[0] = "status=SUCCESS"
        log_lines.extend([
            f"completed_utc={completed.isoformat()}",
            f"completeness={completeness}",
            f"inventory_city_count={len(inventory)}",
            f"expected_city_count={arguments.expected_city_count}",
            f"combined_city_file_count={len(files)}",
            f"missing_city_file_count={len(missing_files)}",
            f"empty_city_file_count={empty_city_count}",
            f"combined_scene_row_count={written_row_count}",
            f"combined_column_count={len(output_fields)}",
            f"output={output_path.relative_to(REPOSITORY_ROOT)}",
        ])
        log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
        print(f"SUCCESS ({completeness}): {output_path}", flush=True)
        print(f"Cities combined: {len(files):,}; scene rows: {written_row_count:,}", flush=True)
        print(f"Log: {log_path}", flush=True)
    except BaseException:
        log_lines[0] = "status=FAILED"
        log_lines.append(traceback.format_exc())
        log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
        print(f"FAILED: see {log_path}", file=sys.stderr, flush=True)
        raise


if __name__ == "__main__":
    main()
