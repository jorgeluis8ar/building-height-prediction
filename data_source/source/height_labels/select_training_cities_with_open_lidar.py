#!/usr/bin/env python3
"""Select training cities whose open-LiDAR status is ready for download.

This is a local CSV join. It does not query a remote service and does not
download LiDAR, imagery, DSM, or DTM data.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LIDAR_INVENTORY = REPOSITORY_ROOT / "data_source/data/height_labels/generated/global_open_lidar_city_inventory.csv"
DEFAULT_SPLIT_MANIFEST = REPOSITORY_ROOT / "data_source/data/planet_imagery/generated/global_scene_selection_split/planet_city_split_manifest.csv"
DEFAULT_OUTPUT_DIRECTORY = REPOSITORY_ROOT / "data_source/data/height_labels/generated/training_open_lidar"
EXPECTED_READY_CITIES = 206
EXPECTED_TRAINING_CITIES = 711

OUTPUT_FIELDS = [
    "wup_urbancode", "city_slug", "city_name", "country",
    "centroid_latitude", "centroid_longitude", "lidar_coverage_status",
    "lidar_aoi_coverage_percent", "lidar_source_program",
    "lidar_official_access_link", "lidar_product_type", "lidar_format",
    "lidar_license", "lidar_acquisition_year", "lidar_download_file_count",
    "lidar_download_endpoint_status", "lidar_representative_download_url",
    "lidar_access_constraint", "lidar_verification_date",
    "lidar_verification_method", "split_group", "split_seed",
    "randomization_algorithm", "randomization_score_sha256",
    "randomized_city_rank", "selected_scene_count",
    "has_exactly_nine_scenes", "aoi_path",
]


def parse_arguments() -> argparse.Namespace:
    """Parse optional input and output overrides."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lidar-inventory", type=Path, default=DEFAULT_LIDAR_INVENTORY)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT_MANIFEST)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument(
        "--allow-count-changes", action="store_true",
        help="Allow source pools other than the expected 206 ready and 711 training cities.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a required CSV and reject missing or malformed input."""
    if not path.is_file():
        raise FileNotFoundError(f"Required input does not exist: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise RuntimeError(f"Invalid or duplicate CSV headers: {path}")
        rows = list(reader)
    if not rows:
        raise RuntimeError(f"Required input is empty: {path}")
    return rows


def require_fields(rows: list[dict[str, str]], fields: set[str], path: Path) -> None:
    """Fail with a useful message when an expected field is absent."""
    missing = fields - set(rows[0])
    if missing:
        raise RuntimeError(f"Missing fields in {path}: {', '.join(sorted(missing))}")


def unique_by_slug(rows: list[dict[str, str]], label: str) -> dict[str, dict[str, str]]:
    """Index city records by stable slug and reject duplicate records."""
    result = {}
    for row in rows:
        slug = row["city_slug"].strip()
        if not slug:
            raise RuntimeError(f"Blank city_slug in {label}")
        if slug in result:
            raise RuntimeError(f"Duplicate city_slug in {label}: {slug}")
        result[slug] = row
    return result


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    """Write atomically so interrupted runs never resemble complete outputs."""
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> None:
    """Join the LiDAR inventory to the PlanetScope training-city manifest."""
    arguments = parse_arguments()
    lidar_rows = read_csv(arguments.lidar_inventory)
    split_rows = read_csv(arguments.split_manifest)
    require_fields(
        lidar_rows,
        {"wup_urbancode", "city_slug", "city_name", "country", "coverage_status",
         "aoi_coverage_percent", "source_program", "official_access_link",
         "product_type", "format", "license", "acquisition_year",
         "verification_date", "verification_method"},
        arguments.lidar_inventory,
    )
    require_fields(
        split_rows,
        {"city_slug", "split_group", "split_seed", "randomization_algorithm",
         "randomization_score_sha256", "randomized_city_rank",
         "selected_scene_count", "has_exactly_nine_scenes", "aoi_path"},
        arguments.split_manifest,
    )

    ready_by_slug = unique_by_slug(
        [row for row in lidar_rows if row["coverage_status"] == "ready_for_download"],
        "ready LiDAR inventory",
    )
    training_by_slug = unique_by_slug(
        [row for row in split_rows if row["split_group"] == "training"],
        "training split",
    )
    if not arguments.allow_count_changes:
        if len(ready_by_slug) != EXPECTED_READY_CITIES:
            raise RuntimeError(
                f"Expected {EXPECTED_READY_CITIES} ready LiDAR cities, found "
                f"{len(ready_by_slug)}; inspect the inventory or pass --allow-count-changes"
            )
        if len(training_by_slug) != EXPECTED_TRAINING_CITIES:
            raise RuntimeError(
                f"Expected {EXPECTED_TRAINING_CITIES} training cities, found "
                f"{len(training_by_slug)}; inspect the split or pass --allow-count-changes"
            )

    overlap_slugs = sorted(
        ready_by_slug.keys() & training_by_slug.keys(),
        key=lambda slug: int(training_by_slug[slug]["randomized_city_rank"]),
    )
    output_rows = []
    for slug in overlap_slugs:
        lidar = ready_by_slug[slug]
        split = training_by_slug[slug]
        output_rows.append({
            "wup_urbancode": lidar["wup_urbancode"], "city_slug": slug,
            "city_name": lidar["city_name"], "country": lidar["country"],
            "centroid_latitude": lidar.get("centroid_latitude", ""),
            "centroid_longitude": lidar.get("centroid_longitude", ""),
            "lidar_coverage_status": lidar["coverage_status"],
            "lidar_aoi_coverage_percent": lidar["aoi_coverage_percent"],
            "lidar_source_program": lidar["source_program"],
            "lidar_official_access_link": lidar["official_access_link"],
            "lidar_product_type": lidar["product_type"],
            "lidar_format": lidar["format"], "lidar_license": lidar["license"],
            "lidar_acquisition_year": lidar["acquisition_year"],
            "lidar_download_file_count": lidar.get("download_file_count", ""),
            "lidar_download_endpoint_status": lidar.get("download_endpoint_status", ""),
            "lidar_representative_download_url": lidar.get("representative_download_url", ""),
            "lidar_access_constraint": lidar.get("access_constraint", ""),
            "lidar_verification_date": lidar["verification_date"],
            "lidar_verification_method": lidar["verification_method"],
            "split_group": split["split_group"], "split_seed": split["split_seed"],
            "randomization_algorithm": split["randomization_algorithm"],
            "randomization_score_sha256": split["randomization_score_sha256"],
            "randomized_city_rank": split["randomized_city_rank"],
            "selected_scene_count": split["selected_scene_count"],
            "has_exactly_nine_scenes": split["has_exactly_nine_scenes"],
            "aoi_path": split["aoi_path"],
        })

    country_counts = Counter(row["country"] for row in output_rows)
    summary_rows = [
        {"country": country, "training_cities_ready_for_lidar": str(count)}
        for country, count in sorted(country_counts.items())
    ]
    summary_rows.append({
        "country": "TOTAL",
        "training_cities_ready_for_lidar": str(len(output_rows)),
    })

    arguments.output_directory.mkdir(parents=True, exist_ok=True)
    city_output = arguments.output_directory / "training_cities_with_open_lidar.csv"
    summary_output = arguments.output_directory / "training_cities_with_open_lidar_by_country.csv"
    write_csv(city_output, OUTPUT_FIELDS, output_rows)
    write_csv(summary_output, ["country", "training_cities_ready_for_lidar"], summary_rows)
    print(f"Ready LiDAR cities: {len(ready_by_slug):,}")
    print(f"Training cities: {len(training_by_slug):,}")
    print(f"Overlap: {len(output_rows):,}")
    print(f"City output: {city_output.relative_to(REPOSITORY_ROOT)}")
    print(f"Country summary: {summary_output.relative_to(REPOSITORY_ROOT)}")


if __name__ == "__main__":
    main()
