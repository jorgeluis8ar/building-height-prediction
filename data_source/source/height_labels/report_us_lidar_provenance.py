#!/usr/bin/env python3
"""Create reproducible city- and project-level US LiDAR provenance reports."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_NDSM_MANIFEST = Path(
    "data_source/data/height_labels/generated/us_training_planet_ndsm/"
    "us_lidar_to_planet_ndsm_manifest.csv"
)
DEFAULT_CITY_UNIVERSE = Path(
    "data_source/data/planet_imagery/generated/processed_us_lidar_scene_selection/"
    "selected_processed_us_lidar_planet_scenes.csv"
)
DEFAULT_OUTPUT_DIR = Path(
    "data_source/data/height_labels/generated/us_training_planet_ndsm/"
    "provenance/us44_v1"
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ndsm-manifest", type=Path, default=DEFAULT_NDSM_MANIFEST)
    parser.add_argument("--city-universe", type=Path, default=DEFAULT_CITY_UNIVERSE)
    parser.add_argument("--expected-cities", type=int, default=44)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Do not refresh official per-project dates and CRS from USGS metadata.",
    )
    return parser.parse_args()


def project_path(path: Path) -> Path:
    resolved = path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()
    if not resolved.is_relative_to(PROJECT_ROOT):
        raise ValueError(f"Path must remain inside repository: {resolved}")
    return resolved


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def atomic_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def require_columns(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = columns - set(frame.columns)
    if missing:
        raise RuntimeError(f"{label} is missing columns: {sorted(missing)}")


def main() -> None:
    args = arguments()
    manifest_path = project_path(args.ndsm_manifest)
    universe_path = project_path(args.city_universe)
    output_dir = project_path(args.output_dir)

    manifest = pd.read_csv(manifest_path, dtype=str).fillna("")
    universe = pd.read_csv(universe_path, dtype=str).fillna("")
    require_columns(
        manifest,
        {
            "city_slug", "city_name", "status", "selected_project_directories",
            "lidar_collect_start", "lidar_collect_end", "lidar_date_source",
            "project_selection_basis", "planned_tile_count", "downloaded_tile_count",
            "downloaded_bytes", "output_ndsm_path", "lidar_deleted", "footprint_path",
            "planet_template_path", "error_message",
        },
        "nDSM manifest",
    )
    require_columns(universe, {"city_slug"}, "city universe")
    cities = sorted(set(universe["city_slug"]))
    if len(cities) != args.expected_cities:
        raise RuntimeError(
            f"Expected {args.expected_cities} cities in authoritative universe; found {len(cities)}"
        )
    selected = manifest[manifest["city_slug"].isin(cities)].copy()
    if len(selected) != args.expected_cities or selected["city_slug"].duplicated().any():
        raise RuntimeError(
            f"Expected one nDSM manifest row for each of {args.expected_cities} cities; "
            f"found {len(selected)} rows"
        )
    if set(selected["city_slug"]) != set(cities):
        missing = sorted(set(cities) - set(selected["city_slug"]))
        raise RuntimeError(f"nDSM manifest is missing authoritative cities: {missing}")

    project_metadata: dict[str, dict[str, str]] = {}
    metadata_status = "offline_not_requested"
    if not args.offline:
        sys.path.insert(0, str(SCRIPT_DIR))
        try:
            from run_us_lidar_to_planet_ndsm import query_project_dates

            project_metadata = query_project_dates()
            metadata_status = "official_usgs_metadata_refreshed"
        except Exception as exc:
            metadata_status = f"official_usgs_metadata_failed:{type(exc).__name__}:{exc}"
        finally:
            sys.path.remove(str(SCRIPT_DIR))

    city_columns = [
        "city_slug", "city_name", "status", "selected_project_directories",
        "lidar_collect_start", "lidar_collect_end", "lidar_date_source",
        "project_selection_basis", "planned_tile_count", "downloaded_tile_count",
        "downloaded_bytes", "output_ndsm_path", "lidar_deleted", "footprint_path",
        "planet_template_path", "error_message",
    ]
    city_report = selected[city_columns].sort_values("city_slug").copy()
    city_report.insert(7, "lidar_acquisition_years", city_report.apply(
        lambda row: ";".join(sorted({
            str(row["lidar_collect_start"])[:4], str(row["lidar_collect_end"])[:4]
        } - {""})),
        axis=1,
    ))
    city_report["raw_lidar_size_gb"] = pd.to_numeric(
        city_report["downloaded_bytes"], errors="coerce"
    ).fillna(0).div(1024 ** 3).round(3)

    project_rows = []
    for row in city_report.itertuples(index=False):
        projects = [value.strip() for value in row.selected_project_directories.split(";") if value.strip()]
        for project in projects:
            official = project_metadata.get(project, {})
            project_rows.append(
                {
                    "city_slug": row.city_slug,
                    "city_name": row.city_name,
                    "project_key": project,
                    "project_collect_start": official.get("start", ""),
                    "project_collect_end": official.get("end", ""),
                    "horizontal_crs_epsg": official.get("horizontal_crs", ""),
                    "city_selected_collect_start": row.lidar_collect_start,
                    "city_selected_collect_end": row.lidar_collect_end,
                    "date_source": row.lidar_date_source,
                    "selection_basis": row.project_selection_basis,
                    "metadata_refresh_status": metadata_status,
                }
            )
    project_report = pd.DataFrame(project_rows).sort_values(["city_slug", "project_key"])

    atomic_csv(city_report, output_dir / "us44_lidar_city_provenance.csv")
    atomic_csv(project_report, output_dir / "us44_lidar_project_provenance.csv")
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "expected_cities": args.expected_cities,
        "city_count": len(city_report),
        "project_count": len(project_report),
        "metadata_refresh_status": metadata_status,
        "ndsm_manifest": str(manifest_path.relative_to(PROJECT_ROOT)),
        "city_universe": str(universe_path.relative_to(PROJECT_ROOT)),
        "status_counts": city_report["status"].value_counts().to_dict(),
    }
    atomic_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", output_dir / "provenance_summary.json")
    atomic_text(
        "# US44 LiDAR Provenance\n\n"
        "This directory records the selected USGS LiDAR projects, acquisition dates, "
        "CRS metadata, processing status, and generated nDSM path for the authoritative "
        "44-city HTC dataset. Regenerate it with `report_us_lidar_provenance.py`; do not "
        "edit generated rows manually.\n",
        output_dir / "README.md",
    )
    print(f"WROTE {output_dir}")
    print(f"Cities: {len(city_report)}; city-project rows: {len(project_report)}")
    print(f"USGS metadata: {metadata_status}")


if __name__ == "__main__":
    main()
