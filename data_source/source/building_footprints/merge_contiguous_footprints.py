"""
Merge Contiguous Building Footprint Pieces

Environment: data_source/source/building_footprints/venv_building_footprints

Requires:
    - data_source/data/building_footprints/generated/<city_slug>/
      <city_slug>_building_footprints_5km.gpkg

Produces:
    - data_source/data/building_footprints/generated/<city_slug>/
      <city_slug>_building_footprints_merged_5km.gpkg
    - data_source/data/building_footprints/generated/<city_slug>/
      <city_slug>_footprint_merge_crosswalk.csv
    - data_source/data/building_footprints/generated/<city_slug>/
      <city_slug>_footprint_merge_diagnostics.csv
    - data_source/data/building_footprints/generated/<city_slug>/
      <city_slug>_footprint_merge_review_sample.gpkg
    - data_source/data/building_footprints/generated/
      building_footprints_merge_summary.csv

Description:
    Conservatively identifies clusters of adjacent footprint polygons that may
    represent one physical building split into multiple pieces. The default
    rule merges polygons that share a meaningful boundary segment and, when a
    city source building identifier exists, have the same source building ID.
    It does not merge corner-only contacts, and tiny-gap merging is opt-in.

Usage:
    python3 data_source/source/building_footprints/merge_contiguous_footprints.py \
      --city new_york_city --city los_angeles
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
VENV_DIR = SCRIPT_DIR / "venv_building_footprints"
VENV_PYTHON = VENV_DIR / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
VENV_MARKER = "BUILDING_FOOTPRINTS_VENV_ACTIVE"

INPUT_SUFFIX = "_building_footprints_5km.gpkg"
OUTPUT_SUFFIX = "_building_footprints_merged_5km.gpkg"
OUTPUT_LAYER = "building_footprints_merged_5km"
OUTPUT_DRIVER = "GPKG"


def relaunch_inside_venv() -> None:
    """Relaunch this script with the building-footprints virtual environment."""
    if os.environ.get(VENV_MARKER) == "1":
        return

    current_python = Path(sys.executable).absolute()
    expected_python = VENV_PYTHON.absolute()
    if current_python == expected_python or Path(sys.prefix).absolute() == VENV_DIR.absolute():
        os.environ[VENV_MARKER] = "1"
        return

    if not VENV_PYTHON.exists():
        print("ERROR: Missing building_footprints virtual environment.")
        print(f"Expected Python executable: {VENV_PYTHON}")
        print("Create it with:")
        print("  python3 -m venv data_source/source/building_footprints/venv_building_footprints")
        print(
            "  data_source/source/building_footprints/venv_building_footprints/bin/python "
            "-m pip install -r data_source/source/building_footprints/requirements.txt"
        )
        sys.exit(1)

    environment = os.environ.copy()
    environment[VENV_MARKER] = "1"
    os.execve(
        str(VENV_PYTHON),
        [str(VENV_PYTHON), str(Path(__file__).resolve())] + sys.argv[1:],
        environment,
    )


relaunch_inside_venv()

import geopandas as gpd
import pandas as pd
from shapely import make_valid
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union


@dataclass
class DisjointSet:
    """Union-find structure for footprint adjacency components."""

    parent: list[int]
    rank: list[int]

    @classmethod
    def create(cls, size: int) -> "DisjointSet":
        return cls(parent=list(range(size)), rank=[0] * size)

    def find(self, item: int) -> int:
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, left: int, right: int) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return
        if self.rank[root_left] < self.rank[root_right]:
            self.parent[root_left] = root_right
        elif self.rank[root_left] > self.rank[root_right]:
            self.parent[root_right] = root_left
        else:
            self.parent[root_right] = root_left
            self.rank[root_left] += 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Conservatively merge contiguous building footprint polygons into "
            "candidate physical-building footprints."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=PROJECT_ROOT / "data_source" / "data" / "building_footprints" / "generated",
        help="Generated city footprint folder.",
    )
    parser.add_argument(
        "--city",
        action="append",
        dest="cities",
        help="Limit to one city slug. May be supplied more than once.",
    )
    parser.add_argument(
        "--local-crs",
        type=int,
        default=None,
        help="Optional projected EPSG code. By default, a UTM CRS is estimated per city.",
    )
    parser.add_argument(
        "--min-shared-boundary-m",
        type=float,
        default=0.5,
        help="Minimum shared boundary length required for a touch merge. Default: 0.5.",
    )
    parser.add_argument(
        "--min-shared-boundary-ratio",
        type=float,
        default=0.02,
        help=(
            "Minimum shared boundary divided by the smaller polygon perimeter. "
            "Default: 0.02."
        ),
    )
    parser.add_argument(
        "--gap-tolerance-m",
        type=float,
        default=0.0,
        help=(
            "Optional tiny-gap tolerance. Keep 0.0 for the conservative default. "
            "Example diagnostic value: 0.25."
        ),
    )
    parser.add_argument(
        "--min-gap-near-boundary-m",
        type=float,
        default=1.0,
        help=(
            "When --gap-tolerance-m is positive, require the buffered polygons "
            "to have this much near-boundary overlap. Default: 1.0."
        ),
    )
    parser.add_argument(
        "--same-height-tolerance-m",
        type=float,
        default=None,
        help=(
            "Optional official-height similarity gate. If supplied, candidate "
            "pairs with both official heights present must differ by no more "
            "than this many meters."
        ),
    )
    parser.add_argument(
        "--no-source-id-gate",
        action="store_true",
        help=(
            "Do not require matching source building IDs for candidate pairs. "
            "This is useful only for exploratory diagnostics; it can overmerge "
            "attached but distinct buildings."
        ),
    )
    parser.add_argument(
        "--max-component-polygons",
        type=int,
        default=25,
        help="Flag larger merged components for manual review. Default: 25.",
    )
    parser.add_argument(
        "--review-sample-size",
        type=int,
        default=100,
        help="Number of merged components to write for visual review. Default: 100.",
    )
    return parser.parse_args()


def require_file(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")


def geometry_is_polygonal(geometry: BaseGeometry | None) -> bool:
    if geometry is None or geometry.is_empty:
        return False
    return geometry.geom_type in {"Polygon", "MultiPolygon"}


def safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    return number


def official_height_field(city_slug: str, columns: Iterable[str]) -> str | None:
    candidates_by_city = {
        "new_york_city": ["HEIGHT_ROO", "height_roo", "official_height_m"],
        "los_angeles": ["HEIGHT", "height", "official_height_m"],
    }
    for candidate in candidates_by_city.get(city_slug, ["official_height_m"]):
        if candidate in columns:
            return candidate
    return None


def official_ground_field(city_slug: str, columns: Iterable[str]) -> str | None:
    candidates_by_city = {
        "new_york_city": ["GROUND_ELE", "ground_ele", "official_ground_m"],
        "los_angeles": ["ELEV", "elev", "official_ground_m"],
    }
    for candidate in candidates_by_city.get(city_slug, ["official_ground_m"]):
        if candidate in columns:
            return candidate
    return None


def source_building_id_field(city_slug: str, columns: Iterable[str]) -> str | None:
    candidates_by_city = {
        "new_york_city": ["BIN", "bin"],
        "los_angeles": ["BLD_ID", "NEW_BLD_ID", "OLD_BLD_ID"],
    }
    for candidate in candidates_by_city.get(city_slug, []):
        if candidate in columns:
            return candidate
    return None


def normalized_source_id(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null"}:
        return ""
    return text


def source_id_gate_passes(
    row_left: Any,
    row_right: Any,
    source_id_field: str | None,
    require_source_id_match: bool,
) -> bool:
    if not require_source_id_match or source_id_field is None:
        return True
    left_id = normalized_source_id(row_left.get(source_id_field))
    right_id = normalized_source_id(row_right.get(source_id_field))
    if not left_id or not right_id:
        return False
    return left_id == right_id


def official_height_to_meters(value: Any, field_name: str | None) -> float | None:
    number = safe_float(value)
    if number is None:
        return None
    if field_name in {"HEIGHT_ROO", "height_roo", "HEIGHT", "height"}:
        return number * 0.3048
    return number


def official_ground_to_meters(value: Any, field_name: str | None) -> float | None:
    number = safe_float(value)
    if number is None:
        return None
    if field_name in {"GROUND_ELE", "ground_ele", "ELEV", "elev"}:
        return number * 0.3048
    return number


def numeric_summary(values: pd.Series, weights: pd.Series) -> dict[str, float | int]:
    valid = values.notna()
    valid_values = values[valid].astype(float)
    valid_weights = weights[valid].astype(float)
    if valid_values.empty:
        return {
            "n": 0,
            "mean_m": math.nan,
            "median_m": math.nan,
            "min_m": math.nan,
            "max_m": math.nan,
            "area_weighted_m": math.nan,
        }
    positive_weight_sum = float(valid_weights[valid_weights > 0].sum())
    if positive_weight_sum > 0:
        area_weighted = float((valid_values * valid_weights).sum() / valid_weights.sum())
    else:
        area_weighted = math.nan
    return {
        "n": int(valid_values.size),
        "mean_m": float(valid_values.mean()),
        "median_m": float(valid_values.median()),
        "min_m": float(valid_values.min()),
        "max_m": float(valid_values.max()),
        "area_weighted_m": area_weighted,
    }


def load_city_footprints(input_dir: Path, city_slug: str) -> gpd.GeoDataFrame:
    input_path = input_dir / city_slug / f"{city_slug}{INPUT_SUFFIX}"
    require_file(input_path, f"{city_slug} clipped footprint GeoPackage")
    footprints = gpd.read_file(input_path)
    if footprints.empty:
        raise ValueError(f"No footprints found in {input_path}")
    if footprints.crs is None:
        raise ValueError(f"Footprints have no CRS: {input_path}")
    if "building_footprint_id" not in footprints.columns:
        raise ValueError(f"Missing building_footprint_id column in {input_path}")

    footprints = footprints.copy()
    footprints["geometry"] = footprints.geometry.map(make_valid)
    footprints = footprints[footprints.geometry.map(geometry_is_polygonal)].copy()
    if footprints.empty:
        raise ValueError(f"No valid polygonal footprints remain in {input_path}")
    return footprints


def project_footprints(
    footprints: gpd.GeoDataFrame,
    local_crs: int | None,
) -> gpd.GeoDataFrame:
    if local_crs is not None:
        return footprints.to_crs(epsg=local_crs)
    estimated = footprints.estimate_utm_crs()
    if estimated is None:
        raise ValueError("Could not estimate local UTM CRS; pass --local-crs.")
    return footprints.to_crs(estimated)


def height_gate_passes(
    row_left: Any,
    row_right: Any,
    height_field: str | None,
    tolerance_m: float | None,
) -> bool:
    if tolerance_m is None or height_field is None:
        return True
    left_height = official_height_to_meters(row_left.get(height_field), height_field)
    right_height = official_height_to_meters(row_right.get(height_field), height_field)
    if left_height is None or right_height is None:
        return True
    return abs(left_height - right_height) <= tolerance_m


def shared_boundary_length(geometry_left: BaseGeometry, geometry_right: BaseGeometry) -> float:
    boundary_intersection = geometry_left.boundary.intersection(geometry_right.boundary)
    return float(boundary_intersection.length)


def should_merge_pair(
    row_left: Any,
    row_right: Any,
    min_shared_boundary_m: float,
    min_shared_boundary_ratio: float,
    gap_tolerance_m: float,
    min_gap_near_boundary_m: float,
    height_field: str | None,
    height_tolerance_m: float | None,
    source_id_field: str | None,
    require_source_id_match: bool,
) -> tuple[bool, str, float, float]:
    geometry_left = row_left.geometry
    geometry_right = row_right.geometry
    if not source_id_gate_passes(
        row_left,
        row_right,
        source_id_field,
        require_source_id_match,
    ):
        return False, "source_id_mismatch", 0.0, 0.0
    if not height_gate_passes(row_left, row_right, height_field, height_tolerance_m):
        return False, "height_mismatch", 0.0, 0.0

    shared_m = shared_boundary_length(geometry_left, geometry_right)
    smaller_perimeter = min(float(geometry_left.length), float(geometry_right.length))
    shared_ratio = shared_m / smaller_perimeter if smaller_perimeter > 0 else 0.0
    if shared_m >= min_shared_boundary_m and shared_ratio >= min_shared_boundary_ratio:
        return True, "shared_boundary", shared_m, shared_ratio

    if gap_tolerance_m <= 0:
        return False, "insufficient_shared_boundary", shared_m, shared_ratio

    distance_m = float(geometry_left.distance(geometry_right))
    if distance_m > gap_tolerance_m:
        return False, "gap_too_large", shared_m, shared_ratio

    buffered_overlap = geometry_left.buffer(gap_tolerance_m).boundary.intersection(
        geometry_right.buffer(gap_tolerance_m).boundary
    )
    near_boundary_m = float(buffered_overlap.length)
    if near_boundary_m >= min_gap_near_boundary_m:
        return True, "tiny_gap", near_boundary_m, 0.0
    return False, "insufficient_gap_boundary", near_boundary_m, 0.0


def component_id(city_slug: str, position: int) -> str:
    return f"{city_slug}_merged_{position:08d}"


def build_merge_components(
    projected: gpd.GeoDataFrame,
    city_slug: str,
    args: argparse.Namespace,
) -> tuple[gpd.GeoDataFrame, list[dict[str, Any]]]:
    height_field = official_height_field(city_slug, projected.columns)
    source_id_field = source_building_id_field(city_slug, projected.columns)
    require_source_id_match = not args.no_source_id_gate
    disjoint = DisjointSet.create(len(projected))
    spatial_index = projected.sindex
    edge_rows: list[dict[str, Any]] = []
    evaluated_pairs: set[tuple[int, int]] = set()

    for left_position, left_row in enumerate(projected.itertuples()):
        candidate_positions = spatial_index.query(left_row.geometry, predicate="intersects")
        if args.gap_tolerance_m > 0:
            gap_candidates = spatial_index.query(
                left_row.geometry.buffer(args.gap_tolerance_m),
                predicate="intersects",
            )
            candidate_positions = list(set(candidate_positions).union(set(gap_candidates)))

        for right_position in candidate_positions:
            right_position = int(right_position)
            if right_position <= left_position:
                continue
            pair_key = (left_position, right_position)
            if pair_key in evaluated_pairs:
                continue
            evaluated_pairs.add(pair_key)

            right_row = projected.iloc[right_position]
            merge, reason, shared_m, shared_ratio = should_merge_pair(
                projected.iloc[left_position],
                right_row,
                args.min_shared_boundary_m,
                args.min_shared_boundary_ratio,
                args.gap_tolerance_m,
                args.min_gap_near_boundary_m,
                height_field,
                args.same_height_tolerance_m,
                source_id_field,
                require_source_id_match,
            )
            if merge:
                disjoint.union(left_position, right_position)
                edge_rows.append(
                    {
                        "city_slug": city_slug,
                        "left_building_footprint_id": projected.iloc[left_position][
                            "building_footprint_id"
                        ],
                        "right_building_footprint_id": right_row["building_footprint_id"],
                        "merge_reason": reason,
                        "shared_or_near_boundary_m": shared_m,
                        "shared_boundary_ratio": shared_ratio,
                    }
                )

    roots = [disjoint.find(position) for position in range(len(projected))]
    root_to_component: dict[int, str] = {}
    component_labels: list[str] = []
    for root in roots:
        if root not in root_to_component:
            root_to_component[root] = component_id(city_slug, len(root_to_component) + 1)
        component_labels.append(root_to_component[root])

    projected = projected.copy()
    projected["merged_building_footprint_id"] = component_labels
    projected["merge_source_id_field"] = source_id_field or ""
    projected["merge_requires_source_id_match"] = require_source_id_match
    return projected, edge_rows


def summarize_component(
    component_label: str,
    group: gpd.GeoDataFrame,
    city_slug: str,
    max_component_polygons: int,
) -> dict[str, Any]:
    source_ids = sorted(str(value) for value in group["building_footprint_id"])
    geometry = unary_union(list(group.geometry))
    polygon_count = len(group)
    original_area_sum_m2 = float(group.geometry.area.sum())
    merged_area_m2 = float(geometry.area)
    merged_perimeter_m = float(geometry.length)
    area_ratio = merged_area_m2 / original_area_sum_m2 if original_area_sum_m2 > 0 else math.nan
    review_flags = []
    if polygon_count > max_component_polygons:
        review_flags.append("large_component")
    if not geometry.is_valid:
        review_flags.append("invalid_merged_geometry")
    if area_ratio < 0.98 or area_ratio > 1.02:
        review_flags.append("area_changed_more_than_2pct")

    height_field = official_height_field(city_slug, group.columns)
    ground_field = official_ground_field(city_slug, group.columns)
    area_weights = group.geometry.area
    if height_field:
        official_height_values_m = group[height_field].map(
            lambda value: official_height_to_meters(value, height_field)
        )
    else:
        official_height_values_m = pd.Series([math.nan] * len(group), index=group.index)
    if ground_field:
        official_ground_values_m = group[ground_field].map(
            lambda value: official_ground_to_meters(value, ground_field)
        )
    else:
        official_ground_values_m = pd.Series([math.nan] * len(group), index=group.index)

    height_summary = numeric_summary(official_height_values_m, area_weights)
    ground_summary = numeric_summary(official_ground_values_m, area_weights)

    return {
        "city_slug": city_slug,
        "merged_building_footprint_id": component_label,
        "source_polygon_count": polygon_count,
        "original_area_sum_m2": original_area_sum_m2,
        "merged_area_m2": merged_area_m2,
        "merged_perimeter_m": merged_perimeter_m,
        "area_ratio": area_ratio,
        "official_height_source": height_field or "",
        "official_height_n": height_summary["n"],
        "official_height_mean_m": height_summary["mean_m"],
        "official_height_median_m": height_summary["median_m"],
        "official_height_min_m": height_summary["min_m"],
        "official_height_max_m": height_summary["max_m"],
        "official_height_area_weighted_m": height_summary["area_weighted_m"],
        "official_ground_source": ground_field or "",
        "official_ground_n": ground_summary["n"],
        "official_ground_mean_m": ground_summary["mean_m"],
        "official_ground_median_m": ground_summary["median_m"],
        "official_ground_min_m": ground_summary["min_m"],
        "official_ground_max_m": ground_summary["max_m"],
        "official_ground_area_weighted_m": ground_summary["area_weighted_m"],
        "review_flag": ";".join(review_flags),
        "source_building_footprint_ids": "|".join(source_ids),
        "geometry": geometry,
    }


def make_outputs(
    footprints: gpd.GeoDataFrame,
    projected_components: gpd.GeoDataFrame,
    city_slug: str,
    max_component_polygons: int,
    review_sample_size: int,
) -> tuple[gpd.GeoDataFrame, pd.DataFrame, pd.DataFrame, gpd.GeoDataFrame]:
    diagnostics_rows = [
        summarize_component(component_label, group, city_slug, max_component_polygons)
        for component_label, group in projected_components.groupby("merged_building_footprint_id")
    ]
    diagnostics_projected = gpd.GeoDataFrame(
        diagnostics_rows,
        geometry="geometry",
        crs=projected_components.crs,
    )

    merged = diagnostics_projected.drop(
        columns=["source_building_footprint_ids"],
        errors="ignore",
    ).to_crs(footprints.crs)
    merged["diagnostic_sample"] = False
    merged["merge_method"] = "conservative_shared_boundary_components"

    review_projected = diagnostics_projected[
        (diagnostics_projected["source_polygon_count"] > 1)
        | (diagnostics_projected["review_flag"].fillna("") != "")
    ].copy()
    review_projected = review_projected.sort_values(
        ["review_flag", "source_polygon_count"],
        ascending=[False, False],
    ).head(review_sample_size)
    review = review_projected.to_crs(footprints.crs)

    crosswalk_columns = [
        "city_slug",
        "building_footprint_id",
        "merged_building_footprint_id",
        "merge_source_id_field",
        "merge_requires_source_id_match",
    ]
    crosswalk = projected_components[crosswalk_columns].copy()
    crosswalk["merge_group_size"] = crosswalk.groupby("merged_building_footprint_id")[
        "building_footprint_id"
    ].transform("size")
    crosswalk["was_merged"] = crosswalk["merge_group_size"] > 1

    diagnostics = pd.DataFrame(diagnostics_rows).drop(columns=["geometry"])
    return merged, crosswalk, diagnostics, review


def write_geopackage(output: gpd.GeoDataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(".tmp.gpkg")
    if temporary_path.exists():
        temporary_path.unlink()
    output.to_file(temporary_path, layer=OUTPUT_LAYER, driver=OUTPUT_DRIVER)
    temporary_path.replace(path)


def write_csv(rows: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(".tmp.csv")
    rows.to_csv(temporary_path, index=False)
    temporary_path.replace(path)


def summary_row(
    city_slug: str,
    original_count: int,
    merged_count: int,
    crosswalk: pd.DataFrame,
    diagnostics: pd.DataFrame,
    edge_rows: list[dict[str, Any]],
    elapsed_seconds: float,
) -> dict[str, Any]:
    merged_groups = diagnostics[diagnostics["source_polygon_count"] > 1]
    return {
        "city_slug": city_slug,
        "original_polygon_count": original_count,
        "merged_polygon_count": merged_count,
        "polygon_count_change": merged_count - original_count,
        "merged_component_count": len(merged_groups),
        "source_polygons_in_merged_components": int(crosswalk["was_merged"].sum()),
        "largest_component_polygon_count": int(diagnostics["source_polygon_count"].max()),
        "merge_edge_count": len(edge_rows),
        "review_flagged_component_count": int((diagnostics["review_flag"].fillna("") != "").sum()),
        "elapsed_seconds": round(elapsed_seconds, 2),
    }


def selected_city_slugs(input_dir: Path, requested: list[str] | None) -> list[str]:
    if requested:
        return requested
    return sorted(
        path.name
        for path in input_dir.iterdir()
        if path.is_dir() and (path / f"{path.name}{INPUT_SUFFIX}").exists()
    )


def write_summary(summary_rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "city_slug",
        "original_polygon_count",
        "merged_polygon_count",
        "polygon_count_change",
        "merged_component_count",
        "source_polygons_in_merged_components",
        "largest_component_polygon_count",
        "merge_edge_count",
        "review_flagged_component_count",
        "elapsed_seconds",
    ]
    temporary_path = output_path.with_suffix(".tmp.csv")
    with temporary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    temporary_path.replace(output_path)


def process_city(city_slug: str, args: argparse.Namespace) -> dict[str, Any]:
    start = time.time()
    print(f"Processing {city_slug}", flush=True)
    footprints = load_city_footprints(args.input_dir, city_slug)
    projected = project_footprints(footprints, args.local_crs)
    projected_components, edge_rows = build_merge_components(projected, city_slug, args)
    merged, crosswalk, diagnostics, review = make_outputs(
        footprints,
        projected_components,
        city_slug,
        args.max_component_polygons,
        args.review_sample_size,
    )

    city_output_dir = args.input_dir / city_slug
    write_geopackage(merged, city_output_dir / f"{city_slug}{OUTPUT_SUFFIX}")
    write_geopackage(
        review,
        city_output_dir / f"{city_slug}_footprint_merge_review_sample.gpkg",
    )
    write_csv(crosswalk, city_output_dir / f"{city_slug}_footprint_merge_crosswalk.csv")
    write_csv(diagnostics, city_output_dir / f"{city_slug}_footprint_merge_diagnostics.csv")

    row = summary_row(
        city_slug,
        len(footprints),
        len(merged),
        crosswalk,
        diagnostics,
        edge_rows,
        time.time() - start,
    )
    print(
        f"  original={row['original_polygon_count']} merged={row['merged_polygon_count']} "
        f"components={row['merged_component_count']} review_flags={row['review_flagged_component_count']}",
        flush=True,
    )
    return row


def main() -> None:
    args = parse_args()
    city_slugs = selected_city_slugs(args.input_dir, args.cities)
    if not city_slugs:
        raise SystemExit("No city footprint inputs found.")

    summary_rows = []
    for city_slug in city_slugs:
        summary_rows.append(process_city(city_slug, args))

    summary_path = args.input_dir / "building_footprints_merge_summary.csv"
    write_summary(summary_rows, summary_path)
    print(f"Wrote merge summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
