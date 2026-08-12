"""
Create maps and summary tables for the global PlanetScope scene selection.

Environment: data_source/source/planet_imagery/venv_planet_imagery

This is a local analysis script. It reads previously generated CSV/GeoJSON
files and never authenticates with Planet, calls an API, orders imagery, or
downloads imagery.

Default inputs:
    data_source/data/city_aois/generated/
        wup2018_cities_over_300k_2018.csv
    data_source/data/city_aois/generated/
        wup2018_city_buffers_5km_by_city/<city_slug>_5km.geojson
    data_source/data/planet_imagery/generated/global_scene_selection/
        selected_global_planet_city_scenes.csv

Default outputs:
    data_source/data/planet_imagery/generated/global_scene_selection_analysis/
        global_aoi_and_scene_centroids.png
        country_city_scene_summary.csv
        selected_scene_numeric_metadata_summary.csv
        selected_scene_categorical_metadata_counts.csv
        selected_scene_acquisition_year_summary.csv
        detailed_city_maps/<city_slug>_selected_scene_map.png
        logs/analyze_planet_global_scene_selection_<UTC timestamp>.log
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import traceback
from typing import Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
VENV_DIR = SCRIPT_DIR / "venv_planet_imagery"
VENV_PYTHON = VENV_DIR / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
VENV_MARKER = "PLANET_IMAGERY_VENV_ACTIVE"

DEFAULT_DETAIL_CITIES = ["aba_21974", "tokyo_21671", "buenos_aires_20058"]
EXPECTED_SELECTED_SCENES = 9

NUMERIC_METADATA_COLUMNS = [
    "population_people",
    "cloud_cover",
    "clear_percent",
    "visible_percent",
    "cloud_percent",
    "shadow_percent",
    "snow_ice_percent",
    "heavy_haze_percent",
    "light_haze_percent",
    "sun_azimuth",
    "sun_elevation",
    "satellite_azimuth",
    "view_angle",
    "gsd",
    "pixel_resolution",
    "aoi_coverage_percent",
    "scene_centroid_offset_km",
    "scene_centroid_bearing_degrees",
    "selection_sun_diversity_gain_degrees",
    "selection_view_angle",
]

CATEGORICAL_METADATA_COLUMNS = [
    "quality_category",
    "instrument",
    "provider",
    "item_type",
    "selection_hemisphere",
    "selection_local_season",
    "selection_cardinal_direction",
    "selection_filter_tier_name",
    "selection_filter_relaxed",
    "selection_year_repeated",
    "selection_direction_target_met",
    "selection_season_target_met",
    "selected_asset_type",
    "asset_is_8band",
    "asset_fallback_used",
    "asset_check_status",
]

REQUIRED_INVENTORY_COLUMNS = {
    "city_slug", "city_name", "country", "latitude", "longitude",
}

REQUIRED_SCENE_COLUMNS = {
    "city_slug", "city_name", "country", "scene_id", "acquired",
    "selection_rank", "aoi_centroid_longitude", "aoi_centroid_latitude",
    "scene_centroid_longitude", "scene_centroid_latitude",
    "scene_geometry_geojson", "strip_id",
}


def relaunch_inside_venv() -> None:
    """Relaunch inside the task environment before importing dependencies."""
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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from shapely.geometry import shape


def parse_args() -> argparse.Namespace:
    """Define portable project-relative inputs and reproducible outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inventory",
        type=Path,
        default=PROJECT_ROOT
        / "data_source/data/city_aois/generated/wup2018_cities_over_300k_2018.csv",
    )
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
        / "data_source/data/planet_imagery/generated/global_scene_selection_analysis",
    )
    parser.add_argument(
        "--detail-city",
        action="append",
        dest="detail_cities",
        help=(
            "City slug for a detailed map. Repeat exactly three times. "
            "Defaults to Aba, Tokyo, and Buenos Aires."
        ),
    )
    parser.add_argument("--dpi", type=int, default=300, help="PNG output resolution.")
    return parser.parse_args()


def resolve_project_path(path: Path, *, output: bool = False) -> Path:
    """Resolve paths from the repository and reject paths outside it."""
    resolved = path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()
    if not resolved.is_relative_to(PROJECT_ROOT):
        role = "Output" if output else "Input"
        raise ValueError(f"{role} path is outside the project repository: {resolved}")
    return resolved


def require_file(path: Path, label: str) -> None:
    """Fail before analysis if a required input is missing or empty."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"Empty {label}: {path}")


def validate_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    """Fail loudly if an upstream schema no longer contains required fields."""
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {', '.join(missing)}")


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    """Write a CSV atomically so interruption cannot leave a false success."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, quoting=csv.QUOTE_MINIMAL)
    temporary.replace(path)


def atomic_save_figure(figure: plt.Figure, path: Path, dpi: int) -> None:
    """Save a complete PNG first, then atomically publish it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    figure.savefig(temporary, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    temporary.replace(path)


def load_inputs(inventory_path: Path, scenes_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read, validate, and type the source tables without modifying them."""
    inventory = pd.read_csv(inventory_path, dtype={"city_slug": str, "wup_urbancode": str})
    scenes = pd.read_csv(scenes_path, dtype={"city_slug": str, "strip_id": str})
    validate_columns(inventory, REQUIRED_INVENTORY_COLUMNS, "City inventory")
    validate_columns(scenes, REQUIRED_SCENE_COLUMNS, "Selected-scenes table")

    if inventory.empty or inventory["city_slug"].duplicated().any():
        raise ValueError("City inventory is empty or has duplicate city_slug values")
    if scenes.empty:
        raise ValueError("Selected-scenes table contains no rows")
    if scenes[["city_slug", "scene_id"]].duplicated().any():
        raise ValueError("Selected-scenes table has duplicate city_slug/scene_id rows")

    for column in ["latitude", "longitude"]:
        inventory[column] = pd.to_numeric(inventory[column], errors="coerce")
    if inventory[["latitude", "longitude"]].isna().any().any():
        raise ValueError("City inventory contains missing or invalid centroid coordinates")

    for column in set(NUMERIC_METADATA_COLUMNS) | {
        "aoi_centroid_longitude", "aoi_centroid_latitude",
        "scene_centroid_longitude", "scene_centroid_latitude", "selection_rank",
    }:
        if column in scenes.columns:
            scenes[column] = pd.to_numeric(scenes[column], errors="coerce")

    scenes["acquired"] = pd.to_datetime(scenes["acquired"], utc=True, errors="coerce")
    if scenes["acquired"].isna().any():
        raise ValueError("Selected-scenes table contains invalid acquired timestamps")
    scenes["acquisition_year"] = scenes["acquired"].dt.year

    unknown = sorted(set(scenes["city_slug"]) - set(inventory["city_slug"]))
    if unknown:
        raise ValueError(f"Selected scenes contain city slugs absent from inventory: {unknown[:10]}")
    return inventory, scenes


def make_country_summary(inventory: pd.DataFrame, scenes: pd.DataFrame) -> pd.DataFrame:
    """Count all sampled cities, represented cities, and selected scenes by country."""
    city_counts = (
        inventory.groupby("country", dropna=False)["city_slug"]
        .nunique()
        .rename("cities_in_sample")
    )
    represented = (
        scenes.groupby("country", dropna=False)["city_slug"]
        .nunique()
        .rename("cities_with_selected_scenes")
    )
    scene_counts = scenes.groupby("country", dropna=False).size().rename("selected_scene_count")
    summary = pd.concat([city_counts, represented, scene_counts], axis=1).fillna(0).reset_index()
    count_columns = ["cities_in_sample", "cities_with_selected_scenes", "selected_scene_count"]
    summary[count_columns] = summary[count_columns].astype(int)
    summary["cities_without_selected_scenes"] = (
        summary["cities_in_sample"] - summary["cities_with_selected_scenes"]
    )
    summary["mean_selected_scenes_per_sample_city"] = (
        summary["selected_scene_count"] / summary["cities_in_sample"]
    )
    summary["share_sample_cities_with_selected_scenes"] = (
        summary["cities_with_selected_scenes"] / summary["cities_in_sample"]
    )
    return summary.sort_values(
        ["cities_in_sample", "selected_scene_count", "country"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def make_numeric_summary(scenes: pd.DataFrame) -> pd.DataFrame:
    """Produce transparent distribution statistics for numeric scene metadata."""
    records: list[dict[str, object]] = []
    for column in NUMERIC_METADATA_COLUMNS:
        if column not in scenes.columns:
            continue
        values = pd.to_numeric(scenes[column], errors="coerce")
        valid = values.dropna()
        records.append(
            {
                "variable": column,
                "total_rows": len(values),
                "non_missing_count": len(valid),
                "missing_count": int(values.isna().sum()),
                "missing_percent": float(values.isna().mean() * 100),
                "mean": valid.mean(),
                "standard_deviation": valid.std(ddof=1),
                "minimum": valid.min(),
                "p25": valid.quantile(0.25),
                "median": valid.median(),
                "p75": valid.quantile(0.75),
                "maximum": valid.max(),
            }
        )
    return pd.DataFrame.from_records(records)


def make_categorical_summary(scenes: pd.DataFrame) -> pd.DataFrame:
    """Count important categorical metadata values, including missing values."""
    pieces: list[pd.DataFrame] = []
    total = len(scenes)
    for column in CATEGORICAL_METADATA_COLUMNS:
        if column not in scenes.columns:
            continue
        values = scenes[column].astype("string").fillna("<missing>")
        counts = values.value_counts(dropna=False).rename_axis("value").reset_index(name="count")
        counts.insert(0, "variable", column)
        counts["percent_of_selected_scenes"] = counts["count"] / total * 100
        pieces.append(counts)
    if not pieces:
        return pd.DataFrame(columns=["variable", "value", "count", "percent_of_selected_scenes"])
    return pd.concat(pieces, ignore_index=True)


def make_year_summary(scenes: pd.DataFrame) -> pd.DataFrame:
    """Summarize scene and city coverage by acquisition calendar year."""
    return (
        scenes.groupby("acquisition_year")
        .agg(selected_scene_count=("scene_id", "size"), cities_represented=("city_slug", "nunique"))
        .reset_index()
        .sort_values("acquisition_year")
    )


def plot_global_map(
    inventory: pd.DataFrame,
    scenes: pd.DataFrame,
    output_path: Path,
    dpi: int,
) -> None:
    """Overlay all AOI centroids and all selected scene centroids on one world map."""
    figure = plt.figure(figsize=(16, 9))
    axis = figure.add_subplot(1, 1, 1, projection=ccrs.Robinson())
    axis.set_global()
    axis.add_feature(cfeature.LAND.with_scale("110m"), facecolor="#F1F3F5", zorder=1)
    axis.add_feature(cfeature.OCEAN.with_scale("110m"), facecolor="#EAF3F8", zorder=0)
    axis.add_feature(cfeature.COASTLINE.with_scale("110m"), edgecolor="#69737D", linewidth=0.35, zorder=2)
    axis.add_feature(cfeature.BORDERS.with_scale("110m"), edgecolor="#A0A8B0", linewidth=0.25, zorder=2)

    # Hollow AOI circles remain readable while allowing the nearby scene
    # centroids to show through at the global scale.
    valid_scenes = scenes.dropna(subset=["scene_centroid_longitude", "scene_centroid_latitude"])
    axis.scatter(
        inventory["longitude"],
        inventory["latitude"],
        s=16,
        marker="o",
        facecolors="none",
        edgecolors="#1769AA",
        linewidths=0.65,
        alpha=0.85,
        label=f"5 km AOI centroids (n={len(inventory):,})",
        transform=ccrs.PlateCarree(),
        zorder=3,
    )
    axis.scatter(
        valid_scenes["scene_centroid_longitude"],
        valid_scenes["scene_centroid_latitude"],
        s=7,
        marker="x",
        linewidths=0.45,
        color="#D92D20",
        alpha=0.45,
        label=f"Selected scene centroids (n={len(valid_scenes):,})",
        transform=ccrs.PlateCarree(),
        zorder=4,
    )
    axis.gridlines(color="#C7D0D9", linewidth=0.4, linestyle="--", alpha=0.7)
    axis.set_title("Global WUP City AOIs and Selected PlanetScope Scene Centroids", fontsize=15, pad=14)
    axis.legend(loc="lower left", frameon=True, framealpha=0.95)
    figure.tight_layout()
    atomic_save_figure(figure, output_path, dpi)


def parse_scene_geometries(city_scenes: pd.DataFrame, city_slug: str) -> list:
    """Parse the exact Planet scene-footprint GeoJSON stored in the selection CSV."""
    geometries = []
    for row_number, value in city_scenes["scene_geometry_geojson"].items():
        try:
            geometry = shape(json.loads(value))
        except Exception as error:
            raise ValueError(
                f"Invalid scene_geometry_geojson for {city_slug} at source row {row_number}"
            ) from error
        if geometry.is_empty or not geometry.is_valid:
            raise ValueError(f"Empty or invalid scene geometry for {city_slug} at row {row_number}")
        geometries.append(geometry)
    return geometries


def load_single_aoi_geometry(path: Path, city_slug: str):
    """Read the one authoritative WGS84 AOI polygon without a GIS file driver."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        features = payload["features"]
        if len(features) != 1:
            raise ValueError(f"expected one feature, found {len(features)}")
        geometry = shape(features[0]["geometry"])
    except Exception as error:
        raise ValueError(f"Invalid 5 km AOI GeoJSON for {city_slug}: {path}") from error
    if geometry.is_empty or not geometry.is_valid:
        raise ValueError(f"Empty or invalid 5 km AOI geometry for {city_slug}: {path}")
    return geometry


def add_polygon_outline(axis: plt.Axes, geometry, **plot_options) -> None:
    """Draw Polygon or MultiPolygon boundaries with ordinary Matplotlib."""
    polygons = [geometry] if geometry.geom_type == "Polygon" else list(geometry.geoms)
    for polygon in polygons:
        x_values, y_values = polygon.exterior.xy
        axis.fill(x_values, y_values, **plot_options)


def padded_bounds(geometries: Iterable, minimum_padding_degrees: float = 0.025) -> tuple[float, float, float, float]:
    """Return combined bounds with padding so outlines do not touch plot edges."""
    bounds = [geometry.bounds for geometry in geometries]
    minimum_x = min(value[0] for value in bounds)
    minimum_y = min(value[1] for value in bounds)
    maximum_x = max(value[2] for value in bounds)
    maximum_y = max(value[3] for value in bounds)
    padding_x = max((maximum_x - minimum_x) * 0.08, minimum_padding_degrees)
    padding_y = max((maximum_y - minimum_y) * 0.08, minimum_padding_degrees)
    return (
        minimum_x - padding_x,
        minimum_y - padding_y,
        maximum_x + padding_x,
        maximum_y + padding_y,
    )


def plot_detailed_city_map(
    city_slug: str,
    inventory: pd.DataFrame,
    scenes: pd.DataFrame,
    aoi_dir: Path,
    output_path: Path,
    dpi: int,
) -> None:
    """Plot one 5 km AOI together with selected centroids and scene footprints."""
    city_rows = inventory[inventory["city_slug"] == city_slug]
    if len(city_rows) != 1:
        raise ValueError(f"Expected exactly one inventory row for detail city {city_slug}")
    city_scenes = scenes[scenes["city_slug"] == city_slug].sort_values("selection_rank").copy()
    if city_scenes.empty:
        raise ValueError(f"Detail city has no selected scenes: {city_slug}")

    aoi_path = aoi_dir / f"{city_slug}_5km.geojson"
    require_file(aoi_path, f"5 km AOI for {city_slug}")
    aoi_geometry = load_single_aoi_geometry(aoi_path, city_slug)
    scene_geometries = parse_scene_geometries(city_scenes, city_slug)

    strip_values = sorted(city_scenes["strip_id"].fillna("<missing>").astype(str).unique())
    color_map = plt.get_cmap("tab10")
    strip_colors = {strip: color_map(index % 10) for index, strip in enumerate(strip_values)}

    figure, axis = plt.subplots(figsize=(10, 10))
    for (_, scene), geometry in zip(city_scenes.iterrows(), scene_geometries):
        strip = str(scene["strip_id"]) if pd.notna(scene["strip_id"]) else "<missing>"
        add_polygon_outline(
            axis,
            geometry,
            facecolor=strip_colors[strip],
            edgecolor=strip_colors[strip],
            alpha=0.10,
            linewidth=1.0,
            zorder=1,
        )

    add_polygon_outline(
        axis,
        aoi_geometry,
        facecolor="#F4C95D",
        edgecolor="#111827",
        alpha=0.28,
        linewidth=2.2,
        zorder=3,
    )
    axis.scatter(
        city_scenes["scene_centroid_longitude"],
        city_scenes["scene_centroid_latitude"],
        s=46,
        marker="x",
        linewidths=1.4,
        color="#B42318",
        zorder=5,
    )
    city = city_rows.iloc[0]
    axis.scatter(
        [city["longitude"]],
        [city["latitude"]],
        s=95,
        marker="*",
        color="#1769AA",
        edgecolor="white",
        linewidth=0.8,
        zorder=6,
    )

    # Rank labels make the figure auditable against selection_rank in the CSV.
    for _, scene in city_scenes.iterrows():
        axis.annotate(
            str(int(scene["selection_rank"])),
            (scene["scene_centroid_longitude"], scene["scene_centroid_latitude"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
            color="#7A271A",
            zorder=7,
        )

    minimum_x, minimum_y, maximum_x, maximum_y = padded_bounds(
        [aoi_geometry, *scene_geometries]
    )
    axis.set_xlim(minimum_x, maximum_x)
    axis.set_ylim(minimum_y, maximum_y)
    mean_latitude = float(city["latitude"])
    axis.set_aspect(1 / max(math.cos(math.radians(mean_latitude)), 0.20), adjustable="box")
    axis.grid(color="#D8DEE4", linewidth=0.5, linestyle="--")
    axis.set_xlabel("Longitude")
    axis.set_ylabel("Latitude")
    axis.set_title(
        f"{city['city_name']}, {city['country']}: 5 km AOI and Selected Scenes",
        fontsize=14,
        pad=12,
    )

    legend_items = [
        Patch(facecolor="#F4C95D", edgecolor="#111827", alpha=0.45, label="5 km AOI boundary"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor="#1769AA", markeredgecolor="white", markersize=12, label="AOI centroid"),
        Line2D([0], [0], marker="x", color="#B42318", linestyle="none", markersize=7, label="Scene centroid (label = rank)"),
    ]
    legend_items.extend(
        Patch(facecolor=strip_colors[strip], edgecolor=strip_colors[strip], alpha=0.22, label=f"Scene footprint; strip {strip}")
        for strip in strip_values
    )
    axis.legend(handles=legend_items, loc="best", fontsize=8, framealpha=0.95)
    figure.tight_layout()
    atomic_save_figure(figure, output_path, dpi)


class Tee:
    """Mirror messages to the terminal and a dated run log."""

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
    """Run all requested summaries and maps, failing loudly on any partial output."""
    args = parse_args()
    inventory_path = resolve_project_path(args.inventory)
    scenes_path = resolve_project_path(args.selected_scenes)
    aoi_dir = resolve_project_path(args.aoi_dir)
    output_dir = resolve_project_path(args.output_dir, output=True)
    detail_cities = args.detail_cities or DEFAULT_DETAIL_CITIES
    if len(detail_cities) != 3 or len(set(detail_cities)) != 3:
        raise ValueError("Provide exactly three distinct --detail-city slugs")
    if args.dpi < 72:
        raise ValueError("--dpi must be at least 72")

    require_file(inventory_path, "global city inventory")
    require_file(scenes_path, "combined selected-scenes CSV")
    if not aoi_dir.is_dir():
        raise FileNotFoundError(f"Missing global AOI directory: {aoi_dir}")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    logger = Tee(output_dir / "logs" / f"analyze_planet_global_scene_selection_{timestamp}.log")
    try:
        logger.write("Planet global scene-selection analysis started")
        logger.write(f"Inventory: {inventory_path.relative_to(PROJECT_ROOT)}")
        logger.write(f"Selected scenes: {scenes_path.relative_to(PROJECT_ROOT)}")
        inventory, scenes = load_inputs(inventory_path, scenes_path)
        logger.write(
            f"Loaded {len(inventory):,} sample cities and {len(scenes):,} selected scenes "
            f"covering {scenes['city_slug'].nunique():,} cities"
        )

        country_summary = make_country_summary(inventory, scenes)
        numeric_summary = make_numeric_summary(scenes)
        categorical_summary = make_categorical_summary(scenes)
        year_summary = make_year_summary(scenes)
        outputs = {
            "country summary": output_dir / "country_city_scene_summary.csv",
            "numeric metadata summary": output_dir / "selected_scene_numeric_metadata_summary.csv",
            "categorical metadata counts": output_dir / "selected_scene_categorical_metadata_counts.csv",
            "acquisition-year summary": output_dir / "selected_scene_acquisition_year_summary.csv",
        }
        for frame, (label, path) in zip(
            [country_summary, numeric_summary, categorical_summary, year_summary], outputs.items()
        ):
            atomic_write_csv(frame, path)
            logger.write(f"Wrote {label}: {path.relative_to(PROJECT_ROOT)} ({len(frame):,} rows)")

        global_map = output_dir / "global_aoi_and_scene_centroids.png"
        plot_global_map(inventory, scenes, global_map, args.dpi)
        logger.write(f"Wrote global map: {global_map.relative_to(PROJECT_ROOT)}")

        for city_slug in detail_cities:
            city_map = output_dir / "detailed_city_maps" / f"{city_slug}_selected_scene_map.png"
            plot_detailed_city_map(city_slug, inventory, scenes, aoi_dir, city_map, args.dpi)
            logger.write(f"Wrote detailed map: {city_map.relative_to(PROJECT_ROOT)}")

        cities_without_scenes = len(inventory) - scenes["city_slug"].nunique()
        cities_below_nine = int(
            (scenes.groupby("city_slug").size().reindex(inventory["city_slug"], fill_value=0) < EXPECTED_SELECTED_SCENES).sum()
        )
        logger.write(f"Cities without selected scenes: {cities_without_scenes:,}")
        logger.write(f"Cities with fewer than nine selected scenes: {cities_below_nine:,}")
        logger.write("SUCCESS: all requested tables and maps were written")
        return 0
    except Exception:
        logger.write("FAILED: analysis did not complete")
        logger.write(traceback.format_exc())
        return 1
    finally:
        logger.close()


if __name__ == "__main__":
    raise SystemExit(main())
