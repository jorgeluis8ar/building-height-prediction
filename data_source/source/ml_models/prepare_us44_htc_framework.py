#!/usr/bin/env python3
"""Build resumable Windows HTC datasets for 44 processed US LiDAR cities."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import pickle
import random
import shutil
import sys
import traceback
from typing import Any, Iterable, Optional, Union

import numpy as np
import pandas as pd

try:
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.features import shapes
    from rasterio.vrt import WarpedVRT
    from rasterio.windows import Window, bounds as window_bounds
except ModuleNotFoundError:
    rasterio = None
    Resampling = shapes = WarpedVRT = Window = window_bounds = None

try:
    import torch
except ModuleNotFoundError:
    torch = None


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_NDSM_MANIFEST = Path("data_source/data/height_labels/generated/us_training_planet_ndsm/us_lidar_to_planet_ndsm_manifest.csv")
DEFAULT_SCENE_METADATA = Path("data_source/data/planet_imagery/generated/processed_us_lidar_scene_selection/selected_processed_us_lidar_planet_scenes.csv")
DEFAULT_SCENE_ROOT = Path("data_source/data/planet_imagery/source/processed_us_lidar")
DEFAULT_OUTPUT_ROOT = Path("data_source/data/ml_models/generated/htc_dc_net")
STAGING_NAME = "us_44_city_staging_v1"
DATASETS = {4: "us_44_rgbnir_4scene_v1", 8: "us_44_rgbnir_8scene_v1"}
STAGES = ("inventory", "split", "canonical-chips", "align-scenes", "build-datasets", "statistics", "validate")
SPLITS = ("train", "val", "test")
CHIP_SIZE = 256
RESOLUTION_M = 3.0
CHANNELS = ("red", "green", "blue", "nir")
UDM_NAMES = ("clear", "snow", "shadow", "light_haze", "heavy_haze", "cloud", "confidence", "unusable")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=(*STAGES, "all"), default="all")
    parser.add_argument("--ndsm-manifest", type=Path, default=DEFAULT_NDSM_MANIFEST)
    parser.add_argument("--scene-metadata", type=Path, default=DEFAULT_SCENE_METADATA)
    parser.add_argument("--scene-root", type=Path, default=DEFAULT_SCENE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--expected-cities", type=int, default=44)
    parser.add_argument("--expected-scenes-per-city", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261005)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--qa-chips-per-city", type=int, default=3)
    return parser.parse_args()


def project_path(path: Path) -> Path:
    resolved = path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()
    try:
        resolved.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise ValueError(f"Path must remain inside repository: {resolved}") from exc
    return resolved


def rel(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT).as_posix()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(str(temporary), str(path))


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(str(temporary), str(path))


def atomic_pickle(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(value, stream)
    os.replace(str(temporary), str(path))


def atomic_torch(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    os.replace(str(temporary), str(path))


def logger_for(root: Path, stage: str) -> tuple[logging.Logger, Path]:
    log_dir = root / STAGING_NAME / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = log_dir / f"{stage}_{stamp}.log"
    logger = logging.getLogger("us44_htc")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)sZ %(levelname)s %(message)s", "%Y-%m-%dT%H:%M:%S")
    for handler in (logging.FileHandler(path, encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger, path


def read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Required CSV is missing: {path}")
    frame = pd.read_csv(path)
    if frame.empty:
        raise RuntimeError(f"Required CSV is empty: {path}")
    return frame


def column(frame: pd.DataFrame, candidates: Iterable[str], label: str) -> str:
    for name in candidates:
        if name in frame.columns:
            return name
    raise RuntimeError(f"Missing {label}; expected one of {list(candidates)}")


def find_one(root: Path, patterns: Iterable[str], label: str) -> Path:
    matches: set[Path] = set()
    for pattern in patterns:
        matches.update(path.resolve() for path in root.glob(pattern) if path.is_file())
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {label} under {root}; found {len(matches)}: {sorted(matches)[:5]}")
    return next(iter(matches))


def choose_delivered_raster(candidates: Iterable[Path], label: str, root: Path) -> Path:
    """Choose one delivered raster, preferring the AOI-clipped product."""
    matches = sorted({path.resolve() for path in candidates if path.is_file()})
    clipped = [path for path in matches if "clip" in path.name.lower()]
    usable = clipped if len(clipped) == 1 else matches
    if len(usable) != 1:
        raise RuntimeError(
            f"Expected one {label} under {root}; found {len(matches)}: {matches[:5]}"
        )
    return usable[0]


def locate_scene(root: Path, city: str, scene: str) -> tuple[Path, Path]:
    city_root = root / city
    # Planet delivery filenames vary by sensor generation and processing
    # bundle. Match by scene ID and product identity instead of assuming one
    # exact 4-band or 8-band suffix.
    scene_tifs = list(city_root.glob(f"**/{scene}*.tif"))
    sr = choose_delivered_raster(
        (
            path for path in scene_tifs
            if "analyticms" in path.name.lower()
            and "_sr" in path.name.lower()
            and "udm" not in path.name.lower()
        ),
        f"SR raster for {city}/{scene}",
        city_root,
    )
    udm = choose_delivered_raster(
        (path for path in scene_tifs if "udm2" in path.name.lower()),
        f"UDM2 raster for {city}/{scene}",
        city_root,
    )
    return sr, udm


def inspect_raster(path: Path, bands: Optional[int] = None) -> None:
    with rasterio.open(path) as src:
        if bands is not None and src.count != bands:
            raise RuntimeError(f"{path}: expected {bands} bands, found {src.count}")
        if src.crs is None or src.width <= 0 or src.height <= 0:
            raise RuntimeError(f"{path}: missing CRS or empty dimensions")
        src.read(1, window=Window(0, 0, min(16, src.width), min(16, src.height)))


def stage_inventory(args: argparse.Namespace, root: Path, log: logging.Logger) -> None:
    ndsm = read_csv(project_path(args.ndsm_manifest))
    scenes = read_csv(project_path(args.scene_metadata))
    ndsm_city = column(ndsm, ("city_slug", "city"), "nDSM city")
    ndsm_file = column(ndsm, ("output_ndsm_path", "ndsm_path", "output_raster"), "nDSM path")
    scene_city = column(scenes, ("city_slug", "city"), "scene city")
    scene_id = column(scenes, ("scene_id", "id"), "scene ID")
    ndsm[ndsm_city] = ndsm[ndsm_city].astype(str)
    scenes[scene_city] = scenes[scene_city].astype(str)
    scenes[scene_id] = scenes[scene_id].astype(str)
    if ndsm[ndsm_city].duplicated().any():
        duplicate = ndsm.loc[ndsm[ndsm_city].duplicated(), ndsm_city].iloc[0]
        raise RuntimeError(f"Duplicate nDSM manifest city: {duplicate}")
    scene_root = project_path(args.scene_root)
    records: list[dict[str, Any]] = []
    normalized: list[dict[str, Any]] = []
    # The selected-scene manifest defines this framework's 44-city universe.
    # The nDSM manifest can legitimately contain additional processed cities.
    selected_cities = set(scenes[scene_city])
    ndsm_only = sorted(set(ndsm[ndsm_city]) - selected_cities)
    atomic_csv(
        pd.DataFrame({"city_slug": ndsm_only, "reason": "not_in_selected_44_city_scene_manifest"}),
        root / STAGING_NAME / "ignored_ndsm_only_cities.csv",
    )
    for city in sorted(selected_cities):
        errors: list[str] = []
        ndsm_rows = ndsm[ndsm[ndsm_city] == city]
        ndsm_path: Optional[Path] = None
        if len(ndsm_rows) != 1:
            errors.append(f"expected one nDSM row; found {len(ndsm_rows)}")
        else:
            ndsm_path = project_path(Path(str(ndsm_rows.iloc[0][ndsm_file])))
            try:
                inspect_raster(ndsm_path, 3)
            except Exception as exc:
                errors.append(f"nDSM: {exc}")
        city_scenes = scenes[scenes[scene_city] == city].copy()
        if len(city_scenes) != args.expected_scenes_per_city:
            errors.append(f"expected {args.expected_scenes_per_city} selected scenes; found {len(city_scenes)}")
        if city_scenes[scene_id].duplicated().any():
            errors.append("duplicate scene IDs")
        downloaded = 0
        for _, row in city_scenes.iterrows():
            sid = str(row[scene_id])
            try:
                sr, udm_path = locate_scene(scene_root, city, sid)
                inspect_raster(sr)
                inspect_raster(udm_path, 8)
                output = row.to_dict()
                output.update(city_slug=city, scene_id=sid, sr_path=rel(sr), udm2_path=rel(udm_path))
                normalized.append(output)
                downloaded += 1
            except Exception as exc:
                errors.append(f"{sid}: {exc}")
        records.append({"city_slug": city, "ndsm_path": rel(ndsm_path) if ndsm_path and ndsm_path.is_file() else "", "selected_scene_count": len(city_scenes), "downloaded_scene_count": downloaded, "inventory_complete": not errors, "error": "; ".join(errors)})
    inventory = pd.DataFrame(records)
    staging = root / STAGING_NAME
    atomic_csv(inventory, staging / "city_input_inventory.csv")
    atomic_csv(inventory[~inventory["inventory_complete"]], staging / "missing_inputs.csv")
    atomic_csv(pd.DataFrame(normalized), staging / "scene_metadata.csv")
    complete = int(inventory["inventory_complete"].sum())
    log.info("Inventory: %d complete of %d discovered cities", complete, len(inventory))
    if complete != args.expected_cities or len(inventory) != args.expected_cities:
        raise RuntimeError(f"Expected exactly {args.expected_cities} complete cities; found {complete}. See missing_inputs.csv")


def stage_split(args: argparse.Namespace, root: Path, log: logging.Logger) -> None:
    staging = root / STAGING_NAME
    inventory = read_csv(staging / "city_input_inventory.csv")
    complete = inventory["inventory_complete"].astype(str).str.lower().eq("true")
    cities = sorted(inventory.loc[complete, "city_slug"].astype(str))
    if len(cities) != 44:
        raise RuntimeError(f"The fixed 19/19/6 split requires 44 cities; found {len(cities)}")
    random.Random(args.seed).shuffle(cities)
    groups = {"train": cities[:19], "val": cities[19:38], "test": cities[38:]}
    rows = []
    rank = 0
    for split in SPLITS:
        atomic_text(staging / f"{split}_cities.txt", "".join(f"{city}\n" for city in groups[split]))
        for city in groups[split]:
            rank += 1
            rows.append({"city_slug": city, "split": split, "randomized_rank": rank, "seed": args.seed})
    atomic_csv(pd.DataFrame(rows), staging / "city_split_manifest.csv")
    log.info("Created city split train=19 val=19 test=6 seed=%d", args.seed)


def profile_for(source: rasterio.DatasetReader, count: int, dtype: str, nodata: Union[float, int], transform: rasterio.Affine) -> dict[str, Any]:
    profile = source.profile.copy()
    profile.update(driver="GTiff", count=count, dtype=dtype, nodata=nodata, width=CHIP_SIZE, height=CHIP_SIZE, transform=transform, compress="DEFLATE", predictor=2, tiled=True, blockxsize=256, blockysize=256, BIGTIFF="IF_SAFER")
    return profile


def write_raster(path: Path, profile: dict[str, Any], data: np.ndarray, descriptions: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.tif")
    with rasterio.open(temporary, "w", **profile) as dst:
        dst.write(data)
        for index, name in enumerate(descriptions, 1):
            dst.set_band_description(index, name)
    os.replace(str(temporary), str(path))


def stage_canonical(args: argparse.Namespace, root: Path, log: logging.Logger) -> None:
    staging = root / STAGING_NAME
    inventory = read_csv(staging / "city_input_inventory.csv")
    assignments = read_csv(staging / "city_split_manifest.csv").set_index("city_slug")
    city_numbers = {city: index + 1 for index, city in enumerate(sorted(assignments.index))}
    combined = []
    for _, item in inventory.sort_values("city_slug").iterrows():
        city = str(item["city_slug"])
        city_root = staging / city
        manifest_path = city_root / "geographic_chips_manifest.csv"
        if args.resume and manifest_path.is_file():
            frame = read_csv(manifest_path)
            combined.append(frame)
            log.info("%s: reused %d target chips", city, len(frame))
            continue
        if city_root.exists() and args.overwrite:
            shutil.rmtree(city_root)
        for folder in ("targets/mask", "targets/ndsm", "targets/qa", "image", "qa"):
            (city_root / folder).mkdir(parents=True, exist_ok=True)
        rows = []
        ndsm_path = project_path(Path(str(item["ndsm_path"])))
        with rasterio.open(ndsm_path) as src:
            if src.count != 3 or not np.allclose(np.abs(src.res), (3, 3), atol=0.01):
                raise RuntimeError(f"{city}: nDSM must have three bands at 3 m")
            for row_off in range(0, src.height, CHIP_SIZE):
                for col_off in range(0, src.width, CHIP_SIZE):
                    window = Window(col_off, row_off, CHIP_SIZE, CHIP_SIZE)
                    data = src.read(window=window, boundless=True, fill_value=0, out_shape=(3, CHIP_SIZE, CHIP_SIZE))
                    agl = np.maximum(data[1].astype("float32"), 0)
                    lidar_qa = np.rint(data[2]).astype("uint8")
                    mask = (lidar_qa == 2).astype("uint8")
                    if mask.sum() == 0:
                        continue
                    gid = f"c{city_numbers[city]:02d}_r{row_off // CHIP_SIZE:03d}c{col_off // CHIP_SIZE:03d}"
                    transform = src.window_transform(window)
                    paths = {
                        "mask": city_root / "targets/mask" / f"{gid}_BLG.tif",
                        "agl": city_root / "targets/ndsm" / f"{gid}_AGL.tif",
                        "lqa": city_root / "targets/qa" / f"{gid}_LQA.tif",
                    }
                    write_raster(paths["mask"], profile_for(src, 1, "uint8", 0, transform), mask[None], ("building_mask",))
                    write_raster(paths["agl"], profile_for(src, 1, "float32", -9999.0, transform), agl[None], ("building_only_ndsm_m",))
                    write_raster(paths["lqa"], profile_for(src, 1, "uint8", 0, transform), lidar_qa[None], ("lidar_qa_code",))
                    heights = agl[mask == 1]
                    hist = np.histogram(heights, bins=[0, 10, 20, 30, 40, 50, np.inf])[0] / heights.size
                    building_count = sum(
                        1 for _, value in shapes(mask, mask=mask.astype(bool), transform=transform)
                        if int(value) == 1
                    )
                    left, bottom, right, top = window_bounds(window, src.transform)
                    rows.append({
                        "geographic_chip_id": gid, "city_slug": city, "split": assignments.loc[city, "split"],
                        "row_off": row_off, "col_off": col_off, "crs": src.crs.to_string(),
                        "transform": ",".join(str(value) for value in transform[:6]), "left": left, "bottom": bottom, "right": right, "top": top,
                        "building_pixels": int(mask.sum()), "building_count": building_count,
                        "valid_lidar_fraction": float((lidar_qa > 0).mean()),
                        "positive_target_fraction": float((heights > 0).mean()), "height_median_m": float(np.median(heights)),
                        "height_p90_m": float(np.percentile(heights, 90)), "height_max_m": float(np.max(heights)),
                        "share_0_10m": hist[0], "share_10_20m": hist[1], "share_20_30m": hist[2],
                        "share_30_40m": hist[3], "share_40_50m": hist[4], "share_50plus_m": hist[5],
                        "mask_path": rel(paths["mask"]), "agl_path": rel(paths["agl"]), "lidar_qa_path": rel(paths["lqa"]),
                    })
        frame = pd.DataFrame(rows)
        if frame.empty:
            raise RuntimeError(f"{city}: no target chips contained building pixels")
        atomic_csv(frame, manifest_path)
        combined.append(frame)
        log.info("%s: wrote %d target chips", city, len(frame))
    atomic_csv(pd.concat(combined, ignore_index=True), staging / "geographic_chips_manifest.csv")


def band_indexes(src: rasterio.DatasetReader) -> tuple[int, int, int, int]:
    descriptions = [(value or "").lower().replace(" ", "_") for value in src.descriptions]
    found = []
    aliases = {"red": ("red",), "green": ("green", "green_ii"), "blue": ("blue",), "nir": ("nir", "near_infrared", "near-infrared")}
    for name in CHANNELS:
        match = next((index + 1 for index, value in enumerate(descriptions) if value in aliases[name]), None)
        if match is None:
            found = []
            break
        found.append(match)
    if found:
        return tuple(found)  # type: ignore[return-value]
    if src.count == 8:
        return 6, 4, 2, 8
    if src.count == 4:
        return 3, 2, 1, 4
    raise RuntimeError(f"Cannot infer RGB+NIR bands from {src.count} bands: {descriptions}")


def read_padded_window(
    source: rasterio.DatasetReader,
    indexes: Iterable[int],
    row_off: int,
    col_off: int,
    dtype: str,
) -> np.ndarray:
    """Read an exact grid window and zero-pad a partial city-edge chip."""
    band_list = list(indexes)
    output = np.zeros((len(band_list), CHIP_SIZE, CHIP_SIZE), dtype=dtype)
    height = min(CHIP_SIZE, source.height - row_off)
    width = min(CHIP_SIZE, source.width - col_off)
    if height <= 0 or width <= 0:
        return output
    values = source.read(band_list, window=Window(col_off, row_off, width, height))
    output[:, :height, :width] = values.astype(dtype, copy=False)
    return output


def stage_align(args: argparse.Namespace, root: Path, log: logging.Logger) -> None:
    staging = root / STAGING_NAME
    metadata = read_csv(staging / "scene_metadata.csv")
    chips = read_csv(staging / "geographic_chips_manifest.csv")
    inventory = read_csv(staging / "city_input_inventory.csv").set_index("city_slug")
    observations, excluded = [], []
    rank_column = "selection_rank" if "selection_rank" in metadata else "scene_id"
    for city, city_chips in chips.groupby("city_slug", sort=True):
        city_root = staging / str(city)
        city_rows = []
        city_scenes = metadata[metadata["city_slug"] == city].sort_values(rank_column)
        for scene_number, (_, scene) in enumerate(city_scenes.iterrows(), 1):
            sr_path = project_path(Path(str(scene["sr_path"])))
            udm_path = project_path(Path(str(scene["udm2_path"])))
            ndsm_path = project_path(Path(str(inventory.loc[city, "ndsm_path"])))
            with rasterio.open(sr_path) as sr, rasterio.open(udm_path) as udm, rasterio.open(ndsm_path) as ndsm:
                indexes = band_indexes(sr)
                vrt_grid = {"crs": ndsm.crs, "transform": ndsm.transform, "width": ndsm.width, "height": ndsm.height}
                with WarpedVRT(sr, **vrt_grid, resampling=Resampling.bilinear, nodata=0) as sr_vrt, WarpedVRT(udm, **vrt_grid, resampling=Resampling.nearest, nodata=0) as udm_vrt:
                    for _, chip in city_chips.iterrows():
                        gid = str(chip["geographic_chip_id"])
                        obs = f"{gid}_s{scene_number}"
                        image_path = city_root / "image" / f"{obs}_IMG.tif"
                        qa_path = city_root / "qa" / f"{obs}_QA.tif"
                        coverage = None
                        if not (args.resume and image_path.is_file() and qa_path.is_file()):
                            row_off, col_off = int(chip["row_off"]), int(chip["col_off"])
                            image = read_padded_window(sr_vrt, indexes, row_off, col_off, "float32")
                            udm_data = read_padded_window(udm_vrt, range(1, 9), row_off, col_off, "uint8")
                            with rasterio.open(project_path(Path(str(chip["lidar_qa_path"])))) as lqa_src:
                                lqa = lqa_src.read(1)
                                template = lqa_src
                                spectral = np.all(np.isfinite(image) & (image > 0), axis=0)
                                usable = spectral & (udm_data[7] == 0)
                                building = lqa == 2
                                coverage = float(usable[building].mean())
                                if coverage >= 0.99:
                                    write_raster(image_path, profile_for(template, 4, "float32", 0, template.transform), image, CHANNELS)
                                    qa = np.concatenate((lqa[None].astype("uint8"), udm_data))
                                    write_raster(qa_path, profile_for(template, 9, "uint8", 0, template.transform), qa, ("lidar_qa_code", *UDM_NAMES))
                        if coverage is not None and coverage < 0.99:
                            excluded.append({"observation_id": obs, "city_slug": city, "scene_id": scene["scene_id"], "geographic_chip_id": gid, "reason": "spectral_building_coverage_below_0.99", "coverage": coverage})
                            continue
                        row = chip.to_dict()
                        row.update(scene.to_dict())
                        row.update(observation_id=obs, scene_number=scene_number, image_path=rel(image_path), qa_path=rel(qa_path), source_band_indexes=",".join(map(str, indexes)), spectral_building_coverage=coverage if coverage is not None else "validated_previous_run")
                        city_rows.append(row)
        city_frame = pd.DataFrame(city_rows)
        if city_frame.empty:
            raise RuntimeError(f"{city}: all scene observations were excluded")
        atomic_csv(city_frame, city_root / "chips_manifest.csv")
        observations.append(city_frame)
        log.info("%s: retained %d scene observations", city, len(city_frame))
    atomic_csv(pd.concat(observations, ignore_index=True), staging / "chips_manifest.csv")
    atomic_csv(pd.DataFrame(excluded, columns=("observation_id", "city_slug", "scene_id", "geographic_chip_id", "reason", "coverage")), staging / "excluded_observations.csv")


def season(row: pd.Series) -> str:
    value = str(row.get("selection_local_season", row.get("season", ""))).lower()
    return "summer" if "summer" in value else "winter" if "winter" in value else "other"


def direction(row: pd.Series) -> str:
    value = str(row.get("selection_cardinal_direction", row.get("scene_centroid_direction", ""))).upper()
    return "N" if value in ("N", "NE", "NW") else "S" if value in ("S", "SE", "SW") else value


def scene_score(row: pd.Series) -> tuple[Any, ...]:
    def number(name: str, default: float) -> float:
        try:
            return float(row.get(name, default))
        except (TypeError, ValueError):
            return default
    return number("selection_filter_tier", 99), -number("aoi_coverage_percent", 0), number("cloud_cover", 1), -number("clear_percent", 0), str(row["scene_id"])


def select_four(frame: pd.DataFrame) -> tuple[set[str], list[dict[str, Any]]]:
    scenes = frame.drop_duplicates("scene_id").copy()
    chosen: list[pd.Series] = []
    audit = []
    for wanted_season, wanted_direction in (("summer", "N"), ("summer", "S"), ("winter", "N"), ("winter", "S")):
        available = scenes[~scenes["scene_id"].isin([row["scene_id"] for row in chosen])]
        exact = available[available.apply(lambda row: season(row) == wanted_season and direction(row) == wanted_direction, axis=1)]
        same_season = available[available.apply(lambda row: season(row) == wanted_season, axis=1)]
        same_direction = available[available.apply(lambda row: direction(row) == wanted_direction, axis=1)]
        pool, match = (exact, "exact") if not exact.empty else (same_season, "same_season") if not same_season.empty else (same_direction, "same_direction") if not same_direction.empty else (available, "best_remaining")
        if pool.empty:
            raise RuntimeError("Fewer than four unique scenes are available")
        selected = min((row for _, row in pool.iterrows()), key=scene_score)
        chosen.append(selected)
        audit.append({"city_slug": selected["city_slug"], "requested_role": f"{wanted_season}_{wanted_direction.lower()}", "scene_id": selected["scene_id"], "selected_season": season(selected), "selected_direction": direction(selected), "match_type": match, "exact_match": match == "exact"})
    return {str(row["scene_id"]) for row in chosen}, audit


def copy_atomic(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    shutil.copy2(source, temporary)
    os.replace(str(temporary), str(destination))


def stage_build(args: argparse.Namespace, root: Path, log: logging.Logger) -> None:
    staging = root / STAGING_NAME
    observations = read_csv(staging / "chips_manifest.csv")
    city_split = read_csv(staging / "city_split_manifest.csv")
    selected_four, audit = {}, []
    for city, frame in observations.groupby("city_slug"):
        selected_four[city], rows = select_four(frame)
        audit.extend(rows)
    for count, name in DATASETS.items():
        dataset = root / name
        if dataset.exists() and args.overwrite and not args.resume:
            shutil.rmtree(dataset)
        for folder in ("image", "mask", "ndsm", "qa", "stats"):
            (dataset / folder).mkdir(parents=True, exist_ok=True)
        selected = observations if count == 8 else observations[observations.apply(lambda row: row["scene_id"] in selected_four[row["city_slug"]], axis=1)]
        dataset_excluded = read_csv(staging / "excluded_observations.csv")
        # A geographic chip is useful only when every required scene passed
        # image QA. Dropping the whole chip prevents an accidental 3/4 or 7/8
        # observation set from changing the scientific weighting.
        complete_chip = selected.groupby("geographic_chip_id")["scene_id"].nunique()
        complete_ids = set(complete_chip[complete_chip == count].index)
        dropped = selected[~selected["geographic_chip_id"].isin(complete_ids)]
        selected = selected[selected["geographic_chip_id"].isin(complete_ids)].copy()
        if not dropped.empty:
            log.warning("%s: dropped %d partial observations from %d incomplete geographic chips", name, len(dropped), dropped["geographic_chip_id"].nunique())
            companion_exclusions = dropped[["observation_id", "city_slug", "scene_id", "geographic_chip_id"]].copy()
            companion_exclusions["reason"] = f"incomplete_{count}_scene_geographic_chip"
            companion_exclusions["coverage"] = np.nan
            dataset_excluded = pd.concat([dataset_excluded, companion_exclusions], ignore_index=True)
        output_rows = []
        for _, row in selected.iterrows():
            obs = str(row["observation_id"])
            pairs = {
                "image": (project_path(Path(str(row["image_path"]))), dataset / "image" / f"{obs}_IMG.tif"),
                "mask": (project_path(Path(str(row["mask_path"]))), dataset / "mask" / f"{obs}_BLG.tif"),
                "ndsm": (project_path(Path(str(row["agl_path"]))), dataset / "ndsm" / f"{obs}_AGL.tif"),
                "qa": (project_path(Path(str(row["qa_path"]))), dataset / "qa" / f"{obs}_QA.tif"),
            }
            for source, destination in pairs.values():
                if not (args.resume and destination.is_file()):
                    copy_atomic(source, destination)
            output = row.to_dict()
            output.update({f"dataset_{key}_path": rel(destination) for key, (_, destination) in pairs.items()})
            output_rows.append(output)
        output = pd.DataFrame(output_rows)
        if output.empty:
            raise RuntimeError(f"{name}: no observations selected")
        all_ids = []
        for split in SPLITS:
            ids = sorted(output.loc[output["split"] == split, "observation_id"].astype(str))
            if not ids:
                raise RuntimeError(f"{name}: {split} split is empty")
            atomic_text(dataset / f"{split}.txt", "".join(f"{value}\n" for value in ids))
            all_ids.extend(ids)
        atomic_text(dataset / "all.txt", "".join(f"{value}\n" for value in sorted(all_ids)))
        atomic_csv(output, dataset / "chips_manifest.csv")
        atomic_csv(city_split, dataset / "city_split_manifest.csv")
        atomic_csv(read_csv(staging / "scene_metadata.csv"), dataset / "scene_metadata.csv")
        atomic_csv(dataset_excluded, dataset / "excluded_observations.csv")
        if count == 4:
            atomic_csv(pd.DataFrame(audit), dataset / "four_scene_selection_audit.csv")
        log.info("%s: materialized %d observations", name, len(output))


def stage_statistics(root: Path, log: logging.Logger) -> None:
    for name in DATASETS.values():
        dataset = root / name
        frame = read_csv(dataset / "chips_manifest.csv")
        train = frame[frame["split"] == "train"]
        sums = np.zeros(4, dtype="float64")
        squares = np.zeros(4, dtype="float64")
        count = 0
        for _, row in train.iterrows():
            with rasterio.open(project_path(Path(str(row["dataset_image_path"])))) as image_src, rasterio.open(project_path(Path(str(row["dataset_qa_path"])))) as qa_src:
                image, qa = image_src.read().astype("float64"), qa_src.read()
            valid = (qa[0] > 0) & (qa[8] == 0) & np.all(np.isfinite(image) & (image > 0), axis=0)
            values = image[:, valid]
            sums += values.sum(axis=1)
            squares += np.square(values).sum(axis=1)
            count += values.shape[1]
        if count == 0:
            raise RuntimeError(f"{name}: no valid training pixels for statistics")
        mean = sums / count
        std = np.sqrt(np.maximum(squares / count - np.square(mean), 0))
        target_values = []
        for _, row in train.drop_duplicates("geographic_chip_id").iterrows():
            with rasterio.open(project_path(Path(str(row["dataset_mask_path"])))) as mask_src, rasterio.open(project_path(Path(str(row["dataset_ndsm_path"])))) as agl_src:
                mask = mask_src.read(1) == 1
                target_values.append(agl_src.read(1)[mask])
        targets = np.concatenate(target_values)
        target_mean, target_std = float(targets.mean()), float(targets.std())
        target_max = float(targets.max())
        target_histogram = np.bincount(np.floor(targets).astype("int64"), minlength=max(1, int(np.ceil(target_max)) + 1))
        image_stats = {"image_mean": mean.astype("float32").tolist(), "image_std": std.astype("float32").tolist(), "stats_split": "train", "channel_order": list(CHANNELS), "valid_pixel_count": count, "training_observations": len(train)}
        ndsm_stats = {"ndsm_positive_mean": target_mean, "ndsm_positive_std": target_std, "ndsm_min": 0.0, "ndsm_max": target_max, "stats_split": "train", "building_pixel_count": int(targets.size), "unique_geographic_chips": int(train["geographic_chip_id"].nunique())}
        atomic_pickle(image_stats, dataset / "stats/image_stats.pickle")
        atomic_pickle(ndsm_stats, dataset / "stats/ndsm_stats.pickle")
        atomic_json(dataset / "stats/statistics_manifest.json", {"image": image_stats, "ndsm": ndsm_stats})
        # The upstream HTC loader reads torch-serialized statistics from the
        # dataset root. Keep these alongside the richer audit metadata.
        atomic_torch([image_stats["image_mean"], image_stats["image_std"]], dataset / "image_stats.pickle")
        atomic_torch([target_mean, target_std, 0.0, target_max, torch.as_tensor(target_histogram)], dataset / "ndsm_stats.pickle")
        log.info("%s: calculated training-only statistics", name)


def signature(path: Path) -> tuple[Any, ...]:
    with rasterio.open(path) as src:
        return src.crs, src.transform, src.width, src.height, src.bounds, src.res


def qa_panels(dataset: Path, frame: pd.DataFrame, count: int) -> None:
    # Plotting is imported only for the final QA stage. Inventory and raster
    # preparation can therefore run and fail clearly even if matplotlib was
    # omitted from a partially installed Windows environment.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = dataset / "visual_qa"
    output.mkdir(parents=True, exist_ok=True)
    for city, rows in frame.groupby("city_slug"):
        for _, row in rows.drop_duplicates("geographic_chip_id").head(count).iterrows():
            obs = str(row["observation_id"])
            with rasterio.open(dataset / "image" / f"{obs}_IMG.tif") as src:
                image = src.read()
            with rasterio.open(dataset / "mask" / f"{obs}_BLG.tif") as src:
                mask = src.read(1)
            with rasterio.open(dataset / "ndsm" / f"{obs}_AGL.tif") as src:
                agl = src.read(1)
            rgb = np.moveaxis(image[:3], 0, -1)
            positive = rgb[rgb > 0]
            low, high = np.percentile(positive, (2, 98)) if positive.size else (0, 1)
            rgb = np.clip((rgb - low) / max(high - low, 1e-6), 0, 1)
            figure, axes = plt.subplots(1, 4, figsize=(14, 4))
            for axis, data, title, cmap in zip(axes, (rgb, image[3], mask, np.where(mask == 1, agl, np.nan)), ("RGB", "NIR", "Building mask", "LiDAR nDSM (m)"), (None, "gray", "gray", "viridis")):
                axis.imshow(data, cmap=cmap); axis.set_title(title); axis.axis("off")
            figure.suptitle(f"{city} | {obs}")
            figure.tight_layout()
            figure.savefig(output / f"{obs}.png", dpi=150)
            plt.close(figure)


def stage_validate(args: argparse.Namespace, root: Path, log: logging.Logger) -> None:
    reports = []
    for expected_scenes, name in DATASETS.items():
        dataset = root / name
        frame = read_csv(dataset / "chips_manifest.csv")
        manifest_ids = set(frame["observation_id"].astype(str))
        split_sets = {split: {line.strip() for line in (dataset / f"{split}.txt").read_text(encoding="utf-8").splitlines() if line.strip()} for split in SPLITS}
        if any(split_sets[a] & split_sets[b] for index, a in enumerate(SPLITS) for b in SPLITS[index + 1:]):
            raise RuntimeError(f"{name}: split overlap")
        if set().union(*split_sets.values()) != manifest_ids:
            raise RuntimeError(f"{name}: split files do not cover manifest")
        if (frame.groupby("city_slug")["split"].nunique() != 1).any():
            raise RuntimeError(f"{name}: a city crosses splits")
        scene_counts = frame.groupby("city_slug")["scene_id"].nunique()
        if (scene_counts != expected_scenes).any():
            raise RuntimeError(f"{name}: incorrect scene counts: {scene_counts[scene_counts != expected_scenes].to_dict()}")
        chip_scene_counts = frame.groupby("geographic_chip_id")["scene_id"].nunique()
        if (chip_scene_counts != expected_scenes).any():
            raise RuntimeError(f"{name}: one or more geographic chips do not have exactly {expected_scenes} observations")
        for _, row in frame.iterrows():
            obs = str(row["observation_id"])
            paths = (dataset / "image" / f"{obs}_IMG.tif", dataset / "mask" / f"{obs}_BLG.tif", dataset / "ndsm" / f"{obs}_AGL.tif", dataset / "qa" / f"{obs}_QA.tif")
            if not all(path.is_file() for path in paths) or len({signature(path) for path in paths}) != 1:
                raise RuntimeError(f"{name}/{obs}: missing or misaligned rasters")
            with rasterio.open(paths[0]) as image, rasterio.open(paths[1]) as mask, rasterio.open(paths[2]) as agl, rasterio.open(paths[3]) as qa:
                if (image.count, mask.count, agl.count, qa.count) != (4, 1, 1, 9) or (image.width, image.height) != (256, 256) or not np.allclose(np.abs(image.res), (3, 3), atol=0.01):
                    raise RuntimeError(f"{name}/{obs}: invalid dimensions, resolution, or band count")
                if not set(np.unique(mask.read(1))).issubset({0, 1}) or float(np.nanmin(agl.read(1))) < 0:
                    raise RuntimeError(f"{name}/{obs}: invalid mask or negative nDSM")
            reports.append({"dataset": name, "observation_id": obs, "city_slug": row["city_slug"], "split": row["split"], "aligned": True, "bands": 4, "width": 256, "height": 256, "resolution_m": 3.0})
        with (dataset / "stats/image_stats.pickle").open("rb") as stream:
            stats = pickle.load(stream)
        if len(stats["image_mean"]) != 4 or len(stats["image_std"]) != 4 or stats["stats_split"] != "train":
            raise RuntimeError(f"{name}: invalid training statistics")
        upstream_mean, upstream_std = torch.load(dataset / "image_stats.pickle", map_location="cpu")
        if len(upstream_mean) != 4 or len(upstream_std) != 4:
            raise RuntimeError(f"{name}: root-level HTC image statistics are not four-channel")
        # Exercise the actual upstream dataset class against one observation
        # from every split. This catches filename or normalization contracts
        # that raster-level checks alone cannot detect.
        external = SCRIPT_DIR / "external/HTC-DC-Net"
        sys.path.insert(0, str(external))
        try:
            from dataloaders import GBHDataset
            for split in SPLITS:
                loader_dataset = GBHDataset(str(dataset), str(dataset / f"{split}.txt"), use_mask=True)
                _, image_tensor, target = loader_dataset[0]
                if image_tensor.shape[0] != 4 or "mask" not in target or "ndsm" not in target:
                    raise RuntimeError(f"{name}: HTC loader contract failed for {split}")
        finally:
            sys.path.remove(str(external))
        qa_panels(dataset, frame, args.qa_chips_per_city)
        atomic_text(dataset / "README.md", f"# {name}\n\nWindows-ready four-channel RGB+NIR HTC dataset with {expected_scenes} independent scene observations per geographic chip. Cities use the deterministic 19/19/6 split with seed {args.seed}; normalization statistics use training cities only.\n")
        log.info("%s: validation passed for %d observations", name, len(frame))
    report = pd.DataFrame(reports)
    atomic_csv(report, root / STAGING_NAME / "alignment_validation_summary.csv")
    for name in DATASETS.values():
        copy_atomic(root / STAGING_NAME / "alignment_validation_summary.csv", root / name / "alignment_validation_summary.csv")


def execute(stage: str, args: argparse.Namespace, root: Path, log: logging.Logger) -> None:
    if stage != "split" and rasterio is None:
        raise RuntimeError(
            "rasterio is not installed. Install requirements-windows-cpu.txt "
            "inside venv_htc_dc_net before running raster stages."
        )
    if stage in ("statistics", "validate") and torch is None:
        raise RuntimeError(
            "PyTorch is not installed. Install requirements-windows-cpu.txt "
            "inside venv_htc_dc_net before creating HTC statistics or validating."
        )
    functions = {"inventory": stage_inventory, "split": stage_split, "canonical-chips": stage_canonical, "align-scenes": stage_align, "build-datasets": stage_build, "statistics": lambda _args, _root, _log: stage_statistics(_root, _log), "validate": stage_validate}
    functions[stage](args, root, log)


def main() -> None:
    args = arguments()
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")
    root = project_path(args.output_root)
    log, log_path = logger_for(root, args.stage)
    status_path = root / STAGING_NAME / "pipeline_status.json"
    status: dict[str, Any] = {"status": "running", "requested_stage": args.stage, "started_utc": datetime.now(timezone.utc).isoformat(), "log_path": rel(log_path), "stages": {}}
    atomic_json(status_path, status)
    try:
        for stage in STAGES if args.stage == "all" else (args.stage,):
            log.info("START stage=%s", stage)
            execute(stage, args, root, log)
            status["stages"][stage] = {"status": "success", "completed_utc": datetime.now(timezone.utc).isoformat()}
            atomic_json(status_path, status)
            log.info("SUCCESS stage=%s", stage)
        status.update(status="success", completed_utc=datetime.now(timezone.utc).isoformat())
        atomic_json(status_path, status)
    except Exception as exc:
        status.update(status="failed", failed_utc=datetime.now(timezone.utc).isoformat(), error=str(exc), traceback=traceback.format_exc())
        atomic_json(status_path, status)
        log.exception("FAILED: %s", exc)
        raise


if __name__ == "__main__":
    main()
