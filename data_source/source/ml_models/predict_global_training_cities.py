#!/usr/bin/env python3
"""Run the selected HTC-DC Net model over every PlanetScope scene by city.

The expected input layout is::

    <scenes_root>/<city>/<order_uuid>/PSScene/<Planet files>

Each surface-reflectance scene is converted to RGB+NIR, predicted independently,
and then combined into a city-level median height raster. The script is designed
for long Windows CPU runs: completed predictions are reused unless --overwrite
is supplied, and every failure is written to the run manifest before exiting.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject


NODATA = -9999.0
SCENE_PATTERN = re.compile(r"^(?P<scene_id>.+?)_3B_AnalyticMS_SR(?:_8b)?_clip\.tif$", re.IGNORECASE)


def repository_root() -> Path:
    """Find the repository independently of the terminal's working directory."""
    for parent in Path(__file__).resolve().parents:
        if (parent / ".git").exists() and (parent / "data_source").is_dir():
            return parent
    raise RuntimeError("Could not find the repository root above this script.")


def project_path(value: Path, root: Path) -> Path:
    """Interpret command-line paths relative to the repository root."""
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenes-root",
        type=Path,
        default=Path("data_source/data/planet_imagery/source/global_training"),
        help="Folder whose direct subfolders are cities.",
    )
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        default=Path("data_source/data/ml_models/generated/htc_dc_net/selected_model_inference_bundle_v1"),
        help="Complete selected_model_inference_bundle_v1 folder.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data_source/data/ml_models/generated/htc_dc_net/global_training_predictions_v1"),
    )
    parser.add_argument("--expected-scenes-per-city", type=int, default=8)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--city", action="append", help="Process only this city folder; repeat as needed.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and list work without inference.")
    parser.add_argument("--overwrite", action="store_true", help="Recreate existing scene predictions.")
    parser.add_argument(
        "--keep-rgbnir",
        action="store_true",
        help="Keep prepared four-band inputs. By default they are temporary to save disk space.",
    )
    return parser.parse_args()


def discover_scenes(city_dir: Path) -> List[Tuple[str, Path]]:
    """Find one analytic surface-reflectance raster for each unique scene ID."""
    scenes: Dict[str, Path] = {}
    for path in sorted(city_dir.rglob("*.tif")):
        match = SCENE_PATTERN.match(path.name)
        if not match:
            continue
        scene_id = match.group("scene_id")
        if scene_id in scenes:
            raise RuntimeError(
                f"Duplicate surface-reflectance products for {scene_id} in {city_dir}: "
                f"{scenes[scene_id]} and {path}"
            )
        scenes[scene_id] = path
    return sorted(scenes.items())


def verify_bundle(bundle_dir: Path) -> Path:
    """Verify the selected checkpoint once before processing hundreds of scenes."""
    manifest_path = bundle_dir / "inference_manifest.json"
    predictor = bundle_dir / "predict_planetscope.py"
    if not manifest_path.is_file() or not predictor.is_file():
        raise FileNotFoundError(
            f"Incomplete inference bundle at {bundle_dir}; manifest or predictor is missing."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    weights = bundle_dir / manifest["files"]["model_weights"]
    digest = hashlib.sha256()
    with weights.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    expected = manifest["model"]["checkpoint_sha256"]
    if digest.hexdigest() != expected:
        raise RuntimeError("The model checkpoint checksum does not match the selected model manifest.")
    return predictor


def prepare_rgbnir(source_path: Path, output_path: Path) -> None:
    """Convert Planet's native band order to the model's RGB+NIR channel order."""
    with rasterio.open(source_path) as source:
        if source.count == 8:
            indexes = (6, 4, 2, 8)  # Planet SuperDove: red, green, blue, NIR.
        elif source.count == 4:
            indexes = (3, 2, 1, 4)  # Legacy PlanetScope: red, green, blue, NIR.
        else:
            raise RuntimeError(f"Expected a 4- or 8-band PlanetScope raster, found {source.count}: {source_path}")
        data = source.read(indexes)
        profile = source.profile.copy()
        profile.update(count=4, compress="deflate", predictor=2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, "w", **profile) as destination:
        destination.write(data)
        for index, name in enumerate(("red", "green", "blue", "nir"), start=1):
            destination.set_band_description(index, name)
        destination.update_tags(source_planetscope_file=str(source_path), channel_order="red,green,blue,nir")


def prediction_is_valid(path: Path) -> bool:
    """A reusable output must open and contain spatially varying valid values."""
    if not path.is_file():
        return False
    try:
        with rasterio.open(path) as source:
            values = source.read(1, masked=True)
            valid = values.compressed()
            return source.count == 1 and valid.size > 0 and np.all(np.isfinite(valid))
    except rasterio.errors.RasterioError:
        return False


def run_prediction(
    predictor: Path,
    rgbnir_path: Path,
    output_path: Path,
    device: str,
    stride: int,
) -> None:
    """Call the tested bundle predictor with the current Python environment."""
    command = [
        sys.executable,
        str(predictor),
        "--input",
        str(rgbnir_path),
        "--output",
        str(output_path),
        "--device",
        device,
        "--stride",
        str(stride),
        "--skip-checksum",  # The wrapper already checked the checkpoint once.
    ]
    subprocess.run(command, check=True)
    if not prediction_is_valid(output_path):
        raise RuntimeError(f"Inference did not create a valid prediction: {output_path}")


def combine_city_predictions(
    prediction_paths: Sequence[Path],
    median_path: Path,
    count_path: Path,
) -> None:
    """Align predictions to the first scene and calculate a per-pixel median."""
    if not prediction_paths:
        raise RuntimeError("Cannot combine an empty prediction list.")
    with rasterio.open(prediction_paths[0]) as reference:
        profile = reference.profile.copy()
        reference_crs = reference.crs
        reference_transform = reference.transform
        shape = (reference.height, reference.width)

    aligned = []
    for path in prediction_paths:
        with rasterio.open(path) as source:
            destination = np.full(shape, np.nan, dtype="float32")
            source_values = source.read(1)
            source_values = np.where(source_values == source.nodata, np.nan, source_values)
            reproject(
                source=source_values,
                destination=destination,
                src_transform=source.transform,
                src_crs=source.crs,
                dst_transform=reference_transform,
                dst_crs=reference_crs,
                src_nodata=np.nan,
                dst_nodata=np.nan,
                resampling=Resampling.bilinear,
            )
            aligned.append(destination)

    stack = np.stack(aligned)
    valid_count = np.sum(np.isfinite(stack), axis=0).astype("uint8")
    with np.errstate(all="ignore"):
        median = np.nanmedian(stack, axis=0).astype("float32")
    median[valid_count == 0] = NODATA

    median_path.parent.mkdir(parents=True, exist_ok=True)
    median_profile = profile.copy()
    median_profile.update(count=1, dtype="float32", nodata=NODATA, compress="deflate", predictor=2)
    with rasterio.open(median_path, "w", **median_profile) as destination:
        destination.write(median, 1)
        destination.set_band_description(1, "median_predicted_agl_m")
        destination.update_tags(units="meters", aggregation="median", scenes=len(prediction_paths))

    count_profile = profile.copy()
    count_profile.update(count=1, dtype="uint8", nodata=0, compress="deflate", predictor=1)
    with rasterio.open(count_path, "w", **count_profile) as destination:
        destination.write(valid_count, 1)
        destination.set_band_description(1, "valid_scene_count")


def write_manifest(path: Path, rows: List[dict]) -> None:
    """Rewrite the manifest after every scene so interrupted runs remain auditable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "city",
        "scene_id",
        "source_raster",
        "prediction_raster",
        "status",
        "elapsed_minutes",
        "message",
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    root = repository_root()
    scenes_root = project_path(args.scenes_root, root)
    bundle_dir = project_path(args.bundle_dir, root)
    output_dir = project_path(args.output_dir, root)
    if not scenes_root.is_dir():
        raise FileNotFoundError(f"Scenes root does not exist: {scenes_root}")
    if args.expected_scenes_per_city < 1:
        raise ValueError("--expected-scenes-per-city must be at least one.")

    predictor = verify_bundle(bundle_dir)
    requested = set(args.city or [])
    city_dirs = sorted(path for path in scenes_root.iterdir() if path.is_dir())
    if requested:
        city_dirs = [path for path in city_dirs if path.name in requested]
        missing = requested - {path.name for path in city_dirs}
        if missing:
            raise FileNotFoundError(f"Requested city folders were not found: {sorted(missing)}")
    if not city_dirs:
        raise RuntimeError(f"No city folders found under {scenes_root}")

    discovered = {city.name: discover_scenes(city) for city in city_dirs}
    wrong_counts = {city: len(scenes) for city, scenes in discovered.items() if len(scenes) != args.expected_scenes_per_city}
    print(f"Cities found: {len(city_dirs)}")
    print(f"Surface-reflectance scenes found: {sum(map(len, discovered.values()))}")
    if wrong_counts:
        raise RuntimeError(
            f"Scene-count validation failed; expected {args.expected_scenes_per_city} per city, "
            f"found mismatches: {wrong_counts}"
        )
    if args.dry_run:
        for city, scenes in discovered.items():
            print(f"{city}: {len(scenes)} scenes")
        print("Dry run passed. No prediction files were created.")
        return

    manifest_path = output_dir / "inference_manifest.csv"
    manifest_rows: List[dict] = []
    run_started = time.perf_counter()
    total_scenes = sum(map(len, discovered.values()))
    completed_scenes = 0
    for city_index, city_dir in enumerate(city_dirs, start=1):
        city = city_dir.name
        scenes = discovered[city]
        print(f"\nCITY {city_index}/{len(city_dirs)}: {city}", flush=True)
        city_output = output_dir / city
        predictions: List[Path] = []
        for scene_index, (scene_id, source_path) in enumerate(scenes, start=1):
            prediction_path = city_output / "scene_predictions" / f"{scene_id}_predicted_agl_m.tif"
            row = {
                "city": city,
                "scene_id": scene_id,
                "source_raster": str(source_path.relative_to(root)),
                "prediction_raster": str(prediction_path.relative_to(root)),
                "status": "started",
                "elapsed_minutes": "",
                "message": "",
            }
            manifest_rows.append(row)
            write_manifest(manifest_path, manifest_rows)
            scene_started = time.perf_counter()
            try:
                if prediction_is_valid(prediction_path) and not args.overwrite:
                    row["status"] = "reused"
                    print(f"  Scene {scene_index}/{len(scenes)} reused: {scene_id}", flush=True)
                else:
                    print(f"  Scene {scene_index}/{len(scenes)} predicting: {scene_id}", flush=True)
                    if args.keep_rgbnir:
                        rgbnir_path = city_output / "prepared_rgbnir" / f"{scene_id}_RGBNIR.tif"
                        prepare_rgbnir(source_path, rgbnir_path)
                        run_prediction(predictor, rgbnir_path, prediction_path, args.device, args.stride)
                    else:
                        with tempfile.TemporaryDirectory(prefix="htc_rgbnir_") as temporary:
                            rgbnir_path = Path(temporary) / f"{scene_id}_RGBNIR.tif"
                            prepare_rgbnir(source_path, rgbnir_path)
                            run_prediction(predictor, rgbnir_path, prediction_path, args.device, args.stride)
                    row["status"] = "completed"
                predictions.append(prediction_path)
            except Exception as error:
                row["status"] = "failed"
                row["message"] = str(error)
                write_manifest(manifest_path, manifest_rows)
                raise
            row["elapsed_minutes"] = f"{(time.perf_counter() - scene_started) / 60:.2f}"
            completed_scenes += 1
            elapsed = time.perf_counter() - run_started
            remaining_seconds = (elapsed / completed_scenes) * (total_scenes - completed_scenes)
            print(
                f"  Overall progress: {completed_scenes}/{total_scenes} scenes; "
                f"estimated remaining time {remaining_seconds / 3600:.1f} hours",
                flush=True,
            )
            write_manifest(manifest_path, manifest_rows)

        median_path = city_output / f"{city}_median_predicted_agl_m_{len(predictions)}scenes.tif"
        count_path = city_output / f"{city}_valid_scene_count.tif"
        print("  Combining scene predictions with the median...", flush=True)
        combine_city_predictions(predictions, median_path, count_path)
        print(f"  Eight scene rasters: {city_output / 'scene_predictions'}", flush=True)
        print(f"  Ninth height raster (median): {median_path}", flush=True)

    summary = {
        "cities_completed": len(city_dirs),
        "scenes_completed": len(manifest_rows),
        "expected_scenes_per_city": args.expected_scenes_per_city,
        "aggregation": "per-pixel median after alignment to each city's first scene",
        "height_units": "meters above ground level",
        "height_rasters_per_city": args.expected_scenes_per_city + 1,
        "elapsed_hours": round((time.perf_counter() - run_started) / 3600, 3),
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nCompleted {len(city_dirs)} cities. Outputs: {output_dir}")


if __name__ == "__main__":
    main()
