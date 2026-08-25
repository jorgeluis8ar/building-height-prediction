#!/usr/bin/env python3
"""Create the versioned portable inference bundle for the selected model."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import torch
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
MODEL_ID = "nyc76_la95_offnadir_rgbnir_4ch_lowrise_binweighted_bg005_seed20260721_epoch50_guarded"
RUN_DIR = PROJECT_ROOT / "data_source/data/ml_models/generated/htc_dc_net/mini_training_runs" / MODEL_ID
DATASET_DIR = PROJECT_ROOT / "data_source/data/ml_models/generated/htc_dc_net/nyc_la_off_nadir_rgb_nir_v1"
DEFAULT_OUTPUT = PROJECT_ROOT / "data_source/data/ml_models/generated/htc_dc_net/selected_model_inference_bundle_v1"
DEFAULT_EFFICIENTNET_SOURCE = Path.home() / ".cache/torch/hub/rwightman_gen-efficientnet-pytorch_master"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--efficientnet-source", type=Path, default=DEFAULT_EFFICIENTNET_SOURCE)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file() and "__pycache__" not in item.parts):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve() if args.output_dir.is_absolute() else (PROJECT_ROOT / args.output_dir).resolve()
    try:
        output.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise ValueError("Bundle output must remain inside the project repository.") from exc
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"Bundle already exists: {output}. Pass --overwrite to replace it.")
        shutil.rmtree(output)
    output.mkdir(parents=True)

    source_weights = RUN_DIR / "model_epoch_050.pth"
    source_stats = DATASET_DIR / "image_stats.pickle"
    source_ndsm_stats = DATASET_DIR / "ndsm_stats.pickle"
    source_predictor = SCRIPT_DIR / "selected_model_inference/predict_planetscope.py"
    source_requirements = SCRIPT_DIR / "htc_dc_net_setup/requirements-windows-cpu.txt"
    efficientnet_source = args.efficientnet_source.resolve()
    for path in (source_weights, source_stats, source_ndsm_stats, source_predictor, source_requirements):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not (efficientnet_source / "hubconf.py").is_file():
        raise FileNotFoundError(
            f"Exact EfficientNet Torch Hub source is missing: {efficientnet_source}. "
            "Pass --efficientnet-source."
        )

    expected_hash = "f6f43953905a5abf209bc8501cf8cc5af070256989f37b6f637878bc9c946331"
    actual_hash = sha256(source_weights)
    if actual_hash != expected_hash:
        raise RuntimeError(f"Selected checkpoint hash mismatch: {actual_hash}")
    checkpoint = torch.load(source_weights, map_location="cpu")
    if not {"state_dict", "cfg"}.issubset(checkpoint):
        raise RuntimeError("Selected checkpoint lacks state_dict or cfg.")
    first_layer = checkpoint["state_dict"].get("model.encoder.original_model.conv_stem.weight")
    if first_layer is None or tuple(first_layer.shape) != (32, 4, 3, 3):
        raise RuntimeError("Checkpoint is not the expected four-channel EfficientNet-B0 model.")
    mean, std = torch.load(source_stats, map_location="cpu")
    if len(mean) != 4 or len(std) != 4:
        raise RuntimeError("Selected normalization file is not four-channel RGB+NIR.")

    shutil.copy2(source_weights, output / "model_weights.pth")
    shutil.copy2(source_stats, output / "image_stats.pickle")
    shutil.copy2(source_ndsm_stats, output / "ndsm_stats.pickle")
    shutil.copy2(source_predictor, output / "predict_planetscope.py")
    shutil.copy2(source_requirements, output / "requirements-windows-cpu.txt")
    shutil.copytree(
        efficientnet_source,
        output / "efficientnet_source",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    config = {
        "model": "htcdc",
        "backbone": "efficientnetb0",
        "in_channels": 4,
        "channel_order": ["red", "green", "blue", "nir"],
        "image_size": 256,
        "input_resolution_m": 3.0,
        "output": "above-ground-level height in meters",
        "normalization_mean": [float(value) for value in mean],
        "normalization_std": [float(value) for value in std],
        "checkpoint_epoch": 50,
    }
    (output / "model_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    manifest = {
        "bundle_schema_version": 1,
        "bundle_id": "selected_model_inference_bundle_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": {
            "model_id": MODEL_ID,
            "architecture": "HTC-DC Net",
            "backbone": "EfficientNet-B0",
            "input_channels": 4,
            "channel_order": ["red", "green", "blue", "nir"],
            "checkpoint_epoch": 50,
            "checkpoint_sha256": actual_hash,
            "checkpoint_size_bytes": source_weights.stat().st_size,
            "state_dict_tensor_count": len(checkpoint["state_dict"]),
            "efficientnet_source_sha256": tree_sha256(output / "efficientnet_source"),
        },
        "input_contract": {
            "format": "GeoTIFF",
            "product": "PlanetScope surface reflectance",
            "band_order": ["red", "green", "blue", "nir"],
            "pixel_resolution_m": 3.0,
            "window_pixels": 256,
            "radiometry": "same scaling as training PlanetScope AnalyticMS SR imagery",
        },
        "output_contract": {"variable": "predicted_agl_m", "units": "meters", "minimum": 0.0},
        "files": {
            "model_weights": "model_weights.pth",
            "model_config": "model_config.yaml",
            "image_stats": "image_stats.pickle",
            "ndsm_stats": "ndsm_stats.pickle",
            "predictor": "predict_planetscope.py",
            "requirements": "requirements-windows-cpu.txt",
            "efficientnet_source": "efficientnet_source/",
        },
        "code_dependency": "data_source/source/ml_models/external/HTC-DC-Net from the project repository",
    }
    (output / "inference_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output / "README.md").write_text(
        "# Selected HTC-DC Net Inference Bundle V1\n\n"
        "This bundle transfers the selected trained RGB+NIR model for inference on another computer. "
        "It does not contain training data. Input must be a four-band PlanetScope surface-reflectance "
        "GeoTIFF ordered red, green, blue, NIR and compatible with the training radiometry.\n\n"
        "## Windows CMD\n\n"
        "```cmd\n"
        "data_source\\source\\ml_models\\venv_htc_dc_net\\Scripts\\python.exe "
        "data_source\\data\\ml_models\\generated\\htc_dc_net\\selected_model_inference_bundle_v1\\predict_planetscope.py ^\n"
        "  --input path\\to\\planet_rgb_nir.tif ^\n"
        "  --output path\\to\\predicted_agl_m.tif ^\n"
        "  --device cpu\n"
        "```\n\n"
        "The script verifies the checkpoint SHA-256, loads all learned parameters strictly, normalizes "
        "RGB+NIR with the saved training statistics, predicts overlapping 256-pixel windows, and writes "
        "a georeferenced AGL-height raster in meters. Building footprints are not model inputs; they may "
        "be applied later as optional post-processing.\n",
        encoding="utf-8",
    )
    print(f"Created bundle: {output.relative_to(PROJECT_ROOT)}")
    print(f"Checkpoint SHA-256: {actual_hash}")
    print(f"Checkpoint tensors: {len(checkpoint['state_dict'])}")


if __name__ == "__main__":
    main()
