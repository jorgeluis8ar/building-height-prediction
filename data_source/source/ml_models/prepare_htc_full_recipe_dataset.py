#!/usr/bin/env python3
"""Create a train-normalized full-recipe view of the confirmed RGB+NIR dataset."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import shutil

import numpy as np
import rasterio
import torch


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3]
DEFAULT_SOURCE = REPO_ROOT / "data_source/data/ml_models/generated/htc_dc_net/nyc_la_off_nadir_rgb_nir_v1"
DEFAULT_OUTPUT = REPO_ROOT / "data_source/data/ml_models/generated/htc_dc_net/nyc_la_off_nadir_rgb_nir_full_recipe_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def link_or_copy(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    for subdir in ("image", "mask", "ndsm", "stats"):
        (output_dir / subdir).mkdir()
    for split in ("train.txt", "val.txt", "test.txt", "all.txt", "chips_manifest.csv"):
        shutil.copy2(source_dir / split, output_dir / split)
    for subdir, suffix in (("image", "_IMG.tif"), ("mask", "_BLG.tif"), ("ndsm", "_AGL.tif")):
        for source in sorted((source_dir / subdir).glob(f"*{suffix}")):
            link_or_copy(source, output_dir / subdir / source.name)

    train_ids = [
        line.strip()
        for line in (source_dir / "train.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    sums = np.zeros(4, dtype="float64")
    squares = np.zeros(4, dtype="float64")
    count = 0
    for chip_id in train_ids:
        with rasterio.open(source_dir / "image" / f"{chip_id}_IMG.tif") as src:
            values = src.read().astype("float64").reshape(4, -1)
        sums += values.sum(axis=1)
        squares += (values**2).sum(axis=1)
        count += values.shape[1]
    mean = (sums / count).tolist()
    std = np.sqrt(squares / count - np.asarray(mean) ** 2).tolist()
    torch.save([mean, std], output_dir / "image_stats.pickle")
    torch.save([mean, std], output_dir / "stats/image_stats.pickle")
    for path in (source_dir / "ndsm_stats.pickle", source_dir / "stats/ndsm_stats.pickle"):
        destination = output_dir / path.relative_to(source_dir)
        shutil.copy2(path, destination)

    with (source_dir / "chips_manifest.csv").open("r", newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    report = {
        "source_dataset": str(source_dir.relative_to(REPO_ROOT)),
        "output_dataset": str(output_dir.relative_to(REPO_ROOT)),
        "split_membership": "copied_unchanged",
        "train_chips": len(train_ids),
        "training_city_counts": {
            city: sum(row["split"] == "train" and row["source_city"] == city for row in rows)
            for city in ("new_york_city", "los_angeles")
        },
        "channel_order": ["red", "green", "blue", "nir"],
        "normalization_scope": "training_split_only",
        "image_mean": mean,
        "image_std": std,
        "storage": "hardlinks_when_supported_copy_fallback",
    }
    (output_dir / "dataset_derivation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output_dir / "README.md").write_text(
        "# NYC + LA Off-Nadir RGB+NIR Full-Recipe Dataset\n\n"
        "This dataset preserves every chip and split from `nyc_la_off_nadir_rgb_nir_v1`. "
        "Its only analytical change is recomputing four-band normalization statistics "
        "from the 171 training chips alone, avoiding validation/test leakage. Raster "
        "files are hard-linked when supported and copied otherwise.\n\n"
        f"- Training: {len(train_ids)} chips (76 NYC, 95 LA)\n"
        "- Validation: 36 chips\n"
        "- Test: 37 chips\n"
        "- Channel order: red, green, blue, NIR\n"
        "- Resolution: 3 m\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

