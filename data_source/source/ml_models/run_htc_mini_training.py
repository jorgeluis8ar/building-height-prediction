#!/usr/bin/env python3
"""Run a tiny HTC-DC Net training job and export predictions.

This script is intentionally project-specific. It trains the vendored HTC-DC
Net implementation on a reproducible sample of 5 NYC and 5 LA training chips,
then saves prediction rasters and chip-level metrics for infrastructure QA.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
import torch
import yaml
from skimage import io


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3]
DEFAULT_DATASET_DIR = REPO_ROOT / "data_source/data/ml_models/generated/htc_dc_net/nyc_la_rgb_v1"
SETUP_DIR = REPO_ROOT / "data_source/source/ml_models/htc_dc_net_setup"
HTC_REPO_DIR = REPO_ROOT / "data_source/source/ml_models/external/HTC-DC-Net"
OUTPUT_ROOT = REPO_ROOT / "data_source/data/ml_models/generated/htc_dc_net/mini_training_runs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nyc-chips", type=int, default=5)
    parser.add_argument("--la-chips", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260707)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument(
        "--in-channels",
        type=int,
        default=3,
        help="Number of image input channels. Project datasets support 3, 4, 5, 6, and 12.",
    )
    parser.add_argument(
        "--use-dataset-splits",
        action="store_true",
        help="Train from the dataset train.txt and validate on val.txt instead of creating a mini sample.",
    )
    parser.add_argument(
        "--balanced-batches",
        action="store_true",
        help="Use city-height balanced batches in --use-dataset-splits mode.",
    )
    parser.add_argument(
        "--augmentation-profile",
        choices=["none", "spatial", "spatial_spectral"],
        default="none",
        help="Synchronized augmentation used only in --use-dataset-splits mode.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Explicit AdamW weight decay. The historical implicit AdamW default was 0.01.",
    )
    parser.add_argument("--validation-every", type=int, default=0)
    parser.add_argument("--min-epochs", type=int, default=20)
    parser.add_argument("--early-stopping-patience", type=int, default=4)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Training batch size. Use 32 to match the upstream HTC-DC Net config if memory allows.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Config patience value to record/use in HTC config. The mini runner still uses collapse guards for early stopping.",
    )
    parser.add_argument("--save-predictions-every", type=int, default=0)
    parser.add_argument("--save-checkpoints-every", type=int, default=0)
    parser.add_argument(
        "--height-loss-weighting",
        choices=["none", "log", "bins"],
        default="none",
        help="Weight the masked height MAE loss by target height.",
    )
    parser.add_argument("--height-log-alpha", type=float, default=0.5)
    parser.add_argument("--height-bin-edges", default="10,25,50")
    parser.add_argument("--height-bin-weights", default="1,2,4,8")
    parser.add_argument(
        "--background-loss-weight",
        type=float,
        default=0.0,
        help="Weak L1 penalty on predicted height outside the building mask.",
    )
    parser.add_argument(
        "--mask-predictions",
        action="store_true",
        help="Write prediction rasters with non-building pixels set to prediction-nodata.",
    )
    parser.add_argument("--prediction-nodata", type=float, default=-9999.0)
    parser.add_argument(
        "--sampling-mode",
        choices=["random", "highrise_oversample"],
        default="random",
        help="Training chip sampling strategy.",
    )
    parser.add_argument("--highrise-share", type=float, default=0.5)
    parser.add_argument("--highrise-p95-threshold", type=float, default=30.0)
    parser.add_argument("--highrise-max-threshold", type=float, default=50.0)
    parser.add_argument(
        "--collapse-std-threshold",
        type=float,
        default=0.05,
        help="Flag a chip as collapsed when prediction std over evaluated pixels is below this value in meters.",
    )
    parser.add_argument(
        "--collapse-min-share",
        type=float,
        default=0.8,
        help="Share of saved prediction chips that must be collapsed to trigger stop-on-collapse.",
    )
    parser.add_argument(
        "--collapse-patience",
        type=int,
        default=1,
        help="Number of consecutive saved-prediction checks above collapse-min-share before stopping.",
    )
    parser.add_argument(
        "--stop-on-collapse",
        action="store_true",
        help="Stop training when saved predictions collapse according to the collapse guardrail settings.",
    )
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_train_ids(dataset_dir: Path) -> set[str]:
    train_path = dataset_dir / "train.txt"
    with train_path.open("r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def chip_height_summary(dataset_dir: Path, chip_id: str) -> dict:
    ndsm_path = dataset_dir / "ndsm" / f"{chip_id}_AGL.tif"
    if not ndsm_path.exists():
        raise FileNotFoundError(ndsm_path)
    with rasterio.open(ndsm_path) as src:
        arr = src.read(1, masked=True).astype("float32")
    values = arr.compressed()
    values = values[np.isfinite(values) & (values > 0)]
    if values.size == 0:
        return {"target_p95_m": float("nan"), "target_max_m": float("nan"), "highrise_chip": False}
    p95 = float(np.percentile(values, 95))
    max_height = float(np.max(values))
    return {"target_p95_m": p95, "target_max_m": max_height}


def annotate_highrise_chips(
    dataset_dir: Path,
    rows: list[dict],
    p95_threshold: float,
    max_threshold: float,
) -> list[dict]:
    annotated = []
    for row in rows:
        out = row.copy()
        summary = chip_height_summary(dataset_dir, row["chip_id"])
        out.update(summary)
        out["highrise_chip"] = bool(
            np.isfinite(summary["target_p95_m"])
            and (
                summary["target_p95_m"] >= p95_threshold
                or summary["target_max_m"] >= max_threshold
            )
        )
        annotated.append(out)
    return annotated


def select_city_random(rows: list[dict], count: int, rng: random.Random) -> list[dict]:
    if len(rows) < count:
        raise RuntimeError(f"Requested {count} chips, but only found {len(rows)}")
    return rng.sample(rows, count)


def select_city_highrise_oversample(
    rows: list[dict],
    count: int,
    highrise_share: float,
    rng: random.Random,
) -> list[dict]:
    highrise_rows = [row for row in rows if row.get("highrise_chip")]
    regular_rows = [row for row in rows if not row.get("highrise_chip")]
    if not highrise_rows:
        raise RuntimeError("No high-rise chips available for highrise_oversample.")
    highrise_count = int(round(count * highrise_share))
    regular_count = count - highrise_count
    selected = [rng.choice(highrise_rows) for _ in range(highrise_count)]
    regular_source = regular_rows or rows
    if regular_count <= len(regular_source):
        selected.extend(rng.sample(regular_source, regular_count))
    else:
        selected.extend(rng.choice(regular_source) for _ in range(regular_count))
    rng.shuffle(selected)
    return selected


def select_chips(
    dataset_dir: Path,
    nyc_chips: int,
    la_chips: int,
    seed: int,
    sampling_mode: str = "random",
    highrise_share: float = 0.5,
    highrise_p95_threshold: float = 30.0,
    highrise_max_threshold: float = 50.0,
) -> list[dict]:
    manifest_path = dataset_dir / "chips_manifest.csv"
    train_ids = read_train_ids(dataset_dir)
    by_city = {"new_york_city": [], "los_angeles": []}

    with manifest_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            chip_id = row["chip_id"]
            city = row["source_city"]
            if chip_id in train_ids and city in by_city:
                by_city[city].append(row)

    rng = random.Random(seed)
    selected = []
    for city, count in [("new_york_city", nyc_chips), ("los_angeles", la_chips)]:
        rows = sorted(by_city[city], key=lambda r: r["chip_id"])
        if sampling_mode == "random":
            selected.extend(select_city_random(rows, count, rng))
        else:
            annotated = annotate_highrise_chips(
                dataset_dir=dataset_dir,
                rows=rows,
                p95_threshold=highrise_p95_threshold,
                max_threshold=highrise_max_threshold,
            )
            highrise_n = sum(bool(row.get("highrise_chip")) for row in annotated)
            print(f"{city}: {highrise_n}/{len(annotated)} train chips flagged high-rise")
            selected.extend(select_city_highrise_oversample(annotated, count, highrise_share, rng))

    selected.sort(key=lambda r: (r["source_city"], r["chip_id"]))
    return selected


def create_mini_dataset(selected: list[dict], run_dir: Path, dataset_dir: Path) -> Path:
    mini_dir = run_dir / "mini_dataset"
    for subdir in ["image", "mask", "ndsm"]:
        (mini_dir / subdir).mkdir(parents=True, exist_ok=True)

    chip_ids = [row["chip_id"] for row in selected]
    for chip_id in chip_ids:
        for subdir, suffix in [
            ("image", "_IMG.tif"),
            ("mask", "_BLG.tif"),
            ("ndsm", "_AGL.tif"),
        ]:
            src = dataset_dir / subdir / f"{chip_id}{suffix}"
            dst = mini_dir / subdir / f"{chip_id}{suffix}"
            if not src.exists():
                raise FileNotFoundError(src)
            shutil.copy2(src, dst)

    for split_name in ["train.txt", "val.txt", "test.txt", "all.txt"]:
        with (mini_dir / split_name).open("w", encoding="utf-8") as f:
            for chip_id in chip_ids:
                f.write(f"{chip_id}\n")

    for stats_name in ["image_stats.pickle", "ndsm_stats.pickle"]:
        shutil.copy2(dataset_dir / stats_name, mini_dir / stats_name)
    stats_dir = mini_dir / "stats"
    stats_dir.mkdir(exist_ok=True)
    for stats_name in ["image_stats.pickle", "ndsm_stats.pickle"]:
        shutil.copy2(dataset_dir / "stats" / stats_name, stats_dir / stats_name)

    with (mini_dir / "chips_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=selected[0].keys())
        writer.writeheader()
        writer.writerows(selected)

    return mini_dir


def prepare_cfg(mini_dir: Path, args: argparse.Namespace) -> dict:
    cfg = load_yaml(SETUP_DIR / "configs/nyc_la_rgb_v1.yaml")
    cfg.update(load_yaml(SETUP_DIR / "configs/htcdc_nyc_la_rgb_v1.yaml"))
    cfg.update(
        {
            "data_dir": str(mini_dir.relative_to(REPO_ROOT)),
            "data_split_dirs": str(mini_dir.relative_to(REPO_ROOT)),
            "test_data_split_dirs": [str(mini_dir.relative_to(REPO_ROOT))],
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "max_epochs": args.epochs,
            "seed": args.seed,
            "device": "cpu",
            "overfit": False,
            "restore": False,
            "name": args.run_name or f"mini_nyc{args.nyc_chips}_la{args.la_chips}",
            "in_channels": args.in_channels,
            "height_loss_weighting": args.height_loss_weighting,
            "height_log_alpha": args.height_log_alpha,
            "height_bin_edges": parse_float_list(args.height_bin_edges),
            "height_bin_weights": parse_float_list(args.height_bin_weights),
            "background_loss_weight": args.background_loss_weight,
            "weight_decay": args.weight_decay,
            "augmentation_profile": args.augmentation_profile,
        }
    )
    if args.lr is not None:
        cfg["lr"] = args.lr
    if args.patience is not None:
        cfg["patience"] = args.patience
    return cfg


def import_htc_modules():
    sys.path.insert(0, str(HTC_REPO_DIR))
    from build import get_model_and_optimizer  # noqa: PLC0415
    from dataloaders import get_train_val_dataloaders  # noqa: PLC0415

    return get_model_and_optimizer, get_train_val_dataloaders


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def image_to_channels_first(image_np: np.ndarray) -> np.ndarray:
    if image_np.ndim == 2:
        return image_np[None, :, :]
    if image_np.ndim == 3 and image_np.shape[0] <= 16 and image_np.shape[-1] > 16:
        return image_np
    return image_np.transpose(2, 0, 1)


def load_image_stats(dataset_dir: Path) -> tuple[list[float], list[float]]:
    """Load image normalization stats and fail loudly if the file is malformed."""
    stats_path = dataset_dir / "image_stats.pickle"
    if not stats_path.exists():
        raise FileNotFoundError(stats_path)
    stats = torch.load(stats_path)
    if not isinstance(stats, (list, tuple)) or len(stats) != 2:
        raise RuntimeError(f"Expected {stats_path} to contain [mean, std].")
    mean, std = stats
    if len(mean) != len(std):
        raise RuntimeError(
            f"Image stats mean/std length mismatch in {stats_path}: "
            f"mean={len(mean)}, std={len(std)}"
        )
    return list(mean), list(std)


def validate_dataset_channels(dataset_dir: Path, in_channels: int) -> None:
    """Check that image chips and normalization stats match --in-channels."""
    mean, std = load_image_stats(dataset_dir)
    if len(mean) != in_channels or len(std) != in_channels:
        raise RuntimeError(
            f"--in-channels={in_channels}, but {dataset_dir / 'image_stats.pickle'} "
            f"has mean/std lengths {len(mean)}/{len(std)}."
        )

    manifest_path = dataset_dir / "chips_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    checked = 0
    with manifest_path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            chip_id = row["chip_id"]
            image_path = dataset_dir / "image" / f"{chip_id}_IMG.tif"
            if not image_path.exists():
                raise FileNotFoundError(image_path)
            with rasterio.open(image_path) as src:
                if src.count != in_channels:
                    raise RuntimeError(
                        f"--in-channels={in_channels}, but {image_path} has {src.count} bands."
                    )
            checked += 1
    if checked == 0:
        raise RuntimeError(f"No image chips found in manifest: {manifest_path}")
    print(f"Validated {checked} image chips with {in_channels} channels.")


def train_model(
    cfg: dict,
    epochs: int,
    selected: list[dict],
    run_dir: Path,
    mini_dir: Path,
    save_predictions_every: int,
    save_checkpoints_every: int,
    collapse_std_threshold: float,
    collapse_min_share: float,
    collapse_patience: int,
    stop_on_collapse: bool,
    mask_predictions: bool,
    prediction_nodata: float,
):
    get_model_and_optimizer, get_train_val_dataloaders = import_htc_modules()
    train_loader, _ = get_train_val_dataloaders(cfg)
    model, optimizer = get_model_and_optimizer(cfg)
    model.to(cfg["device"])

    history = []
    collapse_checks = []
    consecutive_collapsed_checks = 0
    stopped_for_collapse = False
    for epoch in range(epochs):
        model.train()
        losses_epoch = []
        for chip_ids, image, gt in train_loader:
            image = image.to(cfg["device"])
            gt = {key: value.to(cfg["device"]) for key, value in gt.items()}
            losses, _ = model(image, gt)
            loss_total = losses["loss_total"]
            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()
            losses_epoch.append(float(loss_total.detach().cpu()))
            history.append(
                {
                    "epoch": epoch + 1,
                    "chip_id": chip_ids[0],
                    "loss_total": float(loss_total.detach().cpu()),
                }
            )
        print(f"epoch {epoch + 1}: mean_loss={np.mean(losses_epoch):.6f}")
        if save_checkpoints_every and (epoch + 1) % save_checkpoints_every == 0:
            torch.save(
                {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "cfg": cfg,
                    "history": history,
                },
                run_dir / f"model_epoch_{epoch + 1:03d}.pth",
            )
        if save_predictions_every and (epoch + 1) % save_predictions_every == 0:
            prediction_rows = export_predictions(
                model=model,
                cfg=cfg,
                selected=selected,
                run_dir=run_dir,
                data_dir=mini_dir,
                output_subdir=f"predictions_epoch_{epoch + 1:03d}",
                collapse_std_threshold=collapse_std_threshold,
                mask_predictions=mask_predictions,
                prediction_nodata=prediction_nodata,
            )
            write_csv(run_dir / f"predictions_summary_epoch_{epoch + 1:03d}.csv", prediction_rows)
            collapse_check = summarize_collapse_check(
                epoch=epoch + 1,
                prediction_rows=prediction_rows,
                collapse_std_threshold=collapse_std_threshold,
                collapse_min_share=collapse_min_share,
            )
            collapse_checks.append(collapse_check)
            write_csv(run_dir / "prediction_collapse_checks.csv", collapse_checks)
            print(
                "prediction collapse check: "
                f"epoch={collapse_check['epoch']} "
                f"collapsed={collapse_check['collapsed_chips']}/{collapse_check['total_chips']} "
                f"share={collapse_check['collapsed_share']:.3f} "
                f"stop_triggered={collapse_check['stop_triggered']}"
            )
            if collapse_check["stop_triggered"]:
                consecutive_collapsed_checks += 1
            else:
                consecutive_collapsed_checks = 0
            if stop_on_collapse and consecutive_collapsed_checks >= collapse_patience:
                stopped_for_collapse = True
                print(
                    "Stopping early because prediction rasters collapsed for "
                    f"{consecutive_collapsed_checks} consecutive saved-prediction checks."
                )
                break

    return model, history, collapse_checks, stopped_for_collapse


def export_predictions(
    model,
    cfg: dict,
    selected: list[dict],
    run_dir: Path,
    data_dir: Path,
    output_subdir: str = "predictions",
    collapse_std_threshold: float = 0.05,
    mask_predictions: bool = False,
    prediction_nodata: float = -9999.0,
) -> list[dict]:
    predictions_dir = run_dir / output_subdir
    predictions_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    model.eval()
    # Keep BatchNorm/dropout in eval mode, but bypass the upstream legacy
    # evaluation metrics path that expects data/split1+/stats.pkl.
    model.training = True

    with torch.no_grad():
        for row in selected:
            chip_id = row["chip_id"]
            image_path = data_dir / "image" / f"{chip_id}_IMG.tif"
            mask_path = data_dir / "mask" / f"{chip_id}_BLG.tif"
            ndsm_path = data_dir / "ndsm" / f"{chip_id}_AGL.tif"

            image_np = io.imread(image_path).astype(np.float32)
            image = torch.tensor(image_to_channels_first(image_np))
            if image.shape[0] != int(cfg["in_channels"]):
                raise RuntimeError(
                    f"Model expects {cfg['in_channels']} channels, but {image_path} "
                    f"loaded with {image.shape[0]} channels."
                )
            mean, std = load_image_stats(data_dir)
            mean_t = torch.tensor(mean, dtype=image.dtype)[:, None, None]
            std_t = torch.tensor(std, dtype=image.dtype)[:, None, None]
            image = ((image - mean_t) / std_t)[None, :, :, :]

            target_np = np.nan_to_num(io.imread(ndsm_path).astype(np.float32)).clip(0)
            mask_np = (io.imread(mask_path).astype(np.float32) > 0).astype(np.float32)
            gt = {
                "ndsm": torch.tensor(target_np)[None, None, :, :],
                "mask": torch.tensor(mask_np)[None, None, :, :],
            }
            losses, pred = model(image.to(cfg["device"]), {k: v.to(cfg["device"]) for k, v in gt.items()})
            pred_np = pred["ndsm"][0, 0].detach().cpu().numpy().astype(np.float32)
            building_mask = mask_np > 0
            pred_to_write = pred_np.copy()
            if mask_predictions:
                pred_to_write = np.where(building_mask, pred_to_write, prediction_nodata).astype(np.float32)

            pred_path = predictions_dir / f"{chip_id}_ndsm_pred.tif"
            with rasterio.open(ndsm_path) as template:
                profile = template.profile.copy()
                profile.update(
                    count=1,
                    dtype="float32",
                    nodata=prediction_nodata if mask_predictions else None,
                    compress="deflate",
                    predictor=2,
                )
                profile.pop("photometric", None)
                with rasterio.open(pred_path, "w", **profile) as dst:
                    dst.write(pred_to_write, 1)

            positive_target = target_np > 0
            eval_mask = building_mask & positive_target
            if eval_mask.any():
                pred_eval = pred_np[eval_mask]
                target_eval = target_np[eval_mask]
                mae = float(np.mean(np.abs(pred_eval - target_eval)))
                rmse = float(np.sqrt(np.mean((pred_eval - target_eval) ** 2)))
                bias = float(np.mean(pred_eval - target_eval))
                pred_mean = float(np.mean(pred_eval))
                pred_min = float(np.min(pred_eval))
                pred_max = float(np.max(pred_eval))
                pred_std = float(np.std(pred_eval))
                pred_unique_values = int(np.unique(pred_eval).size)
                target_mean = float(np.mean(target_eval))
                target_min = float(np.min(target_eval))
                target_max = float(np.max(target_eval))
                target_std = float(np.std(target_eval))
                target_unique_values = int(np.unique(target_eval).size)
                collapse_flag = bool(pred_std < collapse_std_threshold)
            else:
                mae = rmse = bias = float("nan")
                pred_mean = pred_min = pred_max = pred_std = float("nan")
                target_mean = target_min = target_max = target_std = float("nan")
                pred_unique_values = target_unique_values = 0
                collapse_flag = False

            pred_raster_values = pred_to_write
            if mask_predictions:
                pred_raster_values = pred_raster_values[pred_raster_values != prediction_nodata]
            pred_finite = pred_raster_values[np.isfinite(pred_raster_values)]
            if pred_finite.size:
                pred_raster_min = float(np.min(pred_finite))
                pred_raster_max = float(np.max(pred_finite))
                pred_raster_std = float(np.std(pred_finite))
                pred_raster_unique_values = int(np.unique(pred_finite).size)
            else:
                pred_raster_min = pred_raster_max = pred_raster_std = float("nan")
                pred_raster_unique_values = 0

            rows.append(
                {
                    "chip_id": chip_id,
                    "source_city": row["source_city"],
                    "prediction_path": str(pred_path.relative_to(REPO_ROOT)),
                    "prediction_masked_to_buildings": bool(mask_predictions),
                    "loss_total": float(losses["loss_total"].detach().cpu()),
                    "target_positive_pixels": int(positive_target.sum()),
                    "building_pixels": int(building_mask.sum()),
                    "eval_pixels": int(eval_mask.sum()),
                    "target_mean_m": target_mean,
                    "target_min_m": target_min,
                    "target_max_m": target_max,
                    "target_std_m": target_std,
                    "target_unique_values": target_unique_values,
                    "pred_mean_m": pred_mean,
                    "pred_min_m": pred_min,
                    "pred_max_m": pred_max,
                    "pred_std_m": pred_std,
                    "pred_unique_values": pred_unique_values,
                    "pred_raster_min_m": pred_raster_min,
                    "pred_raster_max_m": pred_raster_max,
                    "pred_raster_std_m": pred_raster_std,
                    "pred_raster_unique_values": pred_raster_unique_values,
                    "collapse_std_threshold_m": collapse_std_threshold,
                    "collapse_flag": collapse_flag,
                    "mae_m": mae,
                    "rmse_m": rmse,
                    "bias_m": bias,
                }
            )

    return rows


def summarize_collapse_check(
    epoch: int | str,
    prediction_rows: list[dict],
    collapse_std_threshold: float,
    collapse_min_share: float,
) -> dict:
    collapsed = sum(bool(row["collapse_flag"]) for row in prediction_rows)
    total = len(prediction_rows)
    collapsed_share = collapsed / total if total else 0.0
    pred_stds = [
        float(row["pred_std_m"])
        for row in prediction_rows
        if row["pred_std_m"] == row["pred_std_m"]
    ]
    pred_unique = [int(row["pred_unique_values"]) for row in prediction_rows]
    return {
        "epoch": epoch,
        "total_chips": total,
        "collapsed_chips": collapsed,
        "collapsed_share": collapsed_share,
        "collapse_std_threshold_m": collapse_std_threshold,
        "collapse_min_share": collapse_min_share,
        "mean_pred_std_m": float(np.mean(pred_stds)) if pred_stds else float("nan"),
        "min_pred_std_m": float(np.min(pred_stds)) if pred_stds else float("nan"),
        "max_pred_std_m": float(np.max(pred_stds)) if pred_stds else float("nan"),
        "min_pred_unique_values": min(pred_unique) if pred_unique else 0,
        "max_pred_unique_values": max(pred_unique) if pred_unique else 0,
        "stop_triggered": bool(collapsed_share >= collapse_min_share),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def write_training_loss_summaries(run_dir: Path, history: list[dict]) -> None:
    """Write epoch-level loss CSV and PNG plot."""
    by_epoch: dict[int, list[float]] = {}
    for row in history:
        by_epoch.setdefault(int(row["epoch"]), []).append(float(row["loss_total"]))
    rows = [
        {
            "epoch": epoch,
            "mean_loss_total": float(np.mean(losses)),
            "min_loss_total": float(np.min(losses)),
            "max_loss_total": float(np.max(losses)),
            "batches": len(losses),
        }
        for epoch, losses in sorted(by_epoch.items())
    ]
    write_csv(run_dir / "training_epoch_loss.csv", rows)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot([row["epoch"] for row in rows], [row["mean_loss_total"] for row in rows], marker="o")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean loss_total")
    ax.set_title("HTC-DC Net mini-run training loss")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(run_dir / "training_loss.png", dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    os.environ.setdefault("WANDB_MODE", "offline")
    os.environ.setdefault("TORCH_HOME", "/private/tmp/torch_htc_cache")
    set_seed(args.seed)
    dataset_dir = args.dataset_dir.resolve()
    if not dataset_dir.exists():
        raise FileNotFoundError(dataset_dir)
    validate_dataset_channels(dataset_dir, args.in_channels)

    run_name = args.run_name or (
        f"nyc{args.nyc_chips}_la{args.la_chips}_seed{args.seed}_"
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    run_dir = OUTPUT_ROOT / run_name
    if run_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{run_dir} exists. Use --overwrite or a new --run-name.")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)

    if args.use_dataset_splits:
        if not args.balanced_batches:
            raise RuntimeError("The spatial training workflow requires --balanced-batches.")
        if args.augmentation_profile != "spatial_spectral":
            raise RuntimeError(
                "The requested five-channel workflow requires "
                "--augmentation-profile spatial_spectral."
            )
        if args.validation_every <= 0:
            raise ValueError("--validation-every must be positive in dataset-split mode")
        cfg = prepare_cfg(dataset_dir, args)
        with (run_dir / "config_used.yaml").open("w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        metadata = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "seed": args.seed,
            "epochs": args.epochs,
            "lr": cfg["lr"],
            "batch_size": cfg["batch_size"],
            "num_workers": cfg["num_workers"],
            "save_checkpoints_every": args.save_checkpoints_every,
            "validation_every": args.validation_every,
            "min_epochs": args.min_epochs,
            "early_stopping_patience": args.early_stopping_patience,
            "early_stopping_min_delta": args.early_stopping_min_delta,
            "dataset_dir": str(dataset_dir.relative_to(REPO_ROOT)),
            "use_dataset_splits": True,
            "balanced_batches": True,
            "augmentation_profile": args.augmentation_profile,
            "weight_decay": args.weight_decay,
            "in_channels": args.in_channels,
            "height_loss_weighting": args.height_loss_weighting,
            "height_bin_edges": parse_float_list(args.height_bin_edges),
            "height_bin_weights": parse_float_list(args.height_bin_weights),
            "background_loss_weight": args.background_loss_weight,
            "collapse_std_threshold": args.collapse_std_threshold,
            "collapse_min_share": args.collapse_min_share,
            "stop_on_collapse": args.stop_on_collapse,
        }
        with (run_dir / "run_metadata.json").open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        get_model_and_optimizer, _ = import_htc_modules()
        from htc_spatial_training import run_spatial_training  # noqa: PLC0415

        result = run_spatial_training(
            args=args,
            cfg=cfg,
            dataset_dir=dataset_dir,
            run_dir=run_dir,
            get_model_and_optimizer=get_model_and_optimizer,
            export_predictions=export_predictions,
            summarize_collapse_check=summarize_collapse_check,
            write_csv=write_csv,
            write_training_loss_summaries=write_training_loss_summaries,
        )
        metadata.update(result)
        with (run_dir / "run_metadata.json").open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        print(f"Run directory: {run_dir}")
        print(f"Best checkpoint: {run_dir / 'model_best.pth'}")
        print(f"Best epoch: {result['best_epoch']}")
        print(
            "Best city-balanced validation RMSE: "
            f"{result['best_city_balanced_rmse_m']:.4f} m"
        )
        return

    selected = select_chips(
        dataset_dir=dataset_dir,
        nyc_chips=args.nyc_chips,
        la_chips=args.la_chips,
        seed=args.seed,
        sampling_mode=args.sampling_mode,
        highrise_share=args.highrise_share,
        highrise_p95_threshold=args.highrise_p95_threshold,
        highrise_max_threshold=args.highrise_max_threshold,
    )
    mini_dir = create_mini_dataset(selected, run_dir, dataset_dir)
    cfg = prepare_cfg(mini_dir, args)

    with (run_dir / "config_used.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    with (run_dir / "run_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "seed": args.seed,
                "epochs": args.epochs,
                "nyc_chips": args.nyc_chips,
                "la_chips": args.la_chips,
                "lr": cfg["lr"],
                "batch_size": cfg["batch_size"],
                "num_workers": cfg["num_workers"],
                "patience": cfg.get("patience"),
                "save_predictions_every": args.save_predictions_every,
                "save_checkpoints_every": args.save_checkpoints_every,
                "collapse_std_threshold": args.collapse_std_threshold,
                "collapse_min_share": args.collapse_min_share,
                "collapse_patience": args.collapse_patience,
                "stop_on_collapse": args.stop_on_collapse,
                "dataset_dir": str(dataset_dir.relative_to(REPO_ROOT)),
                "in_channels": args.in_channels,
                "height_loss_weighting": args.height_loss_weighting,
                "height_log_alpha": args.height_log_alpha,
                "height_bin_edges": parse_float_list(args.height_bin_edges),
                "height_bin_weights": parse_float_list(args.height_bin_weights),
                "background_loss_weight": args.background_loss_weight,
                "mask_predictions": args.mask_predictions,
                "prediction_nodata": args.prediction_nodata,
                "sampling_mode": args.sampling_mode,
                "highrise_share": args.highrise_share,
                "highrise_p95_threshold": args.highrise_p95_threshold,
                "highrise_max_threshold": args.highrise_max_threshold,
            },
            f,
            indent=2,
        )

    model, history, collapse_checks, stopped_for_collapse = train_model(
        cfg=cfg,
        epochs=args.epochs,
        selected=selected,
        run_dir=run_dir,
        mini_dir=mini_dir,
        save_predictions_every=args.save_predictions_every,
        save_checkpoints_every=args.save_checkpoints_every,
        collapse_std_threshold=args.collapse_std_threshold,
        collapse_min_share=args.collapse_min_share,
        collapse_patience=args.collapse_patience,
        stop_on_collapse=args.stop_on_collapse,
        mask_predictions=args.mask_predictions,
        prediction_nodata=args.prediction_nodata,
    )
    torch.save(
        {
            "state_dict": model.state_dict(),
            "cfg": cfg,
            "history": history,
            "collapse_checks": collapse_checks,
            "stopped_for_collapse": stopped_for_collapse,
        },
        run_dir / "model_last.pth",
    )
    write_csv(run_dir / "training_history.csv", history)
    write_training_loss_summaries(run_dir, history)
    write_csv(run_dir / "prediction_collapse_checks.csv", collapse_checks)

    prediction_rows = export_predictions(
        model=model,
        cfg=cfg,
        selected=selected,
        run_dir=run_dir,
        data_dir=mini_dir,
        output_subdir="predictions",
        collapse_std_threshold=args.collapse_std_threshold,
        mask_predictions=args.mask_predictions,
        prediction_nodata=args.prediction_nodata,
    )
    write_csv(run_dir / "predictions_summary.csv", prediction_rows)
    final_check = summarize_collapse_check(
        epoch="final",
        prediction_rows=prediction_rows,
        collapse_std_threshold=args.collapse_std_threshold,
        collapse_min_share=args.collapse_min_share,
    )
    write_csv(run_dir / "prediction_collapse_final_check.csv", [final_check])

    print(f"Run directory: {run_dir}")
    print(f"Predictions: {run_dir / 'predictions'}")
    print(f"Summary CSV: {run_dir / 'predictions_summary.csv'}")


if __name__ == "__main__":
    main()
