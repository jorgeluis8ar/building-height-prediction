"""Balanced, augmented training support for spatially split HTC datasets."""

from __future__ import annotations

from collections import Counter
import csv
import math
from pathlib import Path
import random
from typing import Any, Callable

import numpy as np
import rasterio
from skimage.measure import label as connected_components
import torch
from torch.utils.data import BatchSampler, DataLoader, Dataset


CITIES = ("los_angeles", "new_york_city")
CATEGORIES = ("lowrise", "midrise", "highrise")


def read_ids(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(path)
    values = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not values:
        raise RuntimeError(f"Empty split file: {path}")
    return values


def read_manifest(dataset_dir: Path) -> dict[str, dict[str, str]]:
    path = dataset_dir / "chips_manifest.csv"
    with path.open("r", newline="", encoding="utf-8") as stream:
        rows = {row["chip_id"]: row for row in csv.DictReader(stream)}
    if not rows:
        raise RuntimeError(f"Empty manifest: {path}")
    return rows


def load_stats(dataset_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    mean, std = torch.load(dataset_dir / "image_stats.pickle")
    mean_array = np.asarray(mean, dtype="float32")
    std_array = np.asarray(std, dtype="float32")
    if mean_array.size != 5 or std_array.size != 5:
        raise RuntimeError("Spatial five-channel training requires five-value normalization stats")
    if np.any(std_array <= 0):
        raise RuntimeError(f"Invalid normalization standard deviations: {std_array}")
    return mean_array, std_array


class SpatialHTCDataset(Dataset):
    """Load aligned HTC chips and apply synchronized training augmentation."""

    def __init__(
        self,
        dataset_dir: Path,
        rows: list[dict[str, str]],
        augment_spatial: bool,
        augment_spectral: bool,
    ) -> None:
        self.dataset_dir = dataset_dir
        self.rows = rows
        self.augment_spatial = augment_spatial
        self.augment_spectral = augment_spectral
        self.mean, self.std = load_stats(dataset_dir)

    def __len__(self) -> int:
        return len(self.rows)

    def _spatial_transform(
        self, image: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rotations = random.randrange(4)
        if rotations:
            image = torch.rot90(image, rotations, dims=(-2, -1))
            target = torch.rot90(target, rotations, dims=(-2, -1))
            mask = torch.rot90(mask, rotations, dims=(-2, -1))
        if random.random() < 0.5:
            image = torch.flip(image, dims=(-1,))
            target = torch.flip(target, dims=(-1,))
            mask = torch.flip(mask, dims=(-1,))
        if random.random() < 0.5:
            image = torch.flip(image, dims=(-2,))
            target = torch.flip(target, dims=(-2,))
            mask = torch.flip(mask, dims=(-2,))
        return image, target, mask

    def __getitem__(self, index: int):
        row = self.rows[index]
        chip_id = row["chip_id"]
        with rasterio.open(self.dataset_dir / "image" / f"{chip_id}_IMG.tif") as src:
            image_np = src.read().astype("float32")
        with rasterio.open(self.dataset_dir / "ndsm" / f"{chip_id}_AGL.tif") as src:
            target_np = np.nan_to_num(src.read(1).astype("float32"), nan=0.0).clip(0)
        with rasterio.open(self.dataset_dir / "mask" / f"{chip_id}_BLG.tif") as src:
            mask_np = (src.read(1) > 0).astype("float32")

        image = torch.from_numpy(image_np)
        target = torch.from_numpy(target_np[None, :, :])
        mask = torch.from_numpy(mask_np[None, :, :])
        if self.augment_spectral:
            common_gain = random.uniform(0.90, 1.10)
            band_gains = torch.tensor(
                [random.uniform(0.97, 1.03) for _ in range(4)], dtype=image.dtype
            )[:, None, None]
            image[:4] = torch.clamp(image[:4] * common_gain * band_gains, min=0)
        if self.augment_spatial:
            image, target, mask = self._spatial_transform(image, target, mask)
        if not torch.equal((image[4] > 0).to(mask.dtype), mask[0]):
            raise RuntimeError(f"Input mask channel lost alignment for {chip_id}")
        if not torch.all((image[4] == 0) | (image[4] == 1)):
            raise RuntimeError(f"Input mask channel became non-binary for {chip_id}")

        mean = torch.from_numpy(self.mean)[:, None, None]
        std = torch.from_numpy(self.std)[:, None, None]
        image = (image - mean) / std
        return chip_id, image, {"ndsm": target, "mask": mask}


class CityHeightBatchSampler(BatchSampler):
    """Yield 4 LA and 4 NYC chips with a 2/1/1 low/mid/high composition."""

    def __init__(self, rows: list[dict[str, str]], seed: int, batch_size: int = 8) -> None:
        if batch_size != 8:
            raise ValueError("Balanced city-height batches require batch_size=8")
        self.seed = seed
        self.epoch = 0
        self.batch_count = math.ceil(len(rows) / batch_size)
        self.exposures: Counter[str] = Counter()
        self.pools: dict[tuple[str, str], list[int]] = {}
        self.rows = rows
        for city in CITIES:
            for category in CATEGORIES:
                indexes = [
                    index
                    for index, row in enumerate(rows)
                    if row["source_city"] == city and row["height_category"] == category
                ]
                if not indexes:
                    raise RuntimeError(f"Missing training pool: city={city}, category={category}")
                self.pools[(city, category)] = indexes

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.batch_count

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        pattern = ("lowrise", "lowrise", "midrise", "highrise")
        for _ in range(self.batch_count):
            indexes = []
            for city in CITIES:
                for category in pattern:
                    indexes.append(rng.choice(self.pools[(city, category)]))
            rng.shuffle(indexes)
            for index in indexes:
                self.exposures[self.rows[index]["chip_id"]] += 1
            yield indexes


def regression_metrics(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    residual = y - x
    slope, intercept = np.polyfit(x, y, deg=1) if x.size >= 2 else (float("nan"), float("nan"))
    fitted = intercept + slope * x
    ss_res = float(np.sum((y - fitted) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return {
        "buildings": int(x.size),
        "rmse_m": float(np.sqrt(np.mean(residual**2))),
        "mae_m": float(np.mean(np.abs(residual))),
        "bias_m": float(np.mean(residual)),
        "r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "slope": float(slope),
        "intercept": float(intercept),
    }


def validation_metrics(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
    minimum_component_pixels: int = 3,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    model.eval()
    model.training = True
    values: dict[str, dict[str, list[float]]] = {
        city: {"target": [], "prediction": []} for city in CITIES
    }
    chip_rows = []
    with torch.no_grad():
        for chip_ids, image, gt in loader:
            image = image.to(device)
            gt_device = {key: value.to(device) for key, value in gt.items()}
            _, prediction = model(image, gt_device)
            pred_batch = prediction["ndsm"][:, 0].detach().cpu().numpy()
            target_batch = gt["ndsm"][:, 0].numpy()
            mask_batch = gt["mask"][:, 0].numpy() > 0
            for batch_index, chip_id in enumerate(chip_ids):
                city = "new_york_city" if chip_id.startswith("new_york_city") else "los_angeles"
                valid = mask_batch[batch_index] & (target_batch[batch_index] > 0)
                labels = connected_components(valid, connectivity=1)
                component_count = 0
                for component_id in range(1, int(labels.max()) + 1):
                    component = labels == component_id
                    if int(component.sum()) < minimum_component_pixels:
                        continue
                    values[city]["target"].append(float(np.median(target_batch[batch_index][component])))
                    values[city]["prediction"].append(float(np.median(pred_batch[batch_index][component])))
                    component_count += 1
                pred_values = pred_batch[batch_index][valid]
                chip_rows.append(
                    {
                        "chip_id": chip_id,
                        "source_city": city,
                        "building_components": component_count,
                        "pred_std_m": float(np.std(pred_values)) if pred_values.size else float("nan"),
                        "collapse_flag": bool(np.std(pred_values) < 0.05) if pred_values.size else False,
                    }
                )
    grouped: dict[str, dict[str, float]] = {}
    for city in CITIES:
        x = np.asarray(values[city]["target"], dtype=float)
        y = np.asarray(values[city]["prediction"], dtype=float)
        if x.size == 0:
            raise RuntimeError(f"No validation building components for {city}")
        grouped[city] = regression_metrics(x, y)
    all_x = np.concatenate([np.asarray(values[city]["target"]) for city in CITIES])
    all_y = np.concatenate([np.asarray(values[city]["prediction"]) for city in CITIES])
    grouped["all"] = regression_metrics(all_x, all_y)
    result = {
        "city_balanced_rmse_m": float(
            np.mean([grouped[city]["rmse_m"] for city in CITIES])
        ),
        **{f"all_{key}": value for key, value in grouped["all"].items()},
        **{
            f"{city}_{key}": value
            for city in CITIES
            for key, value in grouped[city].items()
        },
        "collapsed_chips": int(sum(row["collapse_flag"] for row in chip_rows)),
        "validation_chips": len(chip_rows),
    }
    return result, chip_rows


def run_spatial_training(
    *,
    args: Any,
    cfg: dict[str, Any],
    dataset_dir: Path,
    run_dir: Path,
    get_model_and_optimizer: Callable,
    export_predictions: Callable,
    summarize_collapse_check: Callable,
    write_csv: Callable,
    write_training_loss_summaries: Callable,
) -> dict[str, Any]:
    manifest = read_manifest(dataset_dir)
    train_rows = [manifest[chip_id] for chip_id in read_ids(dataset_dir / "train.txt")]
    val_rows = [manifest[chip_id] for chip_id in read_ids(dataset_dir / "val.txt")]
    augment_spatial = args.augmentation_profile in {"spatial", "spatial_spectral"}
    augment_spectral = args.augmentation_profile == "spatial_spectral"
    train_dataset = SpatialHTCDataset(
        dataset_dir,
        train_rows,
        augment_spatial=augment_spatial,
        augment_spectral=augment_spectral,
    )
    val_dataset = SpatialHTCDataset(
        dataset_dir, val_rows, augment_spatial=False, augment_spectral=False
    )
    batch_sampler = CityHeightBatchSampler(train_rows, seed=args.seed, batch_size=args.batch_size)
    train_loader = DataLoader(train_dataset, batch_sampler=batch_sampler, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model, optimizer = get_model_and_optimizer(cfg)
    for group in optimizer.param_groups:
        group["weight_decay"] = args.weight_decay
    model.to(cfg["device"])

    history = []
    validation_history = []
    collapse_checks = []
    best_metric = float("inf")
    best_epoch = None
    checks_without_improvement = 0
    stopped_early = False
    stopped_for_collapse = False

    for epoch_index in range(args.epochs):
        epoch = epoch_index + 1
        batch_sampler.set_epoch(epoch_index)
        model.train()
        epoch_losses = []
        for chip_ids, image, gt in train_loader:
            image = image.to(cfg["device"])
            gt_device = {key: value.to(cfg["device"]) for key, value in gt.items()}
            losses, _ = model(image, gt_device)
            loss = losses["loss_total"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_value = float(loss.detach().cpu())
            epoch_losses.append(loss_value)
            history.append(
                {
                    "epoch": epoch,
                    "chip_ids": ";".join(chip_ids),
                    "loss_total": loss_value,
                }
            )
        print(f"epoch {epoch}: mean_loss={np.mean(epoch_losses):.6f}")

        checkpoint_due = args.save_checkpoints_every and epoch % args.save_checkpoints_every == 0
        validation_due = args.validation_every and epoch % args.validation_every == 0
        if checkpoint_due:
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "cfg": cfg,
                    "history": history,
                    "weight_decay": args.weight_decay,
                },
                run_dir / f"model_epoch_{epoch:03d}.pth",
            )
        if not validation_due:
            continue

        metrics, validation_chip_rows = validation_metrics(model, val_loader, cfg["device"])
        metrics.update(
            {
                "epoch": epoch,
                "mean_train_loss": float(np.mean(epoch_losses)),
                "best_metric_before_check_m": best_metric,
            }
        )
        metric = float(metrics["city_balanced_rmse_m"])
        improved = metric <= best_metric - args.early_stopping_min_delta
        if improved or best_epoch is None:
            best_metric = metric
            best_epoch = epoch
            checks_without_improvement = 0
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "cfg": cfg,
                    "history": history,
                    "validation_metrics": metrics,
                    "weight_decay": args.weight_decay,
                },
                run_dir / "model_best.pth",
            )
        else:
            checks_without_improvement += 1
        metrics["improved"] = improved or best_epoch == epoch
        metrics["best_epoch"] = best_epoch
        metrics["best_city_balanced_rmse_m"] = best_metric
        metrics["checks_without_improvement"] = checks_without_improvement
        validation_history.append(metrics)
        write_csv(run_dir / "validation_epoch_metrics.csv", validation_history)
        write_csv(
            run_dir / f"validation_chip_collapse_epoch_{epoch:03d}.csv",
            validation_chip_rows,
        )

        prediction_rows = export_predictions(
            model=model,
            cfg=cfg,
            selected=val_rows,
            run_dir=run_dir,
            data_dir=dataset_dir,
            output_subdir=f"val_predictions_epoch_{epoch:03d}",
            collapse_std_threshold=args.collapse_std_threshold,
            mask_predictions=False,
            prediction_nodata=args.prediction_nodata,
        )
        write_csv(run_dir / f"val_predictions_summary_epoch_{epoch:03d}.csv", prediction_rows)
        collapse = summarize_collapse_check(
            epoch=epoch,
            prediction_rows=prediction_rows,
            collapse_std_threshold=args.collapse_std_threshold,
            collapse_min_share=args.collapse_min_share,
        )
        collapse_checks.append(collapse)
        write_csv(run_dir / "prediction_collapse_checks.csv", collapse_checks)
        print(
            f"validation epoch={epoch}: city_balanced_rmse={metric:.4f} "
            f"best={best_metric:.4f} best_epoch={best_epoch} "
            f"patience={checks_without_improvement}/{args.early_stopping_patience} "
            f"collapsed={collapse['collapsed_chips']}/{collapse['total_chips']}"
        )
        if args.stop_on_collapse and collapse["stop_triggered"]:
            stopped_for_collapse = True
            break
        if (
            epoch >= args.min_epochs
            and checks_without_improvement >= args.early_stopping_patience
        ):
            stopped_early = True
            print(f"Stopping early at epoch {epoch}; best epoch was {best_epoch}")
            break

    if best_epoch is None or not (run_dir / "model_best.pth").exists():
        raise RuntimeError("Training finished without a valid best checkpoint")
    torch.save(
        {
            "epoch": history[-1]["epoch"],
            "state_dict": model.state_dict(),
            "cfg": cfg,
            "history": history,
            "validation_history": validation_history,
            "weight_decay": args.weight_decay,
            "stopped_early": stopped_early,
            "stopped_for_collapse": stopped_for_collapse,
        },
        run_dir / "model_last.pth",
    )
    write_csv(run_dir / "training_history.csv", history)
    write_training_loss_summaries(run_dir, history)
    write_csv(
        run_dir / "training_chip_exposures.csv",
        [
            {
                "chip_id": row["chip_id"],
                "source_city": row["source_city"],
                "height_category": row["height_category"],
                "exposures": batch_sampler.exposures[row["chip_id"]],
            }
            for row in train_rows
        ],
    )
    return {
        "best_epoch": best_epoch,
        "best_city_balanced_rmse_m": best_metric,
        "epochs_completed": int(history[-1]["epoch"]),
        "stopped_early": stopped_early,
        "stopped_for_collapse": stopped_for_collapse,
    }
