"""Training utilities for the full four-channel HTC-DC recipe."""

from __future__ import annotations

import csv
import math
import random
from pathlib import Path
from typing import Any, Callable

import numpy as np
import rasterio
from skimage.measure import label as connected_components
import torch
from torch.utils.data import DataLoader, Dataset


CITIES = ("los_angeles", "new_york_city")


def read_ids(path: Path) -> list[str]:
    values = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not values:
        raise RuntimeError(f"Empty split file: {path}")
    return values


def read_manifest(dataset_dir: Path) -> dict[str, dict[str, str]]:
    with (dataset_dir / "chips_manifest.csv").open("r", newline="", encoding="utf-8") as stream:
        rows = {row["chip_id"]: row for row in csv.DictReader(stream)}
    if not rows:
        raise RuntimeError(f"Empty manifest: {dataset_dir / 'chips_manifest.csv'}")
    return rows


def load_stats(dataset_dir: Path, expected_channels: int = 4) -> tuple[np.ndarray, np.ndarray]:
    mean, std = torch.load(dataset_dir / "image_stats.pickle")
    mean_array = np.asarray(mean, dtype="float32")
    std_array = np.asarray(std, dtype="float32")
    if mean_array.size != expected_channels or std_array.size != expected_channels:
        raise RuntimeError(
            f"Expected {expected_channels} normalization values, got "
            f"{mean_array.size}/{std_array.size}"
        )
    if not np.all(np.isfinite(mean_array)) or not np.all(np.isfinite(std_array)):
        raise RuntimeError("Normalization statistics contain non-finite values")
    if np.any(std_array <= 0):
        raise RuntimeError(f"Invalid normalization standard deviations: {std_array}")
    return mean_array, std_array


class FullRecipeDataset(Dataset):
    """Load RGB+NIR while retaining the footprint as separate ground truth."""

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

    @staticmethod
    def _spatial_transform(
        image: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
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
        if image_np.shape != (4, 256, 256):
            raise RuntimeError(f"Expected [4,256,256] image for {chip_id}, got {image_np.shape}")
        with rasterio.open(self.dataset_dir / "ndsm" / f"{chip_id}_AGL.tif") as src:
            target_np = np.nan_to_num(src.read(1).astype("float32"), nan=0.0).clip(0)
        with rasterio.open(self.dataset_dir / "mask" / f"{chip_id}_BLG.tif") as src:
            mask_np = (src.read(1) > 0).astype("float32")

        image = torch.from_numpy(image_np)
        target = torch.from_numpy(target_np[None])
        mask = torch.from_numpy(mask_np[None])
        if self.augment_spectral:
            common_gain = random.uniform(0.90, 1.10)
            gains = torch.tensor(
                [random.uniform(0.97, 1.03) for _ in range(4)], dtype=image.dtype
            )[:, None, None]
            image = torch.clamp(image * common_gain * gains, min=0)
        if self.augment_spatial:
            image, target, mask = self._spatial_transform(image, target, mask)
        if not torch.all((mask == 0) | (mask == 1)):
            raise RuntimeError(f"Separate footprint mask became non-binary for {chip_id}")

        mean = torch.from_numpy(self.mean)[:, None, None]
        std = torch.from_numpy(self.std)[:, None, None]
        image = (image - mean) / std
        return chip_id, image, {"ndsm": target, "mask": mask}


def regression_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    residual = prediction - target
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((target - target.mean()) ** 2))
    slope, intercept = np.polyfit(target, prediction, 1) if target.size >= 2 else (np.nan, np.nan)
    return {
        "count": int(target.size),
        "rmse_m": float(np.sqrt(np.mean(residual**2))),
        "mae_m": float(np.mean(np.abs(residual))),
        "bias_m": float(np.mean(residual)),
        "r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "slope": float(slope),
        "intercept_m": float(intercept),
    }


def validation_metrics(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
    minimum_component_pixels: int = 3,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    model.eval()
    model.training = True
    buildings = {city: {"target": [], "prediction": []} for city in CITIES}
    pixel_sums = {
        city: {
            group: {"count": 0, "abs": 0.0, "sq": 0.0, "signed": 0.0}
            for group in ("all", "positive", "background", "footprint")
        }
        for city in CITIES
    }
    chip_rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for chip_ids, image, gt in loader:
            image_device = image.to(device)
            gt_device = {key: value.to(device) for key, value in gt.items()}
            _, prediction = model(image_device, gt_device)
            pred_batch = prediction["ndsm"][:, 0].detach().cpu().numpy()
            target_batch = gt["ndsm"][:, 0].numpy()
            mask_batch = gt["mask"][:, 0].numpy() > 0
            for index, chip_id in enumerate(chip_ids):
                city = "new_york_city" if chip_id.startswith("new_york_city") else "los_angeles"
                pred = pred_batch[index]
                target = target_batch[index]
                footprint = mask_batch[index]
                groups = {
                    "all": np.isfinite(target),
                    "positive": target > 0,
                    "background": target <= 0,
                    "footprint": footprint,
                }
                for name, valid in groups.items():
                    residual = pred[valid] - target[valid]
                    stats = pixel_sums[city][name]
                    stats["count"] += int(residual.size)
                    stats["abs"] += float(np.abs(residual).sum())
                    stats["sq"] += float((residual**2).sum())
                    stats["signed"] += float(residual.sum())

                component_valid = footprint & (target > 0)
                labels = connected_components(component_valid, connectivity=1)
                component_count = 0
                for component_id in range(1, int(labels.max()) + 1):
                    component = labels == component_id
                    if int(component.sum()) < minimum_component_pixels:
                        continue
                    buildings[city]["target"].append(float(np.median(target[component])))
                    buildings[city]["prediction"].append(float(np.median(pred[component])))
                    component_count += 1
                pred_values = pred[component_valid]
                chip_rows.append(
                    {
                        "chip_id": chip_id,
                        "source_city": city,
                        "building_components": component_count,
                        "pred_std_m": float(np.std(pred_values)) if pred_values.size else float("nan"),
                        "collapse_flag": bool(np.std(pred_values) < 0.05) if pred_values.size else False,
                    }
                )

    metrics: dict[str, float] = {}
    city_building_metrics: dict[str, dict[str, float]] = {}
    for city in CITIES:
        target = np.asarray(buildings[city]["target"], dtype=float)
        prediction = np.asarray(buildings[city]["prediction"], dtype=float)
        if target.size == 0:
            raise RuntimeError(f"No validation buildings found for {city}")
        city_building_metrics[city] = regression_metrics(target, prediction)
        for key, value in city_building_metrics[city].items():
            metrics[f"{city}_building_{key}"] = value
        for group, sums in pixel_sums[city].items():
            count = sums["count"]
            if count == 0:
                continue
            metrics[f"{city}_{group}_pixel_rmse_m"] = math.sqrt(sums["sq"] / count)
            metrics[f"{city}_{group}_pixel_mae_m"] = sums["abs"] / count
            metrics[f"{city}_{group}_pixel_bias_m"] = sums["signed"] / count

    all_target = np.concatenate([np.asarray(buildings[city]["target"]) for city in CITIES])
    all_prediction = np.concatenate([np.asarray(buildings[city]["prediction"]) for city in CITIES])
    for key, value in regression_metrics(all_target, all_prediction).items():
        metrics[f"all_building_{key}"] = value
    metrics["city_balanced_building_rmse_m"] = float(
        np.mean([city_building_metrics[city]["rmse_m"] for city in CITIES])
    )
    metrics["collapsed_chips"] = int(sum(row["collapse_flag"] for row in chip_rows))
    metrics["validation_chips"] = len(chip_rows)
    return metrics, chip_rows


def run_training(
    *,
    cfg: dict[str, Any],
    dataset_dir: Path,
    run_dir: Path,
    epochs: int,
    seed: int,
    device: str,
    num_workers: int,
    validation_every: int,
    checkpoint_every: int,
    get_model_and_optimizer: Callable,
    write_csv: Callable,
) -> dict[str, Any]:
    manifest = read_manifest(dataset_dir)
    train_rows = [manifest[chip_id] for chip_id in read_ids(dataset_dir / "train.txt")]
    val_rows = [manifest[chip_id] for chip_id in read_ids(dataset_dir / "val.txt")]
    train_dataset = FullRecipeDataset(dataset_dir, train_rows, True, True)
    val_dataset = FullRecipeDataset(dataset_dir, val_rows, False, False)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        generator=generator,
        num_workers=num_workers,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
        num_workers=num_workers,
    )
    model, optimizer = get_model_and_optimizer(cfg)
    model.to(device)

    history: list[dict[str, Any]] = []
    validation_history: list[dict[str, Any]] = []
    best_metric = float("inf")
    best_epoch: int | None = None
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_rows: list[dict[str, Any]] = []
        for chip_ids, image, gt in train_loader:
            image = image.to(device)
            gt_device = {key: value.to(device) for key, value in gt.items()}
            losses, prediction = model(image, gt_device)
            loss_total = losses["loss_total"]
            if not torch.isfinite(loss_total):
                raise RuntimeError(f"Non-finite total loss at epoch {epoch}: {loss_total}")
            optimizer.zero_grad(set_to_none=True)
            loss_total.backward()
            if any(
                parameter.grad is not None and not torch.all(torch.isfinite(parameter.grad))
                for parameter in model.parameters()
            ):
                raise RuntimeError(f"Non-finite gradient at epoch {epoch}")
            optimizer.step()
            row = {"epoch": epoch, "chip_ids": ";".join(chip_ids)}
            for key, value in losses.items():
                if torch.is_tensor(value) and value.numel() == 1:
                    row[key] = float(value.detach().cpu())
            row["supervised_levels"] = len(prediction["ndsm_intermediate"])
            epoch_rows.append(row)
        history.extend(epoch_rows)
        mean_loss = float(np.mean([row["loss_total"] for row in epoch_rows]))
        print(f"epoch {epoch}: mean_loss={mean_loss:.6f}", flush=True)

        if checkpoint_every and epoch % checkpoint_every == 0:
            torch.save(
                {"epoch": epoch, "state_dict": model.state_dict(), "cfg": cfg},
                run_dir / f"model_epoch_{epoch:03d}.pth",
            )
        if epoch % validation_every != 0 and epoch != epochs:
            continue
        metrics, chip_rows = validation_metrics(model, val_loader, device)
        metric = float(metrics["city_balanced_building_rmse_m"])
        metrics.update({"epoch": epoch, "mean_train_loss": mean_loss})
        improved = metric < best_metric
        if improved:
            best_metric = metric
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "cfg": cfg,
                    "validation_metrics": metrics,
                },
                run_dir / "model_best.pth",
            )
        metrics.update(
            {
                "improved": improved,
                "best_epoch": best_epoch,
                "best_city_balanced_building_rmse_m": best_metric,
            }
        )
        validation_history.append(metrics)
        write_csv(run_dir / "validation_epoch_metrics.csv", validation_history)
        write_csv(run_dir / f"validation_chip_collapse_epoch_{epoch:03d}.csv", chip_rows)
        print(
            f"validation epoch={epoch}: city_balanced_building_rmse={metric:.4f} "
            f"best={best_metric:.4f} best_epoch={best_epoch} "
            f"collapsed={metrics['collapsed_chips']}/{metrics['validation_chips']}",
            flush=True,
        )

    if best_epoch is None:
        raise RuntimeError("Training completed without a best checkpoint")
    torch.save(
        {"epoch": epochs, "state_dict": model.state_dict(), "cfg": cfg},
        run_dir / "model_last.pth",
    )
    write_csv(run_dir / "training_history.csv", history)
    return {
        "best_epoch": best_epoch,
        "best_city_balanced_building_rmse_m": best_metric,
        "epochs_completed": epochs,
    }

