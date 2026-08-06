"""Overlap-mean and variance helpers for HTC-DC city-scale inference."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def window_starts(length: int, window: int, stride: int) -> list[int]:
    if length <= window:
        return [0]
    starts = list(range(0, length - window + 1, stride))
    if starts[-1] != length - window:
        starts.append(length - window)
    return starts


def overlap_predict(
    image: np.ndarray,
    predictor: Callable[[np.ndarray], np.ndarray],
    window: int = 256,
    stride: int = 128,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predict a channels-first image with overlapping windows."""
    if image.ndim != 3:
        raise ValueError(f"Expected channels-first image, got {image.shape}")
    _, height, width = image.shape
    if height < window or width < window:
        raise ValueError(f"Image {height}x{width} is smaller than window {window}")
    sum_prediction = np.zeros((height, width), dtype="float64")
    sum_squared = np.zeros((height, width), dtype="float64")
    count = np.zeros((height, width), dtype="uint16")
    for row in window_starts(height, window, stride):
        for col in window_starts(width, window, stride):
            prediction = np.asarray(predictor(image[:, row : row + window, col : col + window]))
            if prediction.shape != (window, window):
                raise RuntimeError(f"Predictor returned {prediction.shape}, expected {(window, window)}")
            sum_prediction[row : row + window, col : col + window] += prediction
            sum_squared[row : row + window, col : col + window] += prediction**2
            count[row : row + window, col : col + window] += 1
    if np.any(count == 0):
        raise RuntimeError("Sliding-window inference left uncovered pixels")
    mean = sum_prediction / count
    variance = np.maximum(sum_squared / count - mean**2, 0)
    return mean.astype("float32"), variance.astype("float32"), count

