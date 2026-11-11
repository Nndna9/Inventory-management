"""Forecast accuracy metrics."""
from __future__ import annotations

import numpy as np


def evaluate(actual, predicted) -> dict:
    """MAE, RMSE, MAPE (rows with actual > 0 only), WAPE and bias (mean of predicted - actual)."""
    y = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    if y.shape != p.shape or y.size == 0:
        raise ValueError("actual and predicted must be non-empty and the same length")
    err = p - y
    nz = y > 0
    return {
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "mape": float((np.abs(err[nz]) / y[nz]).mean()) if nz.any() else float("nan"),
        "wape": float(np.abs(err).sum() / y.sum()) if y.sum() > 0 else float("nan"),
        "bias": float(err.mean()),
    }
