"""
tensor_optix.ml_core.metrics_impl.regression — pure-numpy regression
metrics: MAE, RMSE, R-squared.

Framework-agnostic (plain numpy arrays), same reasoning as
metrics_impl/classification.py.
"""
from __future__ import annotations

from typing import Dict

import numpy as np


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    R^2 = 1 - SS_res / SS_tot.

    SS_tot == 0 (every true value identical) makes R^2 undefined in the
    usual formula -- returns 1.0 when predictions also match exactly
    (perfect, trivially-correct fit) and 0.0 otherwise, rather than NaN/inf,
    so this always produces a finite, checkpoint-comparable value.
    """
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else 0.0
    return 1.0 - ss_res / ss_tot


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Full regression metric set as a flat Dict[str, float], suitable for
    MLMetrics.metrics."""
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": root_mean_squared_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }
