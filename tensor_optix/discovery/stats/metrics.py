"""
Scoring metrics for discovery, computed from predictions (labels, class
probabilities or values) -- not through the ML adapters' ``evaluate()``,
which only sees hard labels.

Conventions:
  * classification targets are integer-coded 0..K-1 (discovery encodes labels
    before scoring);
  * binary probability inputs are P(class 1), shape (n,) or (n, 2);
  * multiclass probability inputs have shape (n, K) and rows summing to 1
    (they are renormalised defensively).

``METRICS`` maps names to ``Metric`` records with the direction of
improvement, so comparisons never mix up "higher is better" and "lower is
better".
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import numpy as np

_EPS = 1e-15


def rankdata(x: np.ndarray) -> np.ndarray:
    """Average ranks (1-based), ties share the mean rank (like scipy 'average')."""
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    ranks = np.empty(len(x), dtype=np.float64)
    i = 0
    n = len(x)
    while i < n:
        j = i
        while j + 1 < n and xs[j + 1] == xs[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def _as_binary_proba(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    if p.ndim == 2:
        if p.shape[1] != 2:
            raise ValueError("binary probabilities must have shape (n,) or (n, 2)")
        p = p[:, 1]
    return p


def _as_proba_matrix(p: np.ndarray, k: int) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    if p.ndim == 1:
        if k != 2:
            raise ValueError("1-D probabilities are only valid for binary targets")
        p = np.column_stack([1.0 - p, p])
    if p.shape[1] != k:
        raise ValueError(f"expected {k} probability columns, got {p.shape[1]}")
    p = np.clip(p, 0.0, None)
    s = p.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    return p / s


def roc_auc_binary(y: np.ndarray, p: np.ndarray) -> float:
    """Mann-Whitney AUC with tie correction. NaN if only one class is present."""
    y = np.asarray(y).astype(int)
    p = _as_binary_proba(p)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = rankdata(p)
    return float((r[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def roc_auc_macro(y: np.ndarray, p: np.ndarray, k: int) -> float:
    """Macro one-vs-rest AUC over classes present in ``y``."""
    y = np.asarray(y).astype(int)
    pm = _as_proba_matrix(p, k)
    vals = []
    for c in range(k):
        yc = (y == c).astype(int)
        if 0 < yc.sum() < len(yc):
            vals.append(roc_auc_binary(yc, pm[:, c]))
    return float(np.mean(vals)) if vals else float("nan")


def log_loss(y: np.ndarray, p: np.ndarray, k: int) -> float:
    y = np.asarray(y).astype(int)
    pm = _as_proba_matrix(p, k)
    picked = pm[np.arange(len(y)), y]
    return float(-np.mean(np.log(np.clip(picked, _EPS, 1.0))))


def brier(y: np.ndarray, p: np.ndarray, k: int) -> float:
    """Multi-class Brier score: mean squared error of the probability vector."""
    y = np.asarray(y).astype(int)
    pm = _as_proba_matrix(p, k)
    onehot = np.zeros_like(pm)
    onehot[np.arange(len(y)), y] = 1.0
    return float(np.mean(np.sum((pm - onehot) ** 2, axis=1)))


def accuracy(y: np.ndarray, yhat: np.ndarray) -> float:
    y, yhat = np.asarray(y).astype(int), np.asarray(yhat).astype(int)
    return float(np.mean(y == yhat)) if len(y) else float("nan")


def balanced_accuracy(y: np.ndarray, yhat: np.ndarray) -> float:
    """Mean per-class recall over classes present in ``y``."""
    y, yhat = np.asarray(y).astype(int), np.asarray(yhat).astype(int)
    recalls = [float(np.mean(yhat[y == c] == c)) for c in np.unique(y)]
    return float(np.mean(recalls)) if recalls else float("nan")


def macro_f1(y: np.ndarray, yhat: np.ndarray) -> float:
    """Macro F1 over classes present in ``y`` or ``yhat``."""
    y, yhat = np.asarray(y).astype(int), np.asarray(yhat).astype(int)
    f1s = []
    for c in np.union1d(np.unique(y), np.unique(yhat)):
        tp = float(np.sum((y == c) & (yhat == c)))
        fp = float(np.sum((y != c) & (yhat == c)))
        fn = float(np.sum((y == c) & (yhat != c)))
        denom = 2 * tp + fp + fn
        f1s.append(2 * tp / denom if denom > 0 else 0.0)
    return float(np.mean(f1s)) if f1s else float("nan")


def rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    y, yhat = np.asarray(y, dtype=np.float64), np.asarray(yhat, dtype=np.float64)
    return float(math.sqrt(np.mean((y - yhat) ** 2)))


def mae(y: np.ndarray, yhat: np.ndarray) -> float:
    y, yhat = np.asarray(y, dtype=np.float64), np.asarray(yhat, dtype=np.float64)
    return float(np.mean(np.abs(y - yhat)))


def r2(y: np.ndarray, yhat: np.ndarray) -> float:
    y, yhat = np.asarray(y, dtype=np.float64), np.asarray(yhat, dtype=np.float64)
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot == 0.0:
        return float("nan")
    return float(1.0 - np.sum((y - yhat) ** 2) / ss_tot)


def poisson_deviance(y: np.ndarray, yhat: np.ndarray) -> float:
    """Mean Poisson deviance; predictions are clipped to a small positive floor."""
    y = np.asarray(y, dtype=np.float64)
    mu = np.clip(np.asarray(yhat, dtype=np.float64), 1e-9, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(y > 0, y * np.log(y / mu), 0.0)
    return float(np.mean(2.0 * (term - (y - mu))))


@dataclass(frozen=True)
class Metric:
    name: str
    tasks: Tuple[str, ...]
    higher_is_better: bool
    needs: str                    # "proba" | "label" | "value"
    fn: Callable


def _proba_fn(f):
    return lambda y, pred, k: f(y, pred, k)


METRICS: Dict[str, Metric] = {
    "auc": Metric("auc", ("binary",), True, "proba", lambda y, p, k: roc_auc_binary(y, p)),
    "macro_auc": Metric("macro_auc", ("multiclass",), True, "proba", lambda y, p, k: roc_auc_macro(y, p, k)),
    "log_loss": Metric("log_loss", ("binary", "multiclass"), False, "proba", _proba_fn(log_loss)),
    "brier": Metric("brier", ("binary", "multiclass"), False, "proba", _proba_fn(brier)),
    "accuracy": Metric("accuracy", ("binary", "multiclass"), True, "label", lambda y, l, k: accuracy(y, l)),
    "balanced_accuracy": Metric("balanced_accuracy", ("binary", "multiclass"), True, "label",
                                lambda y, l, k: balanced_accuracy(y, l)),
    "macro_f1": Metric("macro_f1", ("binary", "multiclass"), True, "label", lambda y, l, k: macro_f1(y, l)),
    "rmse": Metric("rmse", ("regression", "count"), False, "value", lambda y, v, k: rmse(y, v)),
    "mae": Metric("mae", ("regression", "count"), False, "value", lambda y, v, k: mae(y, v)),
    "r2": Metric("r2", ("regression", "count"), True, "value", lambda y, v, k: r2(y, v)),
    "poisson_deviance": Metric("poisson_deviance", ("count",), False, "value",
                               lambda y, v, k: poisson_deviance(y, v)),
}


def get_metric(name: str) -> Metric:
    try:
        return METRICS[name]
    except KeyError:
        raise ValueError(f"unknown metric {name!r}; known: {sorted(METRICS)}") from None


def score(metric: str, y: np.ndarray, *, proba: np.ndarray = None, labels: np.ndarray = None,
          values: np.ndarray = None, n_classes: int = 2) -> float:
    """Score one prediction set with the metric's required input kind."""
    m = get_metric(metric)
    if m.needs == "proba":
        if proba is None:
            raise ValueError(f"metric {metric!r} needs class probabilities")
        return float(m.fn(y, proba, n_classes))
    if m.needs == "label":
        if labels is None:
            if proba is None:
                raise ValueError(f"metric {metric!r} needs labels or probabilities")
            labels = np.argmax(_as_proba_matrix(proba, n_classes), axis=1)
        return float(m.fn(y, labels, n_classes))
    if values is None:
        raise ValueError(f"metric {metric!r} needs predicted values")
    return float(m.fn(y, values, n_classes))
