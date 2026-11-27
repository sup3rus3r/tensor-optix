"""
Cross-validated evaluation of one family / configuration.

For every fold: fit the family's ``ColumnPipeline`` on the training rows only,
fit the estimator, predict the held-out rows, score. Per-fold scores are kept
(paired by fold id across families) together with a Nadeau-Bengio corrected
interval of their mean. A fold that fails records ``None`` and the error; the
experiment is ``failed`` only if every fold failed.

With a ``budget`` (``BudgetLedger``), folds stop being run once it is
exhausted: the remaining folds record ``None`` (reason ``budget``), and an
experiment cut before any fold finished is ``skipped_budget``.
"""
from __future__ import annotations

import math
import time
from dataclasses import replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..rng import int_seed_for
from ..schema import ExperimentResult
from ..stats.intervals import corrected_resampled_t
from ..stats.metrics import get_metric, score
from .data import Table
from .families.base import CLASSIFICATION, Family
from .preprocess import ColumnPipeline
from .types import ColumnType

FoldSpec = Tuple[str, np.ndarray, np.ndarray]


def predict_and_score(est, X: np.ndarray, y: np.ndarray, task: str, n_classes: int,
                      metrics: Sequence[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if task in CLASSIFICATION:
        proba = est.predict_proba(X)
        for m in metrics:
            out[m] = score(m, y, proba=proba, n_classes=n_classes)
    else:
        pred = est.predict(X)
        if task == "count":
            pred = np.clip(pred, 0.0, None)
        for m in metrics:
            out[m] = score(m, y, values=pred, n_classes=n_classes)
    return out


ROW_LOSS_METRICS = ("rmse", "mae", "log_loss", "brier", "poisson_deviance")


def per_row_losses(metric: str, est, X: np.ndarray, y: np.ndarray, task: str, n_classes: int) -> np.ndarray:
    """Per-row loss whose mean is the metric (RMSE -> squared error: its mean is MSE)."""
    eps = 1e-15
    if task in CLASSIFICATION:
        P = np.asarray(est.predict_proba(X), dtype=np.float64)
        yi = np.asarray(y).astype(int)
        if metric == "log_loss":
            return -np.log(np.clip(P[np.arange(len(yi)), yi], eps, 1.0))
        if metric == "brier":
            onehot = np.zeros_like(P)
            onehot[np.arange(len(yi)), yi] = 1.0
            return np.sum((P - onehot) ** 2, axis=1)
        raise ValueError(f"{metric!r} has no per-row loss for {task}")
    pred = np.asarray(est.predict(X), dtype=np.float64)
    if task == "count":
        pred = np.clip(pred, 0.0, None)
    yf = np.asarray(y, dtype=np.float64)
    if metric == "rmse":
        return (pred - yf) ** 2
    if metric == "mae":
        return np.abs(pred - yf)
    if metric == "poisson_deviance":
        mu = np.clip(pred, 1e-9, None)
        with np.errstate(divide="ignore", invalid="ignore"):
            term = np.where(yf > 0, yf * np.log(yf / mu), 0.0)
        return 2.0 * (term - (yf - mu))
    raise ValueError(f"{metric!r} has no per-row loss for {task}")


def _finite_or_none(v: float) -> Optional[float]:
    return float(v) if v is not None and math.isfinite(v) else None


def evaluate_family(table: Table, types: Dict[str, ColumnType], y: np.ndarray, folds: Sequence[FoldSpec],
                    family: Family, config: Dict, *, task: str, n_classes: int, metric: str,
                    secondary: Sequence[str] = (), seed: int = 0, stage: str = "trial",
                    drop: Sequence[str] = (), budget=None,
                    row_losses: Optional[np.ndarray] = None) -> ExperimentResult:
    """
    ``y`` is the encoded target for *all* rows of ``table`` (indexed by the folds).
    ``row_losses``: optional float array of length ``len(y)``; each fold's test
    rows receive their per-row loss for ``metric`` (see ``per_row_losses``).
    """
    m = get_metric(metric)
    metrics = [metric] + [s for s in secondary if s != metric]
    opts = family.preprocessing
    if drop:
        opts = replace(opts, drop=tuple(sorted(set(opts.drop) | set(drop))))
    fold_ids, scores = [], []
    sec: Dict[str, List[Optional[float]]] = {s: [] for s in metrics[1:]}
    stops, errors = [], []
    t0 = time.perf_counter()
    n_train, n_test = [], []
    cut = False
    for fid, tr, te in folds:
        fold_ids.append(fid)
        if cut or (budget is not None and budget.exhausted()):
            cut = True
            scores.append(None)
            for s in metrics[1:]:
                sec[s].append(None)
            stops.append("budget")
            continue
        try:
            pipe = ColumnPipeline(types, opts, task=task, n_classes=n_classes,
                                  seed=int_seed_for(seed, "pipeline", family.name, fid))
            Xtr = pipe.fit_transform(table, tr, y[tr])
            Xte = pipe.transform(table, te)
            est = family.make(config, int_seed_for(seed, "estimator", family.name, fid), task, n_classes)
            est.fit(Xtr, y[tr])
            vals = predict_and_score(est, Xte, y[te], task, n_classes, metrics)
            if row_losses is not None:
                row_losses[te] = per_row_losses(metric, est, Xte, y[te], task, n_classes)
            scores.append(_finite_or_none(vals[metric]))
            for s in metrics[1:]:
                sec[s].append(_finite_or_none(vals[s]))
            stops.append(getattr(est, "stop_reason", "closed_form"))
            n_train.append(len(tr))
            n_test.append(len(te))
        except Exception as exc:                          # one bad fold must not hide the others
            scores.append(None)
            for s in metrics[1:]:
                sec[s].append(None)
            errors.append(f"{fid}: {type(exc).__name__}: {exc}")
        if budget is not None:
            budget.record_fit(len(tr))
    elapsed = time.perf_counter() - t0
    ok = [s for s in scores if s is not None]
    status = "ok" if ok else ("skipped_budget" if cut and not errors else "failed")
    mean, interval = None, None
    if ok:
        mean = float(np.mean(ok))
        if len(ok) >= 2:
            iv = corrected_resampled_t(ok, float(np.mean(n_train)), float(np.mean(n_test)))
            interval = (iv.lo, iv.hi)
    return ExperimentResult(
        family=family.name, config=dict(config), stage=stage, metric=metric,
        higher_is_better=m.higher_is_better, fold_ids=tuple(fold_ids), scores=tuple(scores),
        secondary={k: tuple(v) for k, v in sec.items()}, interval=interval, mean=mean,
        fit_time_s=round(elapsed, 4), stop_reasons=tuple(stops), status=status,
        error="; ".join(errors[:3]) if errors else None,
    )
