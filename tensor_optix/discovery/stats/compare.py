"""
Comparing two model families on the same folds.

Decision rule (region of practical equivalence, ROPE):

    CI of (a - b) entirely above +rope        -> "better"
    CI entirely below -rope                   -> "worse"
    CI entirely inside [-rope, +rope]         -> "equivalent"
    otherwise                                 -> "undetermined"

"undetermined" is a first-class outcome: with the data and budget available
the two families cannot be told apart in a way that matters, and the report
says so instead of crowning a lucky winner.
"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .intervals import Interval, corrected_resampled_t, t_interval


def rope_verdict(lo: float, hi: float, rope: float) -> str:
    if rope < 0:
        raise ValueError("rope must be >= 0")
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return "undetermined"
    if lo > rope:
        return "better"
    if hi < -rope:
        return "worse"
    if lo >= -rope and hi <= rope:
        return "equivalent"
    return "undetermined"


def paired_deltas(scores_a: Sequence[Optional[float]], scores_b: Sequence[Optional[float]],
                  higher_is_better: bool = True) -> np.ndarray:
    """
    Per-fold deltas oriented so positive = ``a`` better. Folds where either
    score is missing are dropped *pairwise* (never filled in).
    """
    if len(scores_a) != len(scores_b):
        raise ValueError("paired comparison needs scores on the same folds")
    out = []
    for x, y in zip(scores_a, scores_b):
        if x is None or y is None:
            continue
        x, y = float(x), float(y)
        if not (math.isfinite(x) and math.isfinite(y)):
            continue
        out.append(x - y if higher_is_better else y - x)
    return np.asarray(out, dtype=np.float64)


def compare_paired(scores_a: Sequence[Optional[float]], scores_b: Sequence[Optional[float]], *,
                   rope: float, higher_is_better: bool = True,
                   n_train: Optional[float] = None, n_test: Optional[float] = None,
                   level: float = 0.95) -> Tuple[str, Interval]:
    """
    Paired comparison of two families' per-fold scores. With fold sizes given,
    uses the Nadeau-Bengio corrected interval (resampled CV); without them, a
    plain paired t interval (for genuinely independent replicates, e.g. RL seeds).
    """
    d = paired_deltas(scores_a, scores_b, higher_is_better)
    if len(d) == 0:
        raise ValueError("no complete pairs to compare")
    if n_train is not None and n_test is not None:
        iv = corrected_resampled_t(d, n_train, n_test, level)
    else:
        iv = t_interval(d, level)
    return rope_verdict(iv.lo, iv.hi, rope), iv


def holm(p_values: Sequence[Optional[float]]) -> List[Optional[float]]:
    """Holm-Bonferroni adjusted p-values (step-down, monotone). None stays None."""
    idx = [i for i, p in enumerate(p_values) if p is not None]
    m = len(idx)
    adjusted: List[Optional[float]] = [None] * len(p_values)
    order = sorted(idx, key=lambda i: p_values[i])
    running = 0.0
    for rank, i in enumerate(order):
        val = min(1.0, (m - rank) * float(p_values[i]))
        running = max(running, val)
        adjusted[i] = running
    return adjusted


def default_rope(metric: str, *, target_std: Optional[float] = None,
                 scale: Optional[float] = None) -> float:
    """
    Smallest difference that matters in practice, per metric. These are
    defaults the report states explicitly; users can override them.
    """
    m = metric.lower()
    if m in ("auc", "roc_auc", "macro_auc"):
        return 0.005
    if m in ("accuracy", "balanced_accuracy", "macro_f1", "f1"):
        return 0.01
    if m in ("r2",):
        return 0.01
    if m in ("rmse", "mae") and target_std is not None and target_std > 0:
        return 0.01 * target_std
    if m in ("log_loss", "brier", "poisson_deviance") and scale is not None and scale > 0:
        return 0.01 * scale                      # 1% of the baseline (dummy) loss
    if m == "return" and scale is not None and scale > 0:
        return 0.05 * scale                      # RL: 5% of random-to-best return gap
    return 0.0
