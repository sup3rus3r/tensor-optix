"""
Confidence intervals used to compare model families honestly.

Resampled cross-validation scores are *not* independent: every pair of folds
shares most of its training rows, so the naive t interval over fold scores is
too narrow and over-declares differences. ``corrected_resampled_t`` applies the
Nadeau & Bengio (2003) correction -- variance multiplied by
``1/J + n_test/n_train`` instead of ``1/J`` -- which is the standard remedy and
keeps the type-I error near nominal for repeated k-fold.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np

from .distributions import t_ppf, t_sf


@dataclass(frozen=True)
class Interval:
    mean: float
    lo: float
    hi: float
    se: float
    n: int
    df: Optional[float]
    method: str
    t_stat: Optional[float] = None
    p_two_sided: Optional[float] = None     # H0: true mean == 0


def _var(a: np.ndarray) -> float:
    """Sample variance, with round-off on (numerically) constant input snapped to 0."""
    v = float(a.var(ddof=1))
    scale = max(1.0, float(np.max(np.abs(a))))
    return 0.0 if v <= (1e-12 * scale) ** 2 else v


def _clean(values: Sequence[Optional[float]]) -> np.ndarray:
    a = np.asarray([v for v in values if v is not None], dtype=np.float64)
    return a[np.isfinite(a)]


def t_interval(values: Sequence[Optional[float]], level: float = 0.95) -> Interval:
    """Plain Student-t interval for the mean of (treated-as-independent) values."""
    a = _clean(values)
    n = len(a)
    if n == 0:
        raise ValueError("t_interval needs at least one finite value")
    m = float(a.mean())
    if n == 1:
        return Interval(m, -math.inf, math.inf, math.inf, 1, None, "t")
    s = math.sqrt(_var(a))
    se = s / math.sqrt(n)
    df = n - 1
    q = t_ppf(0.5 + level / 2.0, df)
    if se == 0.0:
        t_stat = math.inf if m > 0 else (-math.inf if m < 0 else 0.0)
    else:
        t_stat = m / se
    p = _two_sided_p(t_stat, df)
    return Interval(m, m - q * se, m + q * se, se, n, df, "t", t_stat, p)


def corrected_resampled_t(deltas: Sequence[Optional[float]], n_train: float, n_test: float,
                          level: float = 0.95) -> Interval:
    """
    Nadeau-Bengio corrected resampled t interval for the mean of ``deltas``
    (per-fold paired score differences from repeated / k-fold CV).

    ``n_train`` / ``n_test`` are the (average) training and test fold sizes.
    """
    a = _clean(deltas)
    J = len(a)
    if J == 0:
        raise ValueError("corrected_resampled_t needs at least one finite value")
    if n_train <= 0 or n_test <= 0:
        raise ValueError("n_train and n_test must be positive")
    m = float(a.mean())
    if J == 1:
        return Interval(m, -math.inf, math.inf, math.inf, 1, None, "corrected_t")
    s2 = _var(a)
    se = math.sqrt((1.0 / J + n_test / n_train) * s2)
    df = J - 1
    q = t_ppf(0.5 + level / 2.0, df)
    if se == 0.0:
        t_stat = math.inf if m > 0 else (-math.inf if m < 0 else 0.0)
    else:
        t_stat = m / se
    p = _two_sided_p(t_stat, df)
    return Interval(m, m - q * se, m + q * se, se, J, df, "corrected_t", t_stat, p)


def _two_sided_p(t_stat: float, df: float) -> float:
    if math.isinf(t_stat):
        return 0.0
    if t_stat == 0.0:
        return 1.0
    return float(min(1.0, 2.0 * t_sf(abs(t_stat), df)))


def bootstrap_interval(values: Sequence[float], rng: np.random.Generator, *,
                       stat: Callable[[np.ndarray], float] = np.mean, n_boot: int = 2000,
                       level: float = 0.95, block: int = 1,
                       groups: Optional[Sequence] = None) -> Interval:
    """
    Percentile bootstrap interval of ``stat``.

    * ``groups`` given: resample whole groups (cluster bootstrap) -- rows of a
      group are never split.
    * ``block > 1``: moving-block bootstrap for serially dependent values
      (keeps runs of ``block`` consecutive values together).
    """
    a = np.asarray(values, dtype=np.float64)
    n = len(a)
    if n == 0:
        raise ValueError("bootstrap_interval needs at least one value")
    point = float(stat(a))
    boots = np.empty(n_boot, dtype=np.float64)
    if groups is not None:
        g = np.asarray(groups)
        if len(g) != n:
            raise ValueError("groups must align with values")
        uniq, inv = np.unique(g, return_inverse=True)
        members = [np.flatnonzero(inv == k) for k in range(len(uniq))]
        for b in range(n_boot):
            pick = rng.integers(0, len(uniq), size=len(uniq))
            idx = np.concatenate([members[k] for k in pick])
            boots[b] = stat(a[idx])
        method = "cluster_bootstrap"
    elif block > 1:
        block = min(block, n)
        n_blocks = int(math.ceil(n / block))
        for b in range(n_boot):
            starts = rng.integers(0, n - block + 1, size=n_blocks)
            idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]
            boots[b] = stat(a[idx])
        method = "block_bootstrap"
    else:
        for b in range(n_boot):
            boots[b] = stat(a[rng.integers(0, n, size=n)])
        method = "bootstrap"
    alpha = 1.0 - level
    lo, hi = np.quantile(boots, [alpha / 2.0, 1.0 - alpha / 2.0])
    return Interval(point, float(lo), float(hi), float(boots.std(ddof=1)), n, None, method)
