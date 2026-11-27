"""
Hypothesis tests discovery's detectors use. All take an explicit
``numpy.random.Generator`` -- results are reproducible and never touch global
RNG state.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from .distributions import kolmogorov_sf


@dataclass(frozen=True)
class TestResult:
    statistic: float
    p_value: float
    null_mean: Optional[float] = None
    null_std: Optional[float] = None
    n_perm: Optional[int] = None


def permutation_test(statistic: Callable[[np.ndarray], float], labels: np.ndarray,
                     rng: np.random.Generator, *, n_perm: int = 200,
                     alternative: str = "greater",
                     permute: Optional[Callable[[np.ndarray, np.random.Generator], np.ndarray]] = None
                     ) -> TestResult:
    """
    Monte-Carlo permutation test. ``statistic(labels)`` is computed on the
    real labels and on ``n_perm`` permutations of them (or of whatever
    ``permute`` returns, e.g. a block- or group-preserving shuffle).

    p-value uses the (b + 1) / (n_perm + 1) correction, so it is never 0 and
    the test is exact-level under exchangeability.
    """
    if alternative not in ("greater", "less", "two-sided"):
        raise ValueError("alternative must be 'greater', 'less' or 'two-sided'")
    obs = float(statistic(labels))
    perm = permute if permute is not None else (lambda x, r: r.permutation(x))
    null = np.empty(n_perm, dtype=np.float64)
    for i in range(n_perm):
        null[i] = statistic(perm(labels, rng))
    if alternative == "greater":
        b = int(np.sum(null >= obs))
    elif alternative == "less":
        b = int(np.sum(null <= obs))
    else:
        centre = float(null.mean())
        b = int(np.sum(np.abs(null - centre) >= abs(obs - centre)))
    p = (b + 1.0) / (n_perm + 1.0)
    return TestResult(obs, p, float(null.mean()), float(null.std(ddof=1)) if n_perm > 1 else None, n_perm)


def block_permutation(block: int) -> Callable[[np.ndarray, np.random.Generator], np.ndarray]:
    """Shuffle contiguous blocks of length ``block`` (keeps short-range serial
    dependence inside blocks, breaks it across them)."""
    def _perm(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        n = len(x)
        b = max(1, min(block, n))
        starts = np.arange(0, n, b)
        order = rng.permutation(len(starts))
        return np.concatenate([x[starts[k]:starts[k] + b] for k in order])
    return _perm


def ks_2samp(a: np.ndarray, b: np.ndarray) -> TestResult:
    """
    Two-sample Kolmogorov-Smirnov test, asymptotic p-value with Stephens'
    small-sample correction. Ties are handled by evaluating both ECDFs on the
    pooled sorted values. NaNs are ignored.
    """
    a = np.sort(np.asarray(a, dtype=np.float64)[~np.isnan(np.asarray(a, dtype=np.float64))])
    b = np.sort(np.asarray(b, dtype=np.float64)[~np.isnan(np.asarray(b, dtype=np.float64))])
    n1, n2 = len(a), len(b)
    if n1 == 0 or n2 == 0:
        raise ValueError("ks_2samp needs non-empty samples")
    pooled = np.concatenate([a, b])
    cdf1 = np.searchsorted(a, pooled, side="right") / n1
    cdf2 = np.searchsorted(b, pooled, side="right") / n2
    d = float(np.max(np.abs(cdf1 - cdf2)))
    ne = n1 * n2 / (n1 + n2)
    sq = math.sqrt(ne)
    lam = (sq + 0.12 + 0.11 / sq) * d
    return TestResult(d, kolmogorov_sf(lam))


def autocorrelation(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Sample autocorrelation at lags 1..max_lag (biased estimator, standard)."""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        return np.zeros(max_lag)
    xc = x - x.mean()
    denom = float((xc * xc).sum())
    if denom == 0.0:
        return np.zeros(max_lag)
    out = np.zeros(max_lag)
    for k in range(1, max_lag + 1):
        if k >= n:
            break
        out[k - 1] = float((xc[k:] * xc[:-k]).sum()) / denom
    return out


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population stability index over quantile bins of ``expected``."""
    e = np.asarray(expected, dtype=np.float64)
    a = np.asarray(actual, dtype=np.float64)
    e, a = e[np.isfinite(e)], a[np.isfinite(a)]
    if len(e) == 0 or len(a) == 0:
        raise ValueError("psi needs non-empty samples")
    edges = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    pe = np.histogram(e, edges)[0] / len(e)
    pa = np.histogram(a, edges)[0] / len(a)
    eps = 1e-6
    pe, pa = np.clip(pe, eps, None), np.clip(pa, eps, None)
    return float(np.sum((pa - pe) * np.log(pa / pe)))
