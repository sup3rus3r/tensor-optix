"""Small-sample-corrected confidence intervals.

tensor_optix/selection/uncertainty.py's score_interval() uses a fixed
z=1.96 normal-approximation margin regardless of n. At the small trial
counts realistic for this project (n=3-5), that understates the true
margin — a t-distribution critical value is meaningfully wider.

This module supersedes (does not edit) selection/uncertainty.py for
Nanonets' own use. The critical-value table below is hardcoded rather
than computed via scipy.stats.t.ppf to avoid adding scipy as a dependency
(scipy is only importable on the current dev machine as a side effect of
unrelated packages, not a declared project dependency) — see Phase 4
plan Sec 8. Values cover df 1-30 (two-tailed, 97.5th percentile); beyond
df=30 the z=1.96 approximation is already accurate, so no table entry is
needed there.
"""

import math
from dataclasses import dataclass
from typing import List

# t.ppf(0.975, df) for df = 1..30, i.e. the two-tailed 95% critical value.
_T_CRITICAL_975 = {
    1: 12.7062, 2: 4.3027, 3: 3.1824, 4: 2.7764, 5: 2.5706,
    6: 2.4469, 7: 2.3646, 8: 2.3060, 9: 2.2622, 10: 2.2281,
    11: 2.2010, 12: 2.1788, 13: 2.1604, 14: 2.1448, 15: 2.1314,
    16: 2.1199, 17: 2.1098, 18: 2.1009, 19: 2.0930, 20: 2.0860,
    21: 2.0796, 22: 2.0739, 23: 2.0687, 24: 2.0639, 25: 2.0595,
    26: 2.0555, 27: 2.0518, 28: 2.0484, 29: 2.0452, 30: 2.0423,
}
_Z_975 = 1.96  # normal approximation, accurate beyond df=30


def t_critical_975(df: int) -> float:
    """95% two-tailed critical value for df degrees of freedom."""
    if df < 1:
        raise ValueError(f"degrees of freedom must be >= 1, got {df}")
    if df in _T_CRITICAL_975:
        return _T_CRITICAL_975[df]
    return _Z_975


@dataclass
class SmallNInterval:
    mean: float
    std: float
    n: int
    lo95: float
    hi95: float


def small_n_interval(scores: List[float]) -> SmallNInterval:
    """Mean/std and a t-corrected 95% CI, honest at small n.

    Raises ValueError on an empty list, matching
    selection/uncertainty.py's score_interval() convention.
    """
    if not scores:
        raise ValueError("small_n_interval() requires at least one score")

    n = len(scores)
    mean = sum(scores) / n

    if n > 1:
        variance = sum((s - mean) ** 2 for s in scores) / (n - 1)
        std = math.sqrt(variance)
    else:
        std = 0.0

    if n > 1:
        critical = t_critical_975(n - 1)
        margin = critical * std / math.sqrt(n)
    else:
        margin = 0.0

    return SmallNInterval(mean=mean, std=std, n=n, lo95=mean - margin, hi95=mean + margin)
