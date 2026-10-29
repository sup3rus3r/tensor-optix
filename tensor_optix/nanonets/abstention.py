"""Abstention mechanism: the hard percentile gate plus continuous
similarity-decay weighting that prevents Nanonets from blindly
transferring a bad or dissimilar source task's evidence.

Two lines of defense, matching the Phase 4 plan Sec 4:

1. Hard gate (Feurer et al. style): a source task is eligible only if its
   median score is not worse than the target's own 95th percentile. If
   every source fails this gate, the caller falls back to the baseline
   StrategySelector entirely.
2. Continuous similarity-decay: among sources that pass the gate, scale
   each one's contribution down further by a crude, honest proxy
   (algorithm-family overlap and relative sample-size ratio) rather than
   treating "passed the gate" as a binary all-or-nothing green light.

No learned embedding or similarity metric is used — none is validated
for RL per the research report backing this phase.
"""

import math
from typing import List, Optional

MIN_TARGET_N = 2


def normalize_score(raw_score: float, solve_threshold: float) -> Optional[float]:
    """Normalize a raw score to 'fraction of solve_threshold magnitude
    above/below solved', making scores comparable across tasks whose raw
    scales and signs differ (e.g. CartPole ~0-500 vs Acrobot ~-500-0).

    Plain division (raw/threshold) is unsafe across a sign boundary — it
    inverts the above/below-threshold sense for a negative threshold
    (e.g. Acrobot's -17 vs -100 gives 0.17, misreading a genuinely good
    score as "far below solved"). Signed distance scaled by the
    threshold's own magnitude avoids this: positive means at/above that
    task's own notion of solved, negative means below, for any sign of
    threshold.

    Returns None if solve_threshold is 0.0 (degenerate — no magnitude to
    scale by); callers must treat None the same as "threshold unknown."
    """
    if solve_threshold == 0.0:
        return None
    return (raw_score - solve_threshold) / abs(solve_threshold)


def normalize_scores(raw_scores: List[float], solve_threshold: Optional[float]) -> Optional[List[float]]:
    """Normalize a whole list of scores. Returns None (not a silently
    partial list) if solve_threshold is None or 0.0 — callers must treat
    this the same as "cannot establish a comparable scale" rather than
    falling back to raw, unnormalized comparison."""
    if solve_threshold is None:
        return None
    normalized = [normalize_score(s, solve_threshold) for s in raw_scores]
    if any(n is None for n in normalized):
        return None
    return normalized


def _median(values: List[float]) -> float:
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    mid = n // 2
    if n % 2 == 1:
        return sorted_vals[mid]
    return (sorted_vals[mid - 1] + sorted_vals[mid]) / 2.0


def _percentile(values: List[float], pct: float) -> float:
    """Linear-interpolation percentile, pct in [0, 100]."""
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    rank = (pct / 100.0) * (n - 1)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return sorted_vals[lo]
    frac = rank - lo
    return sorted_vals[lo] + frac * (sorted_vals[hi] - sorted_vals[lo])


def passes_percentile_gate(source_scores: List[float], target_scores: List[float]) -> bool:
    """A source task is eligible iff its median score is not worse than
    the target's own 95th percentile (higher-is-better primary scores).

    Returns False (gate fails) if either list is empty, or the target
    has fewer than MIN_TARGET_N observations to compute a meaningful p95.
    """
    if not source_scores or len(target_scores) < MIN_TARGET_N:
        return False
    target_p95 = _percentile(target_scores, 95.0)
    source_median = _median(source_scores)
    return source_median >= target_p95


def similarity_weight(
    *, source_algorithm: str, target_algorithm: str,
    source_n: int, target_n: int,
) -> float:
    """Crude, honest similarity-decay factor in (0, 1].

    Algorithm-family overlap (exact match) contributes a full weight of
    1.0; a mismatch halves it. Relative sample-size ratio further scales
    the weight down when the source has far more or far fewer
    observations than the target (extreme imbalance is itself a weak
    signal of a differently-behaved task/regime).
    """
    algo_factor = 1.0 if source_algorithm == target_algorithm else 0.5

    if target_n <= 0 or source_n <= 0:
        size_factor = 0.5
    else:
        ratio = min(source_n, target_n) / max(source_n, target_n)
        size_factor = 0.5 + 0.5 * ratio  # ranges (0.5, 1.0]

    return algo_factor * size_factor
