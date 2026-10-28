"""Descriptive-statistics uncertainty over repeated ledger observations.

This is deliberately NOT a predictive uncertainty model. It answers "how
much does this method's score vary across the trials we've already run,"
not "how confident are we about an unseen configuration." Learned predictive
uncertainty belongs to Phase 4+ (the cross-task meta-model / Nanonets).
"""

import math
from dataclasses import dataclass
from typing import List


@dataclass
class ScoreInterval:
    mean: float
    std: float
    n: int
    lo95: float
    hi95: float


def score_interval(scores: List[float]) -> ScoreInterval:
    """Compute mean/std and a normal-approximation 95% CI over `scores`.

    Raises ValueError on an empty list — an empty history is a real caller
    error to surface, not something to paper over with a placeholder
    interval.
    """
    if not scores:
        raise ValueError("score_interval() requires at least one score")

    n = len(scores)
    mean = sum(scores) / n

    if n > 1:
        variance = sum((s - mean) ** 2 for s in scores) / (n - 1)
        std = math.sqrt(variance)
    else:
        std = 0.0

    margin = 1.96 * std / math.sqrt(n) if n > 1 else 0.0
    return ScoreInterval(mean=mean, std=std, n=n, lo95=mean - margin, hi95=mean + margin)
