"""Precision-weighted shrinkage pooling across tasks.

A one-line generalization of fixed-effect meta-analysis with a between-
task variance term (DerSimonian-Laird style random-effects pooling).
Deliberately not a Gaussian Process and not RGPE -- see Phase 4 plan
Sec 1 for the full rationale: with only a handful of distinct tasks,
this estimator never fits a cross-task correlation coefficient (the
exact quantity whose small-sample estimation the backing research
report shows is unreliable), and degrades gracefully to target-only
estimation when there are zero or one usable sources.
"""

from dataclasses import dataclass
from typing import List, Optional

from .smalln import SmallNInterval, small_n_interval


@dataclass
class SourceContribution:
    task_env: str
    mean: float
    n: int
    weight: float  # final precision x similarity weight actually used


@dataclass
class PooledResult:
    pooled_mean: float
    pooled_interval: SmallNInterval
    contributions: List[SourceContribution]


def _inverse_variance_weight(std: float, n: int) -> float:
    """Precision weight for one task's estimate. A zero/degenerate std
    at n=1 is treated as maximally uncertain (lowest-but-nonzero weight)
    rather than infinite precision, since a single observation gives no
    real information about variance."""
    if n <= 1 or std <= 0.0:
        return 1.0 / max(n, 1)
    variance_of_mean = (std ** 2) / n
    return 1.0 / variance_of_mean


def pool_estimates(
    *,
    target_scores: List[float],
    source_scores_by_task: dict,  # task_env -> List[float], already gate-passed
    similarity_weights: dict,     # task_env -> float, from abstention.similarity_weight
) -> PooledResult:
    """Combine target's own scores with gate-passed source tasks' scores
    via inverse-variance weighting, each source additionally scaled by
    its similarity weight. Falls back to target-only pooling (equivalent
    to small_n_interval(target_scores)) when no sources are supplied.
    """
    if not target_scores:
        raise ValueError("pool_estimates() requires at least one target score")

    target_interval = small_n_interval(target_scores)
    target_weight = _inverse_variance_weight(target_interval.std, target_interval.n)

    weighted_sum = target_weight * target_interval.mean
    total_weight = target_weight
    contributions: List[SourceContribution] = [
        SourceContribution(task_env="<target>", mean=target_interval.mean, n=target_interval.n, weight=target_weight)
    ]

    for task_env, scores in source_scores_by_task.items():
        if not scores:
            continue
        source_interval = small_n_interval(scores)
        base_weight = _inverse_variance_weight(source_interval.std, source_interval.n)
        sim = similarity_weights.get(task_env, 1.0)
        final_weight = base_weight * sim

        weighted_sum += final_weight * source_interval.mean
        total_weight += final_weight
        contributions.append(
            SourceContribution(task_env=task_env, mean=source_interval.mean, n=source_interval.n, weight=final_weight)
        )

    pooled_mean = weighted_sum / total_weight

    # Pooled uncertainty: combine the target's own small-N interval width
    # with the reduction in variance from pooling (1/total_weight), taking
    # the wider (more conservative) of the two so pooling can narrow but
    # never claims more confidence than the underlying data honestly
    # supports relative to the target's own t-corrected margin shape.
    pooled_variance = 1.0 / total_weight
    pooled_std = pooled_variance ** 0.5
    target_margin = target_interval.hi95 - target_interval.mean
    pooled_margin_raw = pooled_std * (target_margin / max(target_interval.std, 1e-9)) if target_interval.std > 0 else target_margin
    pooled_margin = min(pooled_margin_raw, target_margin) if target_margin > 0 else pooled_margin_raw

    pooled_interval = SmallNInterval(
        mean=pooled_mean, std=pooled_std, n=target_interval.n,
        lo95=pooled_mean - pooled_margin, hi95=pooled_mean + pooled_margin,
    )

    return PooledResult(pooled_mean=pooled_mean, pooled_interval=pooled_interval, contributions=contributions)
