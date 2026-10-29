"""Phase 4 v1 -- cross-task statistical pooling ("Nanonets").

Scope boundary: this package implements a precision-weighted shrinkage
estimator (hierarchical/empirical-Bayes pooling), NOT a Gaussian Process,
NOT RGPE, and NOT any deep/neural meta-learner. That choice is explicit
and research-grounded -- see reports/Cross-task meta-learning for HPO
transfer.md and the Phase 4 implementation plan. With only a handful of
genuinely distinct task_envs available, this is the simplest method with
real published evidence of working at single-digit task counts, and it
structurally avoids the dominant failure mode (small-sample cross-task
correlation sign-flipping) by never estimating a correlation coefficient
at all.

NanonetsSelector composes tensor_optix.selection.StrategySelector for its
fallback path; it does not edit or subclass it. StrategySelector and
selection/uncertainty.py remain untouched by this package.
"""

from .abstention import passes_percentile_gate, similarity_weight
from .features import TASK_FEATURES_SCHEMA_VERSION, TaskFeatures, extract_task_features
from .pooling import PooledResult, SourceContribution, pool_estimates
from .selector import NanonetsSelector, PooledEstimate
from .smalln import SmallNInterval, small_n_interval, t_critical_975

__all__ = [
    "passes_percentile_gate",
    "similarity_weight",
    "TASK_FEATURES_SCHEMA_VERSION",
    "TaskFeatures",
    "extract_task_features",
    "PooledResult",
    "SourceContribution",
    "pool_estimates",
    "NanonetsSelector",
    "PooledEstimate",
    "SmallNInterval",
    "small_n_interval",
    "t_critical_975",
]
