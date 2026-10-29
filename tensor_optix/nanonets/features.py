"""Task feature extraction (phi(tau)) — ledger-computable only.

Deliberately excludes anything requiring live environment instantiation
(observation/action space dims) or training-time instrumentation that
doesn't exist in the ledger today (gradient norms). See Phase 4 plan
Sec 3 for the full, explicit in/out-of-scope list and the confirmed
absence of gradient-norm fields anywhere in tensor_optix/ledger/ or
core/diagnostic_controller.py.

Feature definitions are versioned per the proposal's "feature
definitions must be versioned" requirement.
"""

from dataclasses import dataclass
from typing import Optional

from tensor_optix.ledger.records import ExperimentState
from tensor_optix.ledger.store import ExperimentStore

TASK_FEATURES_SCHEMA_VERSION = 1


@dataclass
class TaskFeatures:
    task_env: str
    algorithm: str
    n_completed: int
    score_mean: Optional[float]
    score_std: Optional[float]
    generalization_gap_mean: Optional[float]
    schema_version: int = TASK_FEATURES_SCHEMA_VERSION


def extract_task_features(store: ExperimentStore, *, task_env: str) -> TaskFeatures:
    """Compute TaskFeatures for one task_env from ledger history alone.

    If multiple algorithms have been run on this task_env, `algorithm`
    reports the most common one among completed records; callers needing
    per-algorithm features should filter by algorithm themselves before
    calling this (e.g. by querying a narrower set and deriving features
    manually) — this function reports one summary per task_env, matching
    how NanonetsSelector consumes it.
    """
    records = store.query_experiments(task_env=task_env, limit=1000)
    completed = [r for r in records if r.state == ExperimentState.COMPLETED]

    if not completed:
        return TaskFeatures(
            task_env=task_env, algorithm="", n_completed=0,
            score_mean=None, score_std=None, generalization_gap_mean=None,
        )

    algo_counts: dict = {}
    for r in completed:
        algo_counts[r.algorithm] = algo_counts.get(r.algorithm, 0) + 1
    algorithm = max(algo_counts, key=algo_counts.get)

    scores = [r.best_primary_score for r in completed if r.best_primary_score is not None]
    score_mean = sum(scores) / len(scores) if scores else None
    if scores and len(scores) > 1:
        mean = score_mean
        score_std = (sum((s - mean) ** 2 for s in scores) / (len(scores) - 1)) ** 0.5
    else:
        score_std = 0.0 if scores else None

    gaps = [r.generalization_gap for r in completed if r.generalization_gap is not None]
    generalization_gap_mean = sum(gaps) / len(gaps) if gaps else None

    return TaskFeatures(
        task_env=task_env,
        algorithm=algorithm,
        n_completed=len(completed),
        score_mean=score_mean,
        score_std=score_std,
        generalization_gap_mean=generalization_gap_mean,
    )
