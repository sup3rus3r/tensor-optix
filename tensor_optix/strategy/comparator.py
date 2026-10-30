"""Comparator for strategy candidates -- conservative-gate discipline
directly analogous to nanonets/abstention.py's percentile gate (don't
trust a single lucky result, applied to strategy promotion instead of
cross-task transfer).

Gates on the PESSIMISTIC end of the confidence interval (lo95), not the
mean, mirroring passes_percentile_gate's median-vs-p95 conservatism.
Hard-requires MIN_EVALUATIONS distinct seeds before the gate can ever
pass -- directly enforces the proposal's own non-goal: no promotion
from a single lucky seed.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from tensor_optix.ledger.records import StrategyStatus
from tensor_optix.ledger.store import ExperimentStore
from tensor_optix.selection.uncertainty import ScoreInterval, score_interval

MIN_EVALUATIONS = 3


@dataclass
class StrategyComparison:
    candidate_id: str
    task_env: str
    n_evaluations: int
    delta_interval: Optional[ScoreInterval]
    passes_margin_gate: bool
    beats_existing_approved: Dict[str, bool] = field(default_factory=dict)


def compare_candidate(
    store: ExperimentStore, candidate_id: str, task_env: str, *, margin: float = 0.0,
) -> StrategyComparison:
    """Compare a candidate's outcome_delta distribution against `margin`
    (0.0 = "any improvement at all" by default; callers set a stricter
    margin for a meaningful-benefit requirement).
    """
    evals = store.query_strategy_evaluations(candidate_id=candidate_id, task_env=task_env)
    deltas = [e.outcome_delta for e in evals if e.outcome_delta is not None]

    n = len(deltas)
    if n == 0:
        return StrategyComparison(
            candidate_id=candidate_id, task_env=task_env, n_evaluations=0,
            delta_interval=None, passes_margin_gate=False,
        )

    interval = score_interval(deltas)
    passes = n >= MIN_EVALUATIONS and interval.lo95 > margin

    return StrategyComparison(
        candidate_id=candidate_id, task_env=task_env, n_evaluations=n,
        delta_interval=interval, passes_margin_gate=passes,
    )


def compare_against_approved(
    store: ExperimentStore, candidate_id: str, task_env: str, *, margin: float = 0.0,
) -> StrategyComparison:
    """Like compare_candidate, but additionally checks whether the
    candidate's own lo95 exceeds each currently-approved strategy's hi95
    on this task_env -- recomputed live from their evaluations each
    call, never trusting a cached interval (a demoted strategy's old
    numbers must never silently keep influencing this check)."""
    comparison = compare_candidate(store, candidate_id, task_env, margin=margin)
    if comparison.delta_interval is None:
        return comparison

    beats: Dict[str, bool] = {}
    for entry in store.query_approved_strategies():
        if entry.candidate_id == candidate_id or entry.status != StrategyStatus.APPROVED:
            continue
        other = compare_candidate(store, entry.candidate_id, task_env, margin=margin)
        if other.delta_interval is None:
            continue
        beats[entry.candidate_id] = comparison.delta_interval.lo95 > other.delta_interval.hi95

    comparison.beats_existing_approved = beats
    return comparison
