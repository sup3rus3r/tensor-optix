"""The out-of-loop human promotion path. Never imported by
live_runtime.py, experiment_stage.py, observer.py, or baseline.py --
this module is invoked manually, after a human has already reviewed a
recommendation and explicitly called store.approve_strategy(). This
is the ONLY code path that can make a strategy candidate's weights
become the new deployed baseline.
"""

from __future__ import annotations

from typing import Optional

from tensor_optix.ledger.records import EvaluationStatus, ExperimentState, StrategyStatus
from tensor_optix.ledger.store import ExperimentStore

from .baseline import ROLE_BASELINE


def promote_candidate_to_baseline(
    store: ExperimentStore,
    candidate_id: str,
    *,
    reviewer: str,
    task_env: str,
    algorithm: str,
    study_id: str,
) -> str:
    """Promote an already human-approved strategy candidate's best
    weights to become the new deployed baseline.

    Raises ValueError if the candidate has not already been approved
    via store.approve_strategy() -- this function refuses to act on an
    unreviewed candidate; it never performs that review itself.
    Returns the new deployed_baseline experiment_id.
    """
    entry = store.get_strategy_registry_entry(candidate_id)
    if entry is None or entry.status != StrategyStatus.APPROVED:
        raise ValueError(
            f"Candidate {candidate_id!r} is not approved (status="
            f"{entry.status if entry else None}). promote_candidate_to_baseline "
            "refuses to act on a candidate that has not already been reviewed "
            "and approved via store.approve_strategy()."
        )

    weights_path = _best_scoring_weights_path(store, candidate_id, task_env)
    if weights_path is None:
        raise ValueError(
            f"No scored, completed experiment with a recorded weights_path "
            f"was found for candidate {candidate_id!r} on task_env={task_env!r}."
        )

    new_experiment_id = f"{study_id}_baseline_promoted_{candidate_id[:8]}"
    record = store.create_experiment(
        experiment_id=new_experiment_id, task_env=task_env, algorithm=algorithm,
        framework="torch", seed=0, checkpoint_dir="", study_id=study_id,
        config={"role": ROLE_BASELINE, "weights_path": weights_path,
                "promoted_from_candidate_id": candidate_id, "promoted_by": reviewer},
    )
    new_experiment_id = record.experiment_id
    store.start_experiment(new_experiment_id)
    store.finalize_experiment(new_experiment_id, state=ExperimentState.COMPLETED,
                               evaluation_status=EvaluationStatus.VALIDATED)
    return new_experiment_id


def _best_scoring_weights_path(store: ExperimentStore, candidate_id: str, task_env: str) -> Optional[str]:
    evals = store.query_strategy_evaluations(candidate_id=candidate_id, task_env=task_env)
    best_score = float("-inf")
    best_weights_path = None
    for e in evals:
        record = store.get_experiment(e.experiment_id)
        if record is None or record.best_primary_score is None:
            continue
        weights_path = record.final_metrics.get("weights_path")
        if weights_path is None:
            continue
        if record.best_primary_score > best_score:
            best_score = record.best_primary_score
            best_weights_path = weights_path
    return best_weights_path
