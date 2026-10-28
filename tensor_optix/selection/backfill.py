"""Backfill a completed TrialOrchestrator run into the experiment ledger.

TrialOrchestrator (tensor_optix/orchestrator.py) has no awareness of the
ledger — it is a proven, standalone Optuna-TPE trial search. This module
is a one-shot, post-hoc translation: after a study finishes, turn each
optuna FrozenTrial into one ExperimentRecord. It does not execute or
schedule anything, consistent with keeping worker execution separate from
scheduling decisions.
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional

import optuna

from tensor_optix.ledger.records import EvaluationStatus, ExperimentState, ExperimentRecord
from tensor_optix.ledger.store import ExperimentStore

if TYPE_CHECKING:
    from tensor_optix.orchestrator import TrialOrchestrator

_STATE_MAP = {
    optuna.trial.TrialState.COMPLETE: ExperimentState.COMPLETED,
    optuna.trial.TrialState.PRUNED: ExperimentState.CANCELLED,
    optuna.trial.TrialState.FAIL: ExperimentState.FAILED,
    optuna.trial.TrialState.RUNNING: ExperimentState.RUNNING,
    optuna.trial.TrialState.WAITING: ExperimentState.QUEUED,
}


def backfill_study_to_ledger(
    orchestrator: "TrialOrchestrator",
    store: ExperimentStore,
    *,
    task_env: str,
    algorithm: str,
    framework: str = "torch",
    seed: int = 0,
    sampler_label: Optional[str] = None,
) -> List[ExperimentRecord]:
    """Write one ExperimentRecord per trial in `orchestrator.study`.

    Idempotent: safe to call twice on the same study. create_experiment is
    idempotent on an explicit experiment_id, and finalize_experiment only
    writes once per row (WHERE finished_at IS NULL) — both guarantees are
    the ledger's own, not reinvented here.

    started_at/finished_at on the resulting rows reflect backfill wall-clock
    time, not trial.datetime_start/trial.datetime_complete — relative
    ordering and idempotency matter more here than exact historical
    timestamps, since this is a post-hoc translation, not a live recording.
    """
    label = sampler_label or type(orchestrator.sampler).__name__
    study_id = orchestrator.study.study_name
    checkpoint_dir = orchestrator.run_ckpt_dir or ""

    records: List[ExperimentRecord] = []
    for trial in orchestrator.study.trials:
        experiment_id = f"{study_id}-trial-{trial.number}"
        config: Dict[str, Any] = {
            "params": trial.params,
            "sampler": label,
            "trial_number": trial.number,
        }

        store.create_experiment(
            experiment_id=experiment_id,
            task_env=task_env,
            algorithm=algorithm,
            framework=framework,
            seed=seed,
            checkpoint_dir=checkpoint_dir,
            config=config,
            study_id=study_id,
        )
        store.start_experiment(experiment_id)
        if trial.value is not None:
            store.record_progress(experiment_id, best_primary_score=trial.value)

        mapped_state = _STATE_MAP.get(trial.state, ExperimentState.FAILED)
        store.finalize_experiment(
            experiment_id,
            state=mapped_state,
            evaluation_status=EvaluationStatus.PRELIMINARY,
            final_metrics={"trial_number": float(trial.number)},
        )

        record = store.get_experiment(experiment_id)
        assert record is not None
        records.append(record)

    return records
