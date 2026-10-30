"""DeploymentObserver -- wraps BackoffScheduler (reused standalone, not
reimplemented) with a periodic-re-evaluation-cycle calling convention
for the live adaptive runtime's drift/degradation detection.

One observe(score) call per periodic re-evaluation batch, not per
single training episode -- this reframes "episode" as "cycle"
uniformly; BackoffScheduler's thresholds then operate in cycle units,
which is a coarser, less noisy signal than raw episode scores (each
cycle already aggregates eval_episodes real rollouts).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from tensor_optix.core.backoff_scheduler import BackoffScheduler
from tensor_optix.core.types import LoopState
from tensor_optix.ledger.store import ExperimentStore

from .baseline import ROLE_BASELINE_EVAL


@dataclass
class ObservationResult:
    cycle: int
    score: float
    state: LoopState
    is_degrading: bool
    degradation_fired: bool
    best_score: Optional[float]


class DeploymentObserver:
    def __init__(
        self,
        store: ExperimentStore,
        *,
        task_env: str,
        algorithm: str,
        study_id: str,
        scheduler: Optional[BackoffScheduler] = None,
    ):
        self._store = store
        self._task_env = task_env
        self._algorithm = algorithm
        self._study_id = study_id
        self._scheduler = scheduler or BackoffScheduler(
            base_interval=1, plateau_threshold=3, dormant_threshold=10,
            degradation_threshold=0.95, min_episodes_before_degradation=3,
            score_window=20, trend_window=5,
        )
        self._cycle = 0

    def observe(self, score: float) -> ObservationResult:
        """Note on BackoffScheduler's own semantics (verified directly,
        not a bug introduced here): record_improvement() only updates
        best_score once the internal score_window has >= 5 entries --
        this is an intentional warm-up gate against early-noise
        poisoning, carried over unchanged from its single-training-run
        origin. The first ~5 observe() calls may therefore report a
        best_score that lags a genuinely higher early score; behavior
        is correct and stable from cycle 5 onward."""
        self._cycle += 1
        # Checked BEFORE mutating state, so it compares against the best
        # known prior to this cycle -- same usage pattern as the existing
        # LoopController call site.
        degradation_fired = self._scheduler.check_degradation(score)
        if self._scheduler.best_score is None or score > self._scheduler.best_score:
            self._scheduler.record_improvement(score)
        else:
            self._scheduler.record_non_improvement()
        if degradation_fired:
            self._scheduler.record_degradation()
        return ObservationResult(
            cycle=self._cycle, score=score, state=self._scheduler.current_state,
            is_degrading=self._scheduler.is_degrading(),
            degradation_fired=degradation_fired, best_score=self._scheduler.best_score,
        )

    def replay_history(self) -> None:
        """Reconstructs ledger-backed history and replays it through the
        scheduler -- makes Observer state durable across runtime
        restarts with zero new persistence code (the ledger is the
        source of truth)."""
        for score in self.score_history():
            self.observe(score)

    def score_history(self, *, limit: int = 100) -> List[float]:
        records = self._store.query_experiments(
            task_env=self._task_env, algorithm=self._algorithm, study_id=self._study_id, limit=limit,
        )
        evals = [
            r for r in records
            if r.config.get("role") == ROLE_BASELINE_EVAL and r.best_primary_score is not None
        ]
        evals.sort(key=lambda r: r.created_at)
        return [r.best_primary_score for r in evals]
