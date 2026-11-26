"""DeploymentObserver -- wraps ConvergenceMonitor (reused standalone, not
reimplemented) with a periodic-re-evaluation-cycle calling convention
for the live adaptive runtime's drift/degradation detection.

One observe(score) call per periodic re-evaluation batch, not per
single training episode -- this reframes "episode" as "cycle"
uniformly; the monitor's timescales and patience then operate in cycle
units, which is a coarser, less noisy signal than raw episode scores (each
cycle already aggregates eval_episodes real rollouts).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from tensor_optix.core.convergence import ConvergenceMonitor, ConvergenceRegime, ConvergenceReport
from tensor_optix.core.types import LoopState
from tensor_optix.ledger.store import ExperimentStore

from .baseline import ROLE_BASELINE_EVAL

# Cycle-scale settings: a deployment is re-evaluated tens of times, not
# thousands, so the windows are short; patience keeps the historical
# COOLING after 3 / DORMANT after 10 cycles without new ground.
OBSERVER_CONVERGENCE_DEFAULTS = dict(
    timescales=(4, 12, 48), cooling_patience=3, dormant_patience=10, persist=2,
)


@dataclass
class ObservationResult:
    cycle: int
    score: float
    state: LoopState
    is_degrading: bool
    degradation_fired: bool
    best_score: Optional[float]
    report: Optional[ConvergenceReport] = None


class DeploymentObserver:
    """
    Degradation for a deployed policy fires on either:

      * a confirmed collapse (the monitor's COLLAPSING regime), or
      * an acute drop: a single cycle more than ``acute_k`` noise standard
        deviations below the peak. Each cycle already averages many
        rollouts, so one cycle that far down is strong evidence on its own.

    ``degradation_fired`` is one-shot per episode of trouble: after it fires
    it re-arms only once the score is back at the peak (PLATEAU or
    IMPROVING), so a policy that stays down -- or an acute drop that is then
    confirmed as a collapse -- triggers one research cycle, not one per cycle.
    ``is_degrading`` reports the ongoing condition on every cycle.
    """

    def __init__(
        self,
        store: ExperimentStore,
        *,
        task_env: str,
        algorithm: str,
        study_id: str,
        monitor: Optional[ConvergenceMonitor] = None,
        acute_k: float = 4.0,
    ):
        self._store = store
        self._task_env = task_env
        self._algorithm = algorithm
        self._study_id = study_id
        self._monitor = monitor or ConvergenceMonitor(**OBSERVER_CONVERGENCE_DEFAULTS)
        self._acute_k = float(acute_k)
        self._cycle = 0
        self._alarm_active = False

    @property
    def monitor(self) -> ConvergenceMonitor:
        return self._monitor

    def observe(self, score: float) -> ObservationResult:
        self._cycle += 1
        # The acute check compares against the peak and noise known BEFORE
        # this cycle, so the drop being judged cannot inflate its own bar.
        prev = self._monitor.last_report
        acute = (
            prev is not None and prev.peak is not None and prev.sigma is not None
            and score < prev.peak - self._acute_k * prev.sigma
        )
        report = self._monitor.observe(score, at=self._cycle)
        collapsing = report.regime == ConvergenceRegime.COLLAPSING
        if report.regime in (ConvergenceRegime.PLATEAU, ConvergenceRegime.IMPROVING) and not acute:
            self._alarm_active = False          # recovered: re-arm
        fired = (report.collapse_confirmed or acute) and not self._alarm_active
        if fired:
            self._alarm_active = True
        return ObservationResult(
            cycle=self._cycle, score=score, state=report.state,
            is_degrading=collapsing or acute,
            degradation_fired=fired,
            best_score=report.best_score, report=report,
        )

    def replay_history(self) -> None:
        """Reconstructs ledger-backed history and replays it through the
        monitor -- makes Observer state durable across runtime
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
