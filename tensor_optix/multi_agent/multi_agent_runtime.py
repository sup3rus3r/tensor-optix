"""
tensor_optix.multi_agent.multi_agent_runtime — MultiAgentRunner: the
multi-agent training loop.

Each seat gets the same machinery the single-agent LoopController uses,
applied independently per seat:

  - an evaluator scoring that seat's EpisodeData every window,
  - a BackoffScheduler doing trend-based convergence detection
    (ACTIVE -> COOLING -> DORMANT),
  - a CheckpointRegistry holding that seat's best snapshot.

What "converged" means for the run as a whole is an explicit choice
(`stop_when`): by default the run stops once EVERY seat is DORMANT — in a
competitive game one seat plateauing while its opponent is still improving
is not convergence, it is one side losing ground.

Not done here (single-agent LoopController features with no obvious
multi-agent meaning yet): online hyperparameter tuning, degradation
rollback during training, policy spawning.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Hashable, List, Optional

from tensor_optix.core.backoff_scheduler import BackoffScheduler
from tensor_optix.core.checkpoint_registry import CheckpointRegistry
from tensor_optix.core.types import EvalMetrics, HyperparamSet, LoopState, PolicySnapshot

logger = logging.getLogger(__name__)


class MultiAgentCallback:
    """
    Hooks into MultiAgentRunner. Subclass and override what you need
    (any object with some of these methods also works).
    """

    def on_run_start(self) -> None: ...
    def on_window_end(self, window_id: int, metrics: Dict[Hashable, EvalMetrics]) -> None: ...
    def on_improvement(self, agent_id: Hashable, snapshot: Optional[PolicySnapshot], score: float) -> None: ...
    def on_plateau(self, agent_id: Hashable, window_id: int) -> None: ...
    def on_converged(self, agent_id: Hashable, window_id: int) -> None: ...
    def on_run_stop(self, reason: str) -> None: ...


@dataclass
class SeatResult:
    """Per-seat outcome of a run."""
    agent_id: Hashable
    state: LoopState
    converged: bool
    windows_scored: int
    best_score: Optional[float]
    last_score: Optional[float]
    best_snapshot: Optional[PolicySnapshot]


@dataclass
class MultiAgentRunResult:
    """What run() returns."""
    windows_run: int
    stop_reason: str            # "converged" | "max_windows" | "stopped" | "pipeline_exhausted"
    seats: Dict[Hashable, SeatResult] = field(default_factory=dict)

    @property
    def converged(self) -> bool:
        return self.stop_reason == "converged"


class _Seat:
    def __init__(self, agent_id, agent, scheduler: BackoffScheduler, registry: Optional[CheckpointRegistry]):
        self.agent_id = agent_id
        self.agent = agent
        self.scheduler = scheduler
        self.registry = registry
        self.best_score: Optional[float] = None
        self.last_score: Optional[float] = None
        self.best_snapshot: Optional[PolicySnapshot] = None
        self.windows_scored = 0
        self.converged = False


class MultiAgentRunner:
    """
    Owns N agents (one per seat) and one multi-agent pipeline
    (ParallelMultiAgentPipeline or AECMultiAgentPipeline).

    Each window, for every seat that took at least one step:
      1. agent.learn(episode_data)
      2. score the window with `evaluator` (default: mean return of the
         episodes that completed in the window)
      3. checkpoint the seat if the score beats its best by more than
         `improvement_margin` (only when `checkpoint_dir` is set)
      4. feed the score to the seat's BackoffScheduler; a seat whose score
         trend has been flat for `dormant_threshold` consecutive windows is
         converged (DORMANT)

    The run stops when:
      - `stop_when="all"` (default): every seat has converged
      - `stop_when="any"`: the first seat converges
      - `max_windows` is reached, the pipeline ends, or stop() is called
    Convergence stopping can be disabled with `stop_on_convergence=False`.

    When the run ends and `restore_best=True` (default), every seat with a
    checkpoint is restored to its best snapshot. Note for competitive
    games: each seat's "best" is its best score against the opponents it
    faced AT THAT TIME, so restored seats may come from different stages
    of training. Pass `restore_best=False` to keep the final, jointly
    trained weights (best snapshots stay on disk and in the result).

    A seat that got no turn in a window (possible with the AEC pipeline)
    is skipped for that window — not learned, not scored, and it does not
    count toward that seat's plateau counter.

    Args:
        pipeline, agents:      as before.
        evaluator:             a BaseEvaluator, or {agent_id: BaseEvaluator}.
        checkpoint_dir:        root directory; each seat checkpoints into
                               <checkpoint_dir>/<agent_id>. None = no
                               checkpointing (and nothing to restore).
        max_snapshots:         snapshots kept per seat.
        improvement_margin:    minimum score gain over the seat's best for
                               a new checkpoint.
        plateau_threshold / dormant_threshold / min_windows_before_dormant /
        noise_k / score_window / trend_window:
                               per-seat BackoffScheduler settings (same
                               meaning as RLOptimizer's). Each seat is
                               evaluated every window (no eval backoff).
        stop_when:             "all" | "any".
        stop_on_convergence:   False = only max_windows / stop() end the run.
        restore_best:          restore each seat's best snapshot at the end.
        callbacks:             MultiAgentCallback-style objects.
    """

    def __init__(
        self,
        pipeline: Any,
        agents: Dict[Hashable, Any],
        *,
        evaluator: Any = None,
        checkpoint_dir: Optional[str] = None,
        max_snapshots: int = 10,
        improvement_margin: float = 0.0,
        plateau_threshold: int = 5,
        dormant_threshold: int = 20,
        min_windows_before_dormant: int = 0,
        noise_k: float = 2.0,
        score_window: int = 20,
        trend_window: int = 8,
        stop_when: str = "all",
        stop_on_convergence: bool = True,
        restore_best: bool = True,
        callbacks: Optional[List[Any]] = None,
    ) -> None:
        if stop_when not in ("all", "any"):
            raise ValueError(f"stop_when must be 'all' or 'any', got {stop_when!r}")
        self._pipeline = pipeline
        self._agents = agents
        self._improvement_margin = improvement_margin
        self._stop_when = stop_when
        self._stop_on_convergence = stop_on_convergence
        self._restore_best = restore_best
        self._callbacks: List[Any] = list(callbacks or [])
        self._stop_requested = False

        if evaluator is None:
            # Framework-agnostic despite the name: numpy-only mean episode return.
            from tensor_optix.adapters.pytorch.torch_evaluator import TorchEvaluator
            evaluator = TorchEvaluator()
        self._evaluators: Dict[Hashable, Any] = (
            dict(evaluator) if isinstance(evaluator, dict)
            else {aid: evaluator for aid in agents}
        )
        missing = [aid for aid in agents if aid not in self._evaluators]
        if missing:
            raise ValueError(f"No evaluator supplied for seat(s): {missing}")

        self._seats: Dict[Hashable, _Seat] = {}
        for aid, agent in agents.items():
            registry = None
            if checkpoint_dir is not None:
                registry = CheckpointRegistry(
                    os.path.join(checkpoint_dir, _safe_dirname(aid)),
                    max_snapshots=max_snapshots,
                )
            scheduler = BackoffScheduler(
                base_interval=1,
                backoff_factor=1.0,          # every window is evaluated
                plateau_threshold=plateau_threshold,
                dormant_threshold=dormant_threshold,
                min_episodes_before_dormant=min_windows_before_dormant,
                noise_k=noise_k,
                score_window=score_window,
                trend_window=trend_window,
            )
            self._seats[aid] = _Seat(aid, agent, scheduler, registry)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def stop(self) -> None:
        """Stop cleanly after the current window."""
        self._stop_requested = True

    def add_callback(self, callback: Any) -> None:
        self._callbacks.append(callback)

    @property
    def states(self) -> Dict[Hashable, LoopState]:
        return {aid: seat.scheduler.current_state for aid, seat in self._seats.items()}

    @property
    def best_snapshots(self) -> Dict[Hashable, Optional[PolicySnapshot]]:
        return {aid: seat.best_snapshot for aid, seat in self._seats.items()}

    @property
    def best_scores(self) -> Dict[Hashable, Optional[float]]:
        return {aid: seat.best_score for aid, seat in self._seats.items()}

    def run(self, max_windows: Optional[int] = None) -> MultiAgentRunResult:
        self._stop_requested = False
        self._pipeline.setup()
        self._fire("on_run_start")
        windows_run = 0
        stop_reason = "pipeline_exhausted"
        try:
            for window in self._pipeline.episodes():
                window_metrics: Dict[Hashable, EvalMetrics] = {}
                for aid, episode_data in window.items():
                    if len(episode_data.rewards) == 0:
                        continue
                    seat = self._seats.get(aid)
                    if seat is None:
                        continue
                    diagnostics = seat.agent.learn(episode_data) or {}
                    metrics = self._evaluators[aid].score(episode_data, diagnostics)
                    metrics.episode_id = windows_run
                    window_metrics[aid] = metrics
                    self._update_seat(seat, metrics, windows_run)

                self._fire("on_window_end", windows_run, window_metrics)
                windows_run += 1

                if self._stop_requested:
                    stop_reason = "stopped"
                    break
                if self._stop_on_convergence and self._run_converged():
                    stop_reason = "converged"
                    logger.info(
                        "MultiAgentRunner: converged after %d windows (stop_when=%s)",
                        windows_run, self._stop_when,
                    )
                    break
                if max_windows is not None and windows_run >= max_windows:
                    stop_reason = "max_windows"
                    break
        finally:
            try:
                if self._restore_best:
                    self._restore_best_weights()
            finally:
                self._pipeline.teardown()
                self._fire("on_run_stop", stop_reason)

        return MultiAgentRunResult(
            windows_run=windows_run,
            stop_reason=stop_reason,
            seats={
                aid: SeatResult(
                    agent_id=aid,
                    state=seat.scheduler.current_state,
                    converged=seat.converged,
                    windows_scored=seat.windows_scored,
                    best_score=seat.best_score,
                    last_score=seat.last_score,
                    best_snapshot=seat.best_snapshot,
                )
                for aid, seat in self._seats.items()
            },
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _update_seat(self, seat: _Seat, metrics: EvalMetrics, window_id: int) -> None:
        score = float(metrics.primary_score)
        seat.last_score = score
        seat.windows_scored += 1

        # --- checkpoint: raw score beats this seat's best ---
        if seat.best_score is None or score > seat.best_score + self._improvement_margin:
            seat.best_score = score
            if seat.registry is not None:
                seat.best_snapshot = seat.registry.save(
                    seat.agent, metrics, _hyperparams_of(seat.agent, window_id)
                )
            self._fire("on_improvement", seat.agent_id, seat.best_snapshot, score)

        # --- convergence: trend over the seat's recent scores ---
        sched = seat.scheduler
        first = sched.best_score is None
        sched.record_score(score)
        if first or sched.is_improving():
            sched.record_improvement(score, append=False)
            seat.converged = False
            return

        prev_state = sched.current_state
        sched.record_non_improvement()
        state = sched.current_state
        if state == LoopState.COOLING and prev_state != LoopState.COOLING:
            self._fire("on_plateau", seat.agent_id, window_id)
        elif state == LoopState.DORMANT and not seat.converged:
            seat.converged = True
            logger.info("MultiAgentRunner: seat %r converged at window %d", seat.agent_id, window_id)
            self._fire("on_converged", seat.agent_id, window_id)

    def _run_converged(self) -> bool:
        flags = [seat.converged for seat in self._seats.values()]
        if not flags:
            return False
        return all(flags) if self._stop_when == "all" else any(flags)

    def _restore_best_weights(self) -> None:
        for seat in self._seats.values():
            if seat.registry is not None and seat.best_snapshot is not None:
                seat.registry.load_best(seat.agent)
                logger.info(
                    "MultiAgentRunner: seat %r restored to best snapshot (score=%.4f)",
                    seat.agent_id, seat.best_snapshot.eval_metrics.primary_score,
                )

    def _fire(self, event: str, *args) -> None:
        for cb in self._callbacks:
            fn = getattr(cb, event, None)
            if fn is None:
                continue
            try:
                fn(*args)
            except Exception as exc:
                logger.warning("MultiAgentRunner callback %s raised during %s: %s", cb, event, exc)


def _hyperparams_of(agent, window_id: int) -> HyperparamSet:
    getter = getattr(agent, "get_hyperparams", None)
    if getter is not None:
        return getter()
    return HyperparamSet(params={}, episode_id=window_id)


def _safe_dirname(agent_id: Hashable) -> str:
    text = str(agent_id)
    return "".join(c if (c.isalnum() or c in "-_.") else "_" for c in text) or "seat"
