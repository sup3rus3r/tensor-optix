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

Each seat's hyperparameters are tuned online by its own optimizer (SPSA by
default, like RLOptimizer), and `self_play=SelfPlayConfig(...)` lets seats
train against past snapshots of their opponents.

`rollback_on_degradation=True` restores a seat's best checkpoint mid-run
when its score has confirmedly dropped.

Not done here (single-agent LoopController feature with no obvious
multi-agent meaning yet): policy spawning.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Hashable, List, Optional

from tensor_optix.core.backoff_scheduler import BackoffScheduler
from tensor_optix.core.base_optimizer import BaseOptimizer
from tensor_optix.core.checkpoint_registry import CheckpointRegistry
from tensor_optix.core.types import EvalMetrics, HyperparamSet, LoopState, PolicySnapshot

from tensor_optix.multi_agent.self_play import (
    OpponentPool,
    SelfPlayConfig,
    discard_rollout,
    make_stand_in,
)

logger = logging.getLogger(__name__)

_METRICS_HISTORY_LEN = 200


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
    def on_hyperparam_update(self, agent_id: Hashable, old_params: dict, new_params: dict) -> None: ...
    def on_degradation(self, agent_id: Hashable, window_id: int, score: float, rolled_back: bool) -> None: ...
    def on_run_stop(self, reason: str) -> None: ...


@dataclass
class SeatResult:
    """Per-seat outcome of a run."""
    agent_id: Hashable
    state: LoopState
    converged: bool
    windows_scored: int
    completed_episodes: int
    best_score: Optional[float]
    last_score: Optional[float]
    best_snapshot: Optional[PolicySnapshot]
    frozen_windows: int = 0        # windows this seat was played by a pool snapshot
    degradations: int = 0          # confirmed degradations
    rollbacks: int = 0             # of which restored the best checkpoint
    hyperparams: Optional[dict] = None   # the seat's hyperparameters at the end of the run


@dataclass
class MultiAgentRunResult:
    """What run() returns."""
    windows_run: int
    stop_reason: str            # "converged" | "max_windows" | "stopped" | "pipeline_exhausted"
    seats: Dict[Hashable, SeatResult] = field(default_factory=dict)
    pool_sizes: Dict[Hashable, int] = field(default_factory=dict)   # self-play snapshots per seat

    @property
    def converged(self) -> bool:
        return self.stop_reason == "converged"


class _Seat:
    def __init__(self, agent_id, agent, scheduler: BackoffScheduler,
                 registry: Optional[CheckpointRegistry], optimizer: Optional[BaseOptimizer] = None):
        self.agent_id = agent_id
        self.agent = agent
        self.scheduler = scheduler
        self.registry = registry
        self.optimizer = optimizer
        self.metrics_history: List[EvalMetrics] = []
        self.frozen_windows = 0
        self.consecutive_degraded = 0
        self.degradations = 0
        self.rollbacks = 0
        self.best_score: Optional[float] = None
        self.last_score: Optional[float] = None
        self.best_snapshot: Optional[PolicySnapshot] = None
        self.windows_scored = 0
        self.converged = False
        self.completed_episodes = 0
        self.reports_episodes = False
        self.scored_before_first_episode = False


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

    Hyperparameter tuning: after every scored window the seat's optimizer
    is shown that seat's score history and may adjust its hyperparameters
    (`on_hyperparam_update` fires when it does). Windows that are not
    scored do not step the optimizer.

    Self-play pool (`self_play=SelfPlayConfig(...)`): every
    `snapshot_every` windows each seat's weights are saved to that seat's
    pool. Each window, a seat is played by a random past snapshot of itself
    with probability `opponent_prob` — it then neither learns nor is scored
    for that window — while the remaining live seats train against it. At
    least one seat is always live. Live agents are put back before the run
    returns.

    Rollback (`rollback_on_degradation=True`): when a seat's score stays
    clearly below its best for `min_consecutive_degradations` scored
    windows in a row, `on_degradation` fires and — for on-policy agents with
    a checkpoint — the seat's best weights are restored and training
    continues. Off-policy agents are never rolled back (their replay buffer
    would pull restored weights straight back). A score also drops when the
    OPPONENT improves, which is not a collapse: with self-play enabled only
    windows played against at least one frozen (pool) opponent count, since
    those are the windows where the opposition did not change. Without
    self-play, use rollback for cooperative games; in a competitive game it
    would undo real learning. Probe windows of the seat's optimizer never
    count.

    Scores are full episode returns, tracked across windows by the
    pipeline. A window in which a seat finished no episode still trains
    that seat but is not scored (no checkpoint, no plateau count). Until a
    seat's first episode finishes, windows are scored by mean per-step
    reward; when it does, that seat's best score and trend are reset
    because the scale changes.

    Args:
        pipeline, agents:      as before.
        evaluator:             a BaseEvaluator, or {agent_id: BaseEvaluator}.
        checkpoint_dir:        root directory; each seat checkpoints into
                               <checkpoint_dir>/<agent_id>. None = no
                               checkpointing (and nothing to restore).
        max_snapshots:         snapshots kept per seat.
        improvement_margin:    minimum score gain over the seat's best for
                               a new checkpoint.
        min_completed_episodes_before_converged:
                               a seat cannot plateau/converge before it has
                               finished this many episodes (default 10).
        plateau_threshold / dormant_threshold / min_windows_before_dormant /
        noise_k / score_window / trend_window:
                               per-seat BackoffScheduler settings (same
                               meaning as RLOptimizer's). Each seat is
                               evaluated every window (no eval backoff).
        optimizer:             per-seat online hyperparameter tuning.
                               "auto" (default): an SPSAOptimizer over the
                               agent's default_param_bounds, for every seat
                               whose agent declares them. None: no tuning.
                               A callable(agent) -> BaseOptimizer builds one
                               per seat; a dict {agent_id: BaseOptimizer or
                               None} assigns them explicitly. Optimizers are
                               stateful, so a single instance cannot be
                               shared between seats.
        self_play:             SelfPlayConfig enabling the frozen-opponent
                               pool (see below). None (default) = off.
        rollback_on_degradation:
                               restore a seat's best checkpoint mid-run on a
                               confirmed degradation (see below). Requires
                               checkpoint_dir. Default False.
        degradation_threshold: a window is "degraded" when the seat's score
                               is below best - max(|best| * (1 - threshold),
                               the seat's noise floor). Default 0.95.
        min_consecutive_degradations:
                               degraded windows in a row needed to confirm.
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
        min_completed_episodes_before_converged: int = 10,
        noise_k: float = 2.0,
        score_window: int = 20,
        trend_window: int = 8,
        optimizer: Any = "auto",
        self_play: Optional[SelfPlayConfig] = None,
        rollback_on_degradation: bool = False,
        degradation_threshold: float = 0.95,
        min_consecutive_degradations: int = 3,
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
        self._min_completed_episodes = max(0, int(min_completed_episodes_before_converged))
        self._stop_when = stop_when
        self._stop_on_convergence = stop_on_convergence
        self._restore_best = restore_best
        self._callbacks: List[Any] = list(callbacks or [])
        self._stop_requested = False

        if rollback_on_degradation and checkpoint_dir is None:
            raise ValueError("rollback_on_degradation=True requires checkpoint_dir (there is nothing to roll back to).")
        self._rollback_on_degradation = rollback_on_degradation
        self._degradation_threshold = float(degradation_threshold)
        self._min_consecutive_degradations = max(1, int(min_consecutive_degradations))

        self._self_play = self_play
        if self_play is not None and not hasattr(pipeline, "set_agents"):
            raise TypeError(
                "self_play requires a pipeline exposing set_agents() "
                "(ParallelMultiAgentPipeline / AECMultiAgentPipeline)."
            )
        self._pool: Optional[OpponentPool] = None
        self._stand_ins: Dict[Hashable, Any] = {}
        self._frozen: set = set()

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
            self._seats[aid] = _Seat(
                aid, agent, scheduler, registry,
                optimizer=_resolve_optimizer(optimizer, aid, agent, len(agents)),
            )

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
        if self._self_play is not None:
            self._pool = OpponentPool(
                self._self_play.pool_size, self._self_play.pool_dir, self._self_play.seed
            )
            self._frozen = set()
        try:
            for window in self._pipeline.episodes():
                window_metrics: Dict[Hashable, EvalMetrics] = {}
                for aid, episode_data in window.items():
                    if len(episode_data.rewards) == 0:
                        continue
                    seat = self._seats.get(aid)
                    if seat is None:
                        continue
                    if aid in self._frozen:
                        # Played by a pool snapshot this window: the live
                        # agent did not act, so it has nothing to learn from
                        # and no score of its own.
                        seat.frozen_windows += 1
                        discard_rollout(self._stand_ins[aid])
                        continue
                    diagnostics = seat.agent.learn(episode_data) or {}
                    metrics = self._evaluators[aid].score(episode_data, diagnostics)
                    metrics.episode_id = windows_run
                    if not self._account_episodes(seat, metrics):
                        # No episode finished for this seat in this window:
                        # it trained, but there is no return to score.
                        continue
                    window_metrics[aid] = metrics
                    self._update_seat(seat, metrics, windows_run)

                self._fire("on_window_end", windows_run, window_metrics)
                windows_run += 1
                if self._pool is not None:
                    self._advance_self_play(windows_run)

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
            pool_sizes: Dict[Hashable, int] = {}
            if self._pool is not None:
                pool_sizes = self._pool.sizes()
                # Hand the pipeline its live agents back before anything else.
                self._pipeline.set_agents(dict(self._agents))
                self._frozen = set()
                self._pool.cleanup()
                self._pool = None
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
                    completed_episodes=seat.completed_episodes,
                    best_score=seat.best_score,
                    last_score=seat.last_score,
                    best_snapshot=seat.best_snapshot,
                    frozen_windows=seat.frozen_windows,
                    degradations=seat.degradations,
                    rollbacks=seat.rollbacks,
                    hyperparams=dict(_hyperparams_of(seat.agent, windows_run).params),
                )
                for aid, seat in self._seats.items()
            },
            pool_sizes=pool_sizes,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _account_episodes(self, seat: _Seat, metrics: EvalMetrics) -> bool:
        """
        Same rule as LoopController._account_episodes, per seat: a window in
        which no episode finished is not an evaluation (once the seat has
        finished at least one episode), and the first finished episode after
        per-step-scored windows resets the seat's best score and trend.
        Evaluators that do not report "episodes_completed" are always used.
        """
        completed = (metrics.metrics or {}).get("episodes_completed")
        if completed is None:
            return True
        completed = int(completed)
        seat.reports_episodes = True
        if completed == 0:
            if seat.completed_episodes > 0:
                return False
            seat.scored_before_first_episode = True
            return True
        if seat.completed_episodes == 0 and seat.scored_before_first_episode:
            seat.best_score = None
            seat.scheduler.reset_scale()
        seat.completed_episodes += completed
        return True

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

        seat.metrics_history.append(metrics)
        if len(seat.metrics_history) > _METRICS_HISTORY_LEN:
            del seat.metrics_history[:-_METRICS_HISTORY_LEN]

        self._update_convergence(seat, metrics, score, window_id)
        self._tune(seat)

    def _update_convergence(self, seat: _Seat, metrics: EvalMetrics, score: float, window_id: int) -> None:
        # --- convergence: trend over the seat's recent scores ---
        sched = seat.scheduler
        first = sched.best_score is None
        sched.record_score(score)
        if first or sched.is_improving():
            sched.record_improvement(score, append=False)
            seat.converged = False
            seat.consecutive_degraded = 0
            if seat.optimizer is not None:
                seat.optimizer.on_improvement(metrics)
            return

        if self._check_degradation(seat, score, window_id):
            return  # a degraded window is a collapse in progress, not a plateau

        if seat.reports_episodes and seat.completed_episodes < self._min_completed_episodes:
            return  # too few finished episodes to call anything a plateau

        prev_state = sched.current_state
        sched.record_non_improvement()
        state = sched.current_state
        if state == LoopState.COOLING and prev_state != LoopState.COOLING:
            self._fire("on_plateau", seat.agent_id, window_id)
        elif state == LoopState.DORMANT and not seat.converged:
            seat.converged = True
            logger.info("MultiAgentRunner: seat %r converged at window %d", seat.agent_id, window_id)
            if seat.optimizer is not None:
                seat.optimizer.on_plateau(seat.metrics_history)
            self._fire("on_converged", seat.agent_id, window_id)

    def _check_degradation(self, seat: _Seat, score: float, window_id: int) -> bool:
        """Count degraded windows; on confirmation fire on_degradation and
        (when enabled and safe) roll the seat back.

        Returns True when this window is degraded (confirmed or not): such a
        window must not count toward the plateau counter, or a collapsed seat
        would be declared converged at its collapsed level between rollbacks.
        """
        if not self._rollback_on_degradation or seat.best_score is None:
            return False
        if seat.optimizer is not None and getattr(seat.optimizer, "is_probing", False):
            return False  # a probe's score drop is self-inflicted
        if self._pool is not None and not (self._frozen - {seat.agent_id}):
            # Self-play is on and every opponent was live this window: a drop
            # may simply be the opponent improving. Neither count nor reset.
            return False

        best = seat.best_score
        allowed_drop = max(
            abs(best) * (1.0 - self._degradation_threshold),
            seat.scheduler._adaptive_floor(),
        )
        if score >= best - allowed_drop:
            seat.consecutive_degraded = 0
            return False

        seat.consecutive_degraded += 1
        if seat.consecutive_degraded < self._min_consecutive_degradations:
            return True

        seat.consecutive_degraded = 0
        seat.degradations += 1
        rolled_back = False
        if (
            getattr(seat.agent, "is_on_policy", True)
            and seat.registry is not None
            and seat.best_snapshot is not None
        ):
            seat.registry.load_best(seat.agent)
            discard_rollout(seat.agent)
            seat.rollbacks += 1
            rolled_back = True
            logger.info(
                "MultiAgentRunner: seat %r degraded (score=%.4f, best=%.4f) — rolled back at window %d",
                seat.agent_id, score, best, window_id,
            )
        seat.scheduler.record_degradation()
        seat.scheduler._consecutive_non_improvements = 0
        seat.converged = False
        self._fire("on_degradation", seat.agent_id, window_id, score, rolled_back)
        return True

    def _tune(self, seat: _Seat) -> None:
        """One online hyperparameter step for this seat (after a scored window)."""
        if seat.optimizer is None:
            return
        current = seat.agent.get_hyperparams()
        old_params = dict(current.params)
        suggested = seat.optimizer.suggest(current, seat.metrics_history)
        seat.agent.set_hyperparams(suggested)
        if suggested.params != old_params:
            self._fire("on_hyperparam_update", seat.agent_id, old_params, dict(suggested.params))

    def _advance_self_play(self, windows_run: int) -> None:
        """Snapshot on schedule, then choose who is frozen for the next window."""
        cfg = self._self_play
        if windows_run % cfg.snapshot_every == 0:
            for aid, seat in self._seats.items():
                self._pool.add(aid, seat.agent)

        rng = self._pool.rng
        frozen = {
            aid for aid in self._seats
            if self._pool.size(aid) > 0 and rng.random() < cfg.opponent_prob
        }
        if len(frozen) >= len(self._seats):
            # Somebody has to be learning.
            frozen.discard(rng.choice(sorted(frozen, key=str)))

        mapping = dict(self._agents)
        for aid in frozen:
            stand_in = self._stand_ins.get(aid)
            if stand_in is None:
                stand_in = make_stand_in(self._agents[aid], aid, cfg.opponent_factory)
                self._stand_ins[aid] = stand_in
            discard_rollout(stand_in)
            self._pool.load_into(aid, stand_in, self._agents[aid])
            mapping[aid] = stand_in
        self._frozen = frozen
        self._pipeline.set_agents(mapping)

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


def _resolve_optimizer(spec: Any, agent_id: Hashable, agent: Any, n_seats: int) -> Optional[BaseOptimizer]:
    """Turn the runner's `optimizer` argument into this seat's own optimizer."""
    if spec is None:
        return None
    if isinstance(spec, dict):
        return spec.get(agent_id)
    if isinstance(spec, str):
        if spec != "auto":
            raise ValueError(f"optimizer must be 'auto', None, a callable, or a dict; got {spec!r}")
        bounds = getattr(agent, "default_param_bounds", None)
        if not bounds or not hasattr(agent, "get_hyperparams"):
            return None
        from tensor_optix.optimizers.spsa_optimizer import SPSAOptimizer
        return SPSAOptimizer(
            param_bounds=dict(bounds),
            log_params=list(getattr(agent, "default_log_params", []) or []),
        )
    if isinstance(spec, BaseOptimizer):
        if n_seats > 1:
            raise ValueError(
                "A single optimizer instance cannot be shared between seats (it "
                "holds per-run state). Pass a callable(agent) -> BaseOptimizer or "
                "a dict {agent_id: optimizer}."
            )
        return spec
    if callable(spec):
        return spec(agent)
    raise ValueError(f"Unsupported optimizer argument: {spec!r}")


def _safe_dirname(agent_id: Hashable) -> str:
    text = str(agent_id)
    return "".join(c if (c.isalnum() or c in "-_.") else "_" for c in text) or "seat"
