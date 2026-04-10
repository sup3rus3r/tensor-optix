import logging
from collections import deque
from typing import Optional, List
import numpy as np
from .base_agent import BaseAgent
from .base_evaluator import BaseEvaluator
from .base_optimizer import BaseOptimizer
from .base_pipeline import BasePipeline
from .checkpoint_registry import CheckpointRegistry
from .backoff_scheduler import BackoffScheduler
from .types import LoopState, EvalMetrics, PolicySnapshot
from .diagnostic_controller import DiagnosticController

logger = logging.getLogger(__name__)


class LoopCallback:
    """
    Hook into the loop at key events.
    Subclass and override any methods you care about.

    Example:
        class MyLogger(LoopCallback):
            def on_improvement(self, snapshot):
                print(f"New best: {snapshot.eval_metrics.primary_score:.4f}")
    """

    def on_loop_start(self) -> None: ...
    def on_loop_stop(self) -> None: ...
    def on_episode_end(self, episode_id: int, eval_metrics: Optional[EvalMetrics]) -> None: ...
    def on_improvement(self, snapshot: PolicySnapshot) -> None: ...
    def on_plateau(self, episode_id: int, state: LoopState) -> None: ...
    def on_dormant(self, episode_id: int) -> None: ...
    def on_degradation(self, episode_id: int, eval_metrics: EvalMetrics) -> None: ...
    def on_hyperparam_update(self, old_params: dict, new_params: dict) -> None: ...


class LoopController:
    """
    The heart of the library.

    Orchestrates the full autonomous improvement loop:
    1. Cold start: run baseline episode, store snapshot
    2. Main loop: interact → evaluate → compare → tune → repeat
    3. State management via BackoffScheduler
    4. Checkpoint management via CheckpointRegistry
    5. Graceful shutdown on stop(), DORMANT, or max_episodes

    Default behaviour (no configuration needed):
    - DORMANT fires → loop stops. Best known weights are restored before
      run() returns. Caller always gets the optimal policy, not the last one.
    - Callbacks may call stop() to halt early (e.g. PolicyManager after
      spawning variants). Core behaviour is independent of callbacks.

    This class has NO knowledge of TensorFlow or any ML framework.
    This class has NO knowledge of PPO, SAC, DQN, or any RL algorithm.
    All interaction is via the abstract interfaces exclusively.
    """

    def __init__(
        self,
        agent: BaseAgent,
        evaluator: BaseEvaluator,
        optimizer: BaseOptimizer,
        pipeline: BasePipeline,
        checkpoint_registry: CheckpointRegistry,
        backoff_scheduler: BackoffScheduler,
        rollback_on_degradation: bool = False,
        improvement_margin: float = 0.0,
        max_episodes: Optional[int] = None,
        callbacks: Optional[List[LoopCallback]] = None,
        val_pipeline: Optional[BasePipeline] = None,
        score_smoothing: int = 2,
        checkpoint_score_fn=None,
        verbose: bool = False,
        verbose_log_file: Optional[str] = None,
        diagnostic_controller: Optional["DiagnosticController"] = None,
    ):
        self._agent = agent
        self._evaluator = evaluator
        self._optimizer = optimizer
        self._pipeline = pipeline
        self._val_pipeline = val_pipeline
        self._registry = checkpoint_registry
        self._scheduler = backoff_scheduler
        self._rollback_on_degradation = rollback_on_degradation
        self._improvement_margin = improvement_margin
        self._max_episodes = max_episodes
        self._callbacks: List[LoopCallback] = callbacks or []
        self._checkpoint_score_fn = checkpoint_score_fn
        self._verbose = verbose
        self._diagnostic = diagnostic_controller or DiagnosticController(verbose=verbose)
        self._log_file = open(verbose_log_file, "w", encoding="utf-8", buffering=1) if verbose_log_file else None  # line-buffered
        self._stop_requested = False
        self._best_snapshot: Optional[PolicySnapshot] = None
        self._metrics_history: List[EvalMetrics] = []
        self._val_gen = None
        self._rnd_eta_base: Optional[float] = None  # set on first state-aware eta call
        # Three separate concerns — kept deliberately independent:
        #
        # 1. Checkpoint saving  → checkpoint_score (raw external eval when
        #    checkpoint_score_fn is provided, otherwise raw primary_score).
        #    The best checkpoint = weights with the highest true policy quality.
        #    External eval (deterministic, fixed seed) is more accurate than
        #    the noisy training window mean, which diverges from real performance.
        #
        # 2. Convergence / degradation detection → smoothed primary_score.
        #    Rolling mean of the last score_smoothing evals. Prevents a single
        #    lucky window from setting an unreachable "best" that permanently
        #    blocks DORMANT. score_smoothing=2 is intentionally light — just
        #    enough to filter single-episode noise without over-smoothing the
        #    signal and delaying convergence detection.
        #
        # 3. checkpoint_score_fn: Optional[Callable[[BaseAgent], float]]
        #    When provided, called after every eval episode to measure true
        #    policy quality. Drives checkpoint saving independently of the
        #    training signal. When None, falls back to raw primary_score.
        self._score_window: deque = deque(maxlen=max(1, score_smoothing))
        self._best_smoothed: Optional[float] = None
        self._best_raw: Optional[float] = None

    def run(self) -> None:
        """
        Start the loop. Blocks until convergence, stop(), or max_episodes.

        When run() returns, the agent always holds the best known weights —
        whether stopped by convergence, budget, or manual stop().
        """
        self._stop_requested = False
        self._pipeline.setup()
        if self._val_pipeline is not None:
            self._val_pipeline.setup()
            self._val_gen = self._val_pipeline.episodes()
            logger.info("Validation pipeline active — primary_score driven by val performance")
        self._fire("on_loop_start")
        try:
            self._cold_start()
            self._main_loop()
        except Exception as e:
            logger.exception("Loop terminated with exception: %s", e)
            raise
        finally:
            # Always restore best known weights before returning.
            # Caller gets the optimal policy regardless of how the loop ended.
            if self._best_snapshot is not None:
                self._registry.load_best(self._agent)
                logger.info(
                    "run() complete — restored best weights (episode %d, score=%.4f)",
                    self._best_snapshot.episode_id,
                    self._best_snapshot.eval_metrics.primary_score,
                )
            self._pipeline.teardown()
            if self._val_pipeline is not None:
                self._val_pipeline.teardown()
            self._fire("on_loop_stop")
            if self._log_file is not None:
                self._log_file.close()
                self._log_file = None

    def stop(self) -> None:
        """Signal the loop to stop cleanly after the current episode."""
        self._stop_requested = True

    def _cold_start(self) -> PolicySnapshot:
        """
        Run episode 0 to establish the baseline.
        No hyperparameter tuning — just run, evaluate, store.
        """
        logger.info("Cold start: running baseline episode")
        episode_gen = self._pipeline.episodes()
        episode_data = next(episode_gen)
        episode_data.episode_id = 0

        train_diagnostics = self._agent.learn(episode_data)
        eval_metrics = self._evaluator.score(episode_data, train_diagnostics)
        eval_metrics.episode_id = 0

        ckpt_score = (
            self._checkpoint_score_fn(self._agent)
            if self._checkpoint_score_fn is not None
            else eval_metrics.primary_score
        )
        hyperparams = self._agent.get_hyperparams()
        snapshot = self._registry.save(self._agent, eval_metrics, hyperparams)
        self._best_snapshot = snapshot
        self._best_raw = ckpt_score
        self._score_window.append(eval_metrics.primary_score)
        self._best_smoothed = float(np.mean(self._score_window))
        self._scheduler.record_improvement(self._best_smoothed)
        self._metrics_history.append(eval_metrics)

        logger.info(
            "Baseline established: score=%.4f smoothed=%.4f state=%s",
            eval_metrics.primary_score,
            self._best_smoothed,
            self._scheduler.current_state.name,
        )
        self._fire("on_episode_end", 0, eval_metrics)
        return snapshot

    def _main_loop(self) -> None:
        """
        Core loop. Runs until stop() called, DORMANT reached, or max_episodes.
        """
        episode_gen = self._pipeline.episodes()
        episode_id = 1

        for episode_data in episode_gen:
            if self._stop_requested:
                logger.info("Stop requested — exiting loop at episode %d", episode_id)
                break
            if self._max_episodes is not None and episode_id >= self._max_episodes:
                logger.info("Max episodes (%d) reached", self._max_episodes)
                break

            episode_data.episode_id = episode_id
            train_diagnostics = self._agent.learn(episode_data)
            self._diagnostic.step(episode_id, self._agent, train_diagnostics or {})

            eval_metrics: Optional[EvalMetrics] = None

            if self._scheduler.should_adapt(episode_id):
                train_metrics = self._evaluator.score(episode_data, train_diagnostics)
                train_metrics.episode_id = episode_id

                if self._val_gen is not None:
                    val_episode = next(self._val_gen)
                    val_episode.episode_id = episode_id
                    val_metrics = self._evaluator.score_validation(val_episode)
                    eval_metrics = self._evaluator.combine(train_metrics, val_metrics)
                    eval_metrics.episode_id = episode_id
                    logger.debug(
                        "Episode %d: train=%.4f val=%.4f gap=%.4f",
                        episode_id,
                        train_metrics.primary_score,
                        val_metrics.primary_score,
                        train_metrics.primary_score - val_metrics.primary_score,
                    )
                else:
                    eval_metrics = train_metrics

                self._metrics_history.append(eval_metrics)

                raw = eval_metrics.primary_score
                self._score_window.append(raw)
                smoothed = float(np.mean(self._score_window))
                # Feed into scheduler's trend window before any is_improving check
                self._scheduler.record_score(raw)

                if self._verbose:
                    self._verbose_header(episode_id, raw, smoothed)

                # ── Checkpoint saving: checkpoint_score ───────────────────────
                # When checkpoint_score_fn is provided, use it to measure true
                # policy quality (e.g. deterministic external eval). This decouples
                # checkpoint selection from the noisy training signal — the training
                # window mean diverges from real performance under fixed-seed
                # deterministic evaluation, causing the wrong weights to be restored.
                # When not provided, falls back to raw primary_score (default).
                ckpt_score = (
                    self._checkpoint_score_fn(self._agent)
                    if self._checkpoint_score_fn is not None
                    else raw
                )
                is_raw_best = (
                    self._best_raw is None
                    or ckpt_score > self._best_raw + self._improvement_margin
                )
                if is_raw_best:
                    hyperparams = self._agent.get_hyperparams()
                    self._best_snapshot = self._registry.save(
                        self._agent, eval_metrics, hyperparams
                    )
                    self._best_raw = ckpt_score
                    self._fire("on_improvement", self._best_snapshot)
                    if self._verbose:
                        self._vprint(f"  CKPT     SAVED  ckpt_score={ckpt_score:.2f}  (new best)")
                    logger.info(
                        "Episode %d: NEW BEST (ckpt) score=%.4f smoothed=%.4f state=%s interval=%d",
                        episode_id, ckpt_score, smoothed,
                        self._scheduler.current_state.name,
                        self._scheduler.current_interval,
                    )
                elif self._verbose:
                    self._vprint(f"  CKPT     skipped  ckpt_score={ckpt_score:.2f} < best={self._best_raw:.2f}")

                # ── Convergence / degradation: trend-based detection ──────────
                # Use linear regression slope over the recent score window to
                # determine direction of learning. A single unlucky eval cannot
                # fire degradation or reset the improvement counter — the slope
                # must be consistently negative for that. Similarly, improvement
                # requires a sustained upward trend, not a single lucky spike.
                #
                # Falls back to point comparison (latest > best + floor) when
                # the window is too short to fit a meaningful line (< 3 scores).
                trend_improving = self._scheduler.is_improving()
                if self._verbose:
                    self._verbose_trend(trend_improving)
                if trend_improving:
                    self._best_smoothed = smoothed
                    self._scheduler.record_improvement(smoothed)
                    self._optimizer.on_improvement(eval_metrics)
                    self._update_rnd_eta("improvement")
                elif not trend_improving:
                    # Degradation detection only applies to on-policy agents.
                    # Off-policy agents (DQN, SAC) use replay buffers — Q-value
                    # oscillations during training are normal and not recoverable
                    # via rollback. Firing degradation on them resets the
                    # scheduler interval to 1 every episode, which prevents
                    # DORMANT from accumulating and destroys convergence.
                    # Rollback is already skipped for off-policy in
                    # _handle_degradation; skipping the check here too ensures
                    # the scheduler state machine runs cleanly for all algorithms.
                    #
                    # Additionally, suppress degradation during optimizer probe
                    # episodes. When the optimizer applies a parameter perturbation
                    # (finite-difference probe), the score drop is self-inflicted —
                    # not a genuine policy collapse. Firing degradation here resets
                    # the scheduler state and triggers unnecessary rollback/recovery.
                    optimizer_is_probing = getattr(self._optimizer, "is_probing", False)
                    agent_is_on_policy = getattr(self._agent, "is_on_policy", True)
                    if agent_is_on_policy and not optimizer_is_probing and self._scheduler.check_degradation(smoothed):
                        self._handle_degradation(episode_id, eval_metrics)
                    elif not agent_is_on_policy:
                        # Off-policy: skip degradation. Score already recorded
                        # above via record_score(raw) — do not record again.
                        self._scheduler.record_non_improvement()
                        state = self._scheduler.current_state
                        if state == LoopState.COOLING:
                            self._update_rnd_eta("plateau")
                            self._fire("on_plateau", episode_id, state)
                        elif state == LoopState.DORMANT:
                            self._update_rnd_eta("dormant")
                            self._optimizer.on_plateau(self._metrics_history)
                            self._fire("on_dormant", episode_id)
                            if not self._stop_requested:
                                logger.info(
                                    "Episode %d: DORMANT — no callback stopped the loop,"
                                    " stopping by default",
                                    episode_id,
                                )
                                self._stop_requested = True
                            self._update_rnd_eta("restart")
                            self._scheduler.record_restart()
                    else:
                        self._scheduler.record_non_improvement()
                        state = self._scheduler.current_state
                        if state == LoopState.COOLING:
                            self._update_rnd_eta("plateau")
                            self._fire("on_plateau", episode_id, state)
                        elif state == LoopState.DORMANT:
                            self._update_rnd_eta("dormant")
                            self._optimizer.on_plateau(self._metrics_history)
                            self._fire("on_dormant", episode_id)
                            if not self._stop_requested:
                                logger.info(
                                    "Episode %d: DORMANT — no callback stopped the loop,"
                                    " stopping by default",
                                    episode_id,
                                )
                                self._stop_requested = True
                            self._update_rnd_eta("restart")
                            self._scheduler.record_restart()
                    logger.debug(
                        "Episode %d: score=%.4f smoothed=%.4f (no improvement) state=%s interval=%d",
                        episode_id, raw, smoothed,
                        self._scheduler.current_state.name,
                        self._scheduler.current_interval,
                    )

                old_params = self._agent.get_hyperparams().params.copy()
                new_hyperparams = self._optimizer.suggest(
                    self._agent.get_hyperparams(),
                    self._metrics_history,
                )
                self._agent.set_hyperparams(new_hyperparams)
                self._fire("on_hyperparam_update", old_params, new_hyperparams.params)
                if self._verbose:
                    self._verbose_spsa(old_params, new_hyperparams.params)

            self._fire("on_episode_end", episode_id, eval_metrics)
            episode_id += 1

    def _handle_degradation(self, episode_id: int, eval_metrics: EvalMetrics) -> None:
        """
        Called when watchdog detects performance drop.

        Rollback is skipped for off-policy agents (agent.is_on_policy == False)
        even when rollback_on_degradation=True. Off-policy agents maintain a
        replay buffer containing experience from all past policies — restoring
        weights without clearing the buffer produces corrupted Bellman targets
        that immediately drag the policy back down. Their buffers naturally
        smooth through degradations without rollback.
        """
        logger.warning(
            "Episode %d: DEGRADATION detected score=%.4f best=%.4f",
            episode_id,
            eval_metrics.primary_score,
            self._scheduler.best_score or 0.0,
        )
        agent_supports_rollback = getattr(self._agent, "is_on_policy", True)
        if (
            self._rollback_on_degradation
            and agent_supports_rollback
            and self._best_snapshot is not None
            and self._scheduler.current_state == LoopState.DORMANT
        ):
            self._registry.load_best(self._agent)
            logger.info("Rolled back to best snapshot (episode %d)", self._best_snapshot.episode_id)
        elif self._rollback_on_degradation and not agent_supports_rollback:
            logger.debug(
                "Episode %d: rollback skipped — off-policy agent, replay buffer intact",
                episode_id,
            )
        self._scheduler.record_degradation()
        self._fire("on_degradation", episode_id, eval_metrics)

    # ------------------------------------------------------------------
    # Verbose diagnostic helpers
    # ------------------------------------------------------------------

    def _vprint(self, *args, **kwargs) -> None:
        """Print to log file when configured, otherwise stdout."""
        kwargs.pop("flush", None)
        if self._log_file is not None:
            print(*args, file=self._log_file)
        else:
            print(*args, flush=True)

    def _verbose_header(self, episode_id: int, raw: float, smoothed: float) -> None:
        """Print score line at start of each eval block."""
        best = self._best_smoothed if self._best_smoothed is not None else float("nan")
        self._vprint(
            f"\n━━ ep={episode_id:4d} ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        )
        self._vprint(
            f"  SCORE    raw={raw:8.2f}  smoothed={smoothed:8.2f}  best={best:8.2f}",
        )

    def _verbose_trend(self, trend_improving: bool) -> None:
        """Print trend analysis and state machine status."""
        slope = self._scheduler._slope()
        floor = self._scheduler._adaptive_floor()
        floor_per_step = floor / self._scheduler._trend_window
        ni  = self._scheduler.consecutive_non_improvements
        pt  = self._scheduler._plateau_threshold
        dt  = self._scheduler._dormant_threshold
        iv  = self._scheduler.current_interval
        state = self._scheduler.current_state.name

        slope_str = f"{slope:+.4f}/ep" if slope is not None else "n/a (window too short)"
        threshold_str = f"{floor_per_step:+.4f}/ep"
        decision = "→ IMPROVING" if trend_improving else "→ not improving"

        self._vprint(
            f"  TREND    slope={slope_str}  threshold={threshold_str}  floor={floor:.3f}  {decision}",
        )
        self._vprint(
            f"  STATE    {state}  non-improvements={ni}/{dt}  plateau@{pt}  interval={iv}",
        )

    def _verbose_spsa(self, old_params: dict, new_params: dict) -> None:
        """Print SPSA phase and what changed."""
        optimizer = self._optimizer
        phase = getattr(optimizer, "_phase", None)
        is_probing = getattr(optimizer, "is_probing", False)
        phase_str = f"phase={phase}" if phase is not None else ""
        probe_str = " [PROBE — degradation suppressed]" if is_probing else ""

        tuned = {k for k in new_params if k in getattr(optimizer, "_param_bounds", {})}
        if not tuned:
            self._vprint(f"  SPSA     no bounded params — optimizer inactive")
            return

        parts = []
        for k in sorted(tuned):
            if k not in old_params:
                continue
            old_v = float(old_params[k])
            new_v = float(new_params[k])
            diff = new_v - old_v
            if abs(diff) > 1e-9 * max(1.0, abs(old_v)):
                pct = diff / abs(old_v) * 100 if abs(old_v) > 1e-12 else 0
                parts.append(f"{k}: {old_v:.5g}→{new_v:.5g} ({pct:+.1f}%)")
            else:
                parts.append(f"{k}: {old_v:.5g} (no change)")

        self._vprint(f"  SPSA     {phase_str}{probe_str}")
        for p in parts:
            self._vprint(f"           {p}")

    def _update_rnd_eta(self, event: str) -> None:
        """
        Adjust RND exploration scale based on loop state transitions.
        Called only when the pipeline supports set_eta (i.e. RNDPipeline is active).
        No-op otherwise — zero cost for users not using RND.

        Schedule:
            improvement → eta *= 0.9   (getting better, pull back exploration)
            plateau     → eta *= 1.5   (stuck, push exploration)
            dormant     → eta = 0      (converged, stop injecting noise)
            restart     → eta = base   (reset after dormant restart)
        """
        pipeline = self._val_pipeline or self._pipeline
        if not hasattr(pipeline, "set_eta"):
            return

        if self._rnd_eta_base is None:
            # Capture initial eta as the base on first call
            self._rnd_eta_base = getattr(pipeline, "_eta", 0.1)

        current = getattr(pipeline, "_eta", self._rnd_eta_base)

        if event == "improvement":
            pipeline.set_eta(current * 0.9)
        elif event == "plateau":
            pipeline.set_eta(min(current * 1.5, self._rnd_eta_base * 4.0))
        elif event == "dormant":
            pipeline.set_eta(0.0)
        elif event == "restart":
            pipeline.set_eta(self._rnd_eta_base)

    def _fire(self, event: str, *args) -> None:
        for cb in self._callbacks:
            try:
                getattr(cb, event)(*args)
            except Exception as e:
                logger.warning("Callback %s raised during %s: %s", cb, event, e)

    @property
    def state(self) -> LoopState:
        return self._scheduler.current_state

    @property
    def best_snapshot(self) -> Optional[PolicySnapshot]:
        return self._best_snapshot
