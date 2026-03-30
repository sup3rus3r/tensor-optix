import logging
from typing import Optional, List
from .base_agent import BaseAgent
from .base_evaluator import BaseEvaluator
from .base_optimizer import BaseOptimizer
from .base_pipeline import BasePipeline
from .checkpoint_registry import CheckpointRegistry
from .backoff_scheduler import BackoffScheduler
from .types import LoopState, EvalMetrics, PolicySnapshot

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
    5. Graceful shutdown on stop() or exception

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
        self._stop_requested = False
        self._best_snapshot: Optional[PolicySnapshot] = None
        self._metrics_history: List[EvalMetrics] = []
        self._val_gen = None

    def run(self) -> None:
        """
        Start the loop. Blocks until stop() is called or max_episodes reached.
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
            self._pipeline.teardown()
            if self._val_pipeline is not None:
                self._val_pipeline.teardown()
            self._fire("on_loop_stop")

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

        hyperparams = self._agent.get_hyperparams()
        snapshot = self._registry.save(self._agent, eval_metrics, hyperparams)
        self._best_snapshot = snapshot
        self._scheduler.record_improvement(eval_metrics.primary_score)
        self._metrics_history.append(eval_metrics)

        logger.info(
            "Baseline established: score=%.4f state=%s",
            eval_metrics.primary_score,
            self._scheduler.current_state.name,
        )
        self._fire("on_episode_end", 0, eval_metrics)
        return snapshot

    def _main_loop(self) -> None:
        """
        Core loop. Runs until stop() called or max_episodes reached.
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

                is_improvement = (
                    self._best_snapshot is None
                    or self._evaluator.compare(
                        eval_metrics,
                        self._best_snapshot.eval_metrics,
                    )
                    and eval_metrics.beats(
                        self._best_snapshot.eval_metrics, self._improvement_margin
                    )
                )

                if is_improvement:
                    hyperparams = self._agent.get_hyperparams()
                    self._best_snapshot = self._registry.save(
                        self._agent, eval_metrics, hyperparams
                    )
                    self._scheduler.record_improvement(eval_metrics.primary_score)
                    self._optimizer.on_improvement(eval_metrics)
                    self._fire("on_improvement", self._best_snapshot)
                    logger.info(
                        "Episode %d: NEW BEST score=%.4f state=%s interval=%d",
                        episode_id,
                        eval_metrics.primary_score,
                        self._scheduler.current_state.name,
                        self._scheduler.current_interval,
                    )
                else:
                    if self._scheduler.check_degradation(eval_metrics.primary_score):
                        self._handle_degradation(episode_id, eval_metrics)
                    else:
                        self._scheduler.record_non_improvement()
                        state = self._scheduler.current_state
                        if state == LoopState.COOLING:
                            self._fire("on_plateau", episode_id, state)
                        elif state == LoopState.DORMANT:
                            self._optimizer.on_plateau(self._metrics_history)
                            self._fire("on_dormant", episode_id)
                    logger.debug(
                        "Episode %d: score=%.4f (no improvement) state=%s interval=%d",
                        episode_id,
                        eval_metrics.primary_score,
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

            self._fire("on_episode_end", episode_id, eval_metrics)
            episode_id += 1

    def _handle_degradation(self, episode_id: int, eval_metrics: EvalMetrics) -> None:
        """
        Called when watchdog detects performance drop.
        Optionally rolls back to best known weights. Always resets to ACTIVE.
        """
        logger.warning(
            "Episode %d: DEGRADATION detected score=%.4f best=%.4f",
            episode_id,
            eval_metrics.primary_score,
            self._scheduler.best_score or 0.0,
        )
        if (
            self._rollback_on_degradation
            and self._best_snapshot is not None
            and self._scheduler.current_state == LoopState.DORMANT
        ):
            self._registry.load_best(self._agent)
            logger.info("Rolled back to best snapshot (episode %d)", self._best_snapshot.episode_id)
        self._scheduler.record_degradation()
        self._fire("on_degradation", episode_id, eval_metrics)

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
