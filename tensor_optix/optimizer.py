import logging
import os
import shutil
from typing import Optional, List, Callable, Dict, Any, Tuple
from tensor_optix.core.base_agent import BaseAgent
from tensor_optix.core.base_evaluator import BaseEvaluator
from tensor_optix.core.base_optimizer import BaseOptimizer
from tensor_optix.core.base_pipeline import BasePipeline
from tensor_optix.core.loop_controller import LoopController, LoopCallback
from tensor_optix.core.checkpoint_registry import CheckpointRegistry
from tensor_optix.core.backoff_scheduler import BackoffScheduler
from tensor_optix.core.types import LoopState, PolicySnapshot
from tensor_optix.adapters.tensorflow.tf_evaluator import TFEvaluator
from tensor_optix.optimizers.backoff_optimizer import BackoffOptimizer  # noqa: F401 (re-exported)
from tensor_optix.optimizers.spsa_optimizer import SPSAOptimizer        # noqa: F401 (re-exported)
from tensor_optix.core.diagnostic_controller import DiagnosticController

logger = logging.getLogger(__name__)


class RLOptimizer:
    """
    Main public API. This is what users import.

    Assembles all components and delegates to LoopController.
    Sensible defaults for everything — minimal required args: agent, pipeline.

    Minimal usage:
        optimizer = RLOptimizer(agent=my_tf_agent, pipeline=my_pipeline)
        optimizer.run()

    With automatic trial-level hyperparameter search (TrialOrchestrator):
        optimizer = RLOptimizer(
            agent_factory=lambda params: MyAgent(params),
            pipeline_factory=lambda: MyPipeline(),
            param_space={
                "learning_rate": ("log_float", 1e-4, 3e-3),
                "clip_ratio":    ("float", 0.1, 0.3),
            },
            n_trials=20,
            trial_steps_fraction=0.15,   # 15% of max_steps per trial
        )
        optimizer.run()   # trial search runs first, then full training with best params

    When param_space + agent_factory + pipeline_factory are provided:
      1. TrialOrchestrator runs n_trials independent short runs to find best params
      2. agent_factory(best_params) creates the agent for the full run
      3. The main RLOptimizer loop runs with those params + SPSA online adaptation

    The agent and pipeline args become optional when factories are provided.
    """

    def __init__(
        self,
        agent: Optional[BaseAgent] = None,
        pipeline: Optional[BasePipeline] = None,
        evaluator: Optional[BaseEvaluator] = None,
        optimizer: Optional[BaseOptimizer] = None,
        checkpoint_dir: str = "./tensor_optix_checkpoints",
        max_snapshots: int = 10,
        rollback_on_degradation: bool = False,
        improvement_margin: float = 0.0,
        max_episodes: Optional[int] = None,
        base_interval: int = 1,
        backoff_factor: float = 2.0,
        max_interval_episodes: int = 100,
        plateau_threshold: int = 5,
        dormant_threshold: int = 20,
        degradation_threshold: float = 0.95,
        min_degradation_drop: float = 1e-4,
        noise_k: float = 2.0,
        score_window: int = 20,
        trend_window: int = 8,
        min_episodes_before_dormant: int = 0,
        min_episodes_before_degradation: int = 5,
        callbacks: Optional[List[LoopCallback]] = None,
        val_pipeline: Optional[BasePipeline] = None,
        score_smoothing: int = 2,
        checkpoint_score_fn=None,
        verbose: bool = False,
        verbose_log_file: Optional[str] = None,
        # ── DiagnosticController ──────────────────────────────────────────
        diag_loss_spike_factor: float = 5.0,
        diag_entropy_floor: Optional[float] = 0.05,
        diag_target_kl: Optional[float] = 0.02,
        diag_epsilon_patience: int = 20,
        diag_epsilon_reset_value: float = 0.3,
        diag_epsilon_score_threshold: float = 20.0,
        diag_min_episodes: int = 5,
        min_consecutive_degradations: int = 3,
        convergence_patience: int = 5,
        cv_threshold: float = 0.05,
        gap_threshold: float = 0.20,
        target_score: Optional[float] = None,
        # ── Trial-level search (TrialOrchestrator) ────────────────────────
        agent_factory: Optional[Callable[[Dict[str, Any]], BaseAgent]] = None,
        pipeline_factory: Optional[Callable[[], BasePipeline]] = None,
        param_space: Optional[Dict[str, tuple]] = None,
        n_trials: int = 20,
        trial_steps_fraction: float = 0.01,
        val_pipeline_factory: Optional[Callable[[], BasePipeline]] = None,
        trial_agent_factory: Optional[Callable[[Dict[str, Any]], BaseAgent]] = None,
    ):
        if agent is None and agent_factory is None:
            raise ValueError("Either agent or agent_factory must be provided.")
        if pipeline is None and pipeline_factory is None:
            raise ValueError("Either pipeline or pipeline_factory must be provided.")

        # Store trial-search config for use in run()
        self._diag_kwargs = dict(
            loss_spike_factor=diag_loss_spike_factor,
            entropy_floor=diag_entropy_floor,
            target_kl=diag_target_kl,
            epsilon_patience=diag_epsilon_patience,
            epsilon_reset_value=diag_epsilon_reset_value,
            epsilon_score_threshold=diag_epsilon_score_threshold,
            min_episodes=diag_min_episodes,
            verbose=verbose,
        )
        self._agent_factory = agent_factory
        # trial_agent_factory: used only for TrialOrchestrator short runs.
        # When provided, agent_factory is called exactly once (for the main run).
        # Use this to separate trial-only agent construction from main-run setup
        # (e.g. callback registration) that should happen only once.
        self._trial_agent_factory = trial_agent_factory or agent_factory
        self._pipeline_factory = pipeline_factory
        self._val_pipeline_factory = val_pipeline_factory
        self._param_space = param_space
        self._n_trials = n_trials
        self._trial_steps_fraction = trial_steps_fraction
        self._max_episodes = max_episodes
        self._checkpoint_score_fn = checkpoint_score_fn
        self._verbose = verbose

        # Store loop config for deferred controller construction (when using factories)
        self._loop_kwargs = dict(
            evaluator=evaluator,
            optimizer=optimizer,
            min_consecutive_degradations=min_consecutive_degradations,
            convergence_patience=convergence_patience,
            cv_threshold=cv_threshold,
            gap_threshold=gap_threshold,
            target_score=target_score,
            checkpoint_dir=checkpoint_dir,
            max_snapshots=max_snapshots,
            rollback_on_degradation=rollback_on_degradation,
            improvement_margin=improvement_margin,
            max_episodes=max_episodes,
            base_interval=base_interval,
            backoff_factor=backoff_factor,
            max_interval_episodes=max_interval_episodes,
            plateau_threshold=plateau_threshold,
            dormant_threshold=dormant_threshold,
            degradation_threshold=degradation_threshold,
            min_degradation_drop=min_degradation_drop,
            noise_k=noise_k,
            score_window=score_window,
            trend_window=trend_window,
            min_episodes_before_dormant=min_episodes_before_dormant,
            min_episodes_before_degradation=min_episodes_before_degradation,
            callbacks=callbacks,
            val_pipeline=val_pipeline,
            score_smoothing=score_smoothing,
            checkpoint_score_fn=checkpoint_score_fn,
            verbose=verbose,
            verbose_log_file=verbose_log_file,
        )

        # Extra callbacks registered before run() via add_callback()
        self._deferred_callbacks: List[LoopCallback] = []

        if agent_factory is None:
            # Direct mode: agent and pipeline provided — build controller now
            self._controller = self._build_controller(agent, pipeline)
        else:
            # Factory mode: controller built after trial search in run()
            self._controller = None

    def _build_controller(self, agent: BaseAgent, pipeline: BasePipeline) -> LoopController:
        kw = self._loop_kwargs
        evaluator = kw["evaluator"]
        optimizer = kw["optimizer"]

        # Inject agent into pipeline if pipeline supports it
        if hasattr(pipeline, "set_agent"):
            pipeline.set_agent(agent)

        # Off-policy agents (DQN, SAC) need more patience before convergence.
        # Each agent class may declare default_min_episodes_before_dormant to express
        # its own warmup timing (e.g. DQN=60 for epsilon decay, SAC=30 for Q-stability).
        agent_is_on_policy = getattr(agent, "is_on_policy", True)
        dormant_threshold = kw["dormant_threshold"]
        min_episodes_before_dormant = kw["min_episodes_before_dormant"]
        if not agent_is_on_policy:
            if dormant_threshold == 20:
                dormant_threshold = 15
            if min_episodes_before_dormant == 0:
                min_episodes_before_dormant = getattr(
                    agent, "default_min_episodes_before_dormant", 100
                )

        _evaluator = evaluator or TFEvaluator()

        # If no optimizer provided, build SPSAOptimizer with agent's default bounds.
        # If optimizer provided but has no param_bounds, back-fill from agent defaults.
        if optimizer is None:
            agent_bounds = getattr(agent, "default_param_bounds", {})
            agent_log = getattr(agent, "default_log_params", [])
            _optimizer = SPSAOptimizer(param_bounds=agent_bounds, log_params=agent_log)
        else:
            _optimizer = optimizer
            if (
                isinstance(_optimizer, SPSAOptimizer)
                and not _optimizer._param_bounds
            ):
                agent_bounds = getattr(agent, "default_param_bounds", {})
                agent_log = getattr(agent, "default_log_params", [])
                _optimizer._param_bounds = agent_bounds
                _optimizer._log_params = set(agent_log)

        _registry = CheckpointRegistry(
            checkpoint_dir=kw["checkpoint_dir"],
            max_snapshots=kw["max_snapshots"],
        )
        _scheduler = BackoffScheduler(
            base_interval=kw["base_interval"],
            backoff_factor=kw["backoff_factor"],
            max_interval_episodes=kw["max_interval_episodes"],
            plateau_threshold=kw["plateau_threshold"],
            dormant_threshold=dormant_threshold,
            degradation_threshold=kw["degradation_threshold"],
            min_degradation_drop=kw["min_degradation_drop"],
            noise_k=kw["noise_k"],
            score_window=kw["score_window"],
            trend_window=kw["trend_window"],
            min_episodes_before_dormant=min_episodes_before_dormant,
            min_episodes_before_degradation=kw["min_episodes_before_degradation"],
        )

        callbacks = list(kw["callbacks"] or []) + self._deferred_callbacks
        _diagnostic = DiagnosticController(**self._diag_kwargs)
        controller = LoopController(
            agent=agent,
            evaluator=_evaluator,
            optimizer=_optimizer,
            pipeline=pipeline,
            checkpoint_registry=_registry,
            backoff_scheduler=_scheduler,
            rollback_on_degradation=kw["rollback_on_degradation"],
            improvement_margin=kw["improvement_margin"],
            max_episodes=kw["max_episodes"],
            callbacks=callbacks,
            val_pipeline=kw["val_pipeline"],
            score_smoothing=kw["score_smoothing"],
            checkpoint_score_fn=kw["checkpoint_score_fn"],
            verbose=kw["verbose"],
            verbose_log_file=kw.get("verbose_log_file"),
            diagnostic_controller=_diagnostic,
            min_consecutive_degradations=kw["min_consecutive_degradations"],
            convergence_patience=kw["convergence_patience"],
            cv_threshold=kw["cv_threshold"],
            gap_threshold=kw["gap_threshold"],
            target_score=kw["target_score"],
        )
        return controller

    def _run_trial_search(self) -> Tuple[Dict[str, Any], Optional[str], Optional[str]]:
        """
        Run TrialOrchestrator and return (best_params, best_weights_path, trial_base_dir).

        best_weights_path points to the best trial's saved weights on disk.
        trial_base_dir is the root holding all trial checkpoints — the caller
        must shutil.rmtree it after loading the weights into the main agent.
        """
        from tensor_optix.orchestrator import TrialOrchestrator

        trial_steps: int
        if self._max_episodes is not None:
            pipeline_hint = self._pipeline_factory()
            steps_per_ep = getattr(pipeline_hint, "n_steps", None) or getattr(pipeline_hint, "steps_per_episode", None) or 1
            total_steps = self._max_episodes * steps_per_ep
            trial_steps = max(1, int(total_steps * self._trial_steps_fraction)) // steps_per_ep
        else:
            trial_steps = 30

        logger.info(
            "TrialOrchestrator: running %d trials × %d episodes each",
            self._n_trials, trial_steps,
        )
        if self._verbose:
            print(f"  [trial search] running {self._n_trials} trials × {trial_steps} episodes ...", flush=True)

        orch = TrialOrchestrator(
            agent_factory=self._trial_agent_factory,
            pipeline_factory=self._pipeline_factory,
            param_space=self._param_space,
            n_trials=self._n_trials,
            trial_steps=trial_steps,
            val_pipeline_factory=self._val_pipeline_factory,
            checkpoint_score_fn=self._checkpoint_score_fn,
        )
        best_params, best_score = orch.run()

        if self._verbose:
            print(f"  [trial search] complete — best score={best_score:.4f}  params={best_params}", flush=True)
        logger.info("Trial search complete — best score=%.4f params=%s", best_score, best_params)
        return best_params, orch.best_weights_path, orch.run_ckpt_dir

    def run(self) -> None:
        """
        Start the autonomous loop. Blocks until stopped or max_episodes reached.

        If param_space + agent_factory + pipeline_factory were provided:
          1. Runs TrialOrchestrator to find best hyperparameter configuration.
          2. Calls agent_factory(best_params) and pipeline_factory() to build
             the agent and pipeline for the full training run.
          3. Runs the main loop with those params + SPSA online adaptation.
        """
        if self._agent_factory is not None and self._param_space is not None:
            best_params, best_weights_path, trial_base_dir = self._run_trial_search()
            agent = self._agent_factory(best_params)
            # Warm-start: load the best trial's weights into the main agent so
            # trial compute is not wasted starting from random init.
            if best_weights_path and os.path.exists(best_weights_path):
                try:
                    agent.load_weights(best_weights_path)
                    logger.info("Warm-start: loaded trial weights from %s", best_weights_path)
                    if self._verbose:
                        print(f"  [warm-start] loaded best trial weights", flush=True)
                except Exception as e:
                    logger.warning("Warm-start failed (%s) — continuing from random init", e)
            # Clean up all trial checkpoint dirs now that weights are in memory
            if trial_base_dir and os.path.exists(trial_base_dir):
                shutil.rmtree(trial_base_dir, ignore_errors=True)
            pipeline = self._pipeline_factory()
            self._controller = self._build_controller(agent, pipeline)
        elif self._controller is None:
            raise RuntimeError(
                "No controller available. Provide either (agent, pipeline) "
                "or (agent_factory, pipeline_factory, param_space)."
            )
        self._controller.run()

    def add_callback(self, cb: LoopCallback) -> None:
        """
        Register an additional callback. Safe to call before or after run().
        Callbacks added before run() are merged into the controller when it is built.
        """
        if self._controller is not None:
            self._controller._callbacks.append(cb)
        else:
            self._deferred_callbacks.append(cb)

    def stop(self) -> None:
        """Signal graceful shutdown after current episode."""
        self._controller.stop()

    @property
    def state(self) -> LoopState:
        """Current LoopState."""
        return self._controller.state

    @property
    def best_snapshot(self) -> Optional[PolicySnapshot]:
        """Best known PolicySnapshot."""
        return self._controller.best_snapshot
