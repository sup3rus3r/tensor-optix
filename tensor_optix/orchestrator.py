"""
TrialOrchestrator — Optuna-based trial-level hyperparameter optimization.

This is a separate layer *above* RLOptimizer. It is NOT an online optimizer.

Online optimizers (SPSA, BackoffOptimizer) adapt hyperparams *within* a single
training run, episode by episode. They are good at tracking non-stationarity
but cannot explore the full hyperparam space efficiently.

Trial-level optimization (this module) runs N independent training trials, each
with a different hyperparameter configuration drawn from a principled surrogate
model (TPE). After all trials, the best configuration is identified and
optionally used for a final full-budget training run.

When to use each:
  - Use TrialOrchestrator to find a good starting configuration before
    committing to a long training run.
  - Use RLOptimizer (with SPSA) for online adaptation during the actual run.
  - They compose naturally: run TrialOrchestrator first, then pass the best
    params to RLOptimizer for the final run.

Algorithm — Optuna TPE (Tree-structured Parzen Estimator):
  TPE models p(x | good) and p(x | bad) as kernel density estimates over
  past trial results. It selects the next configuration by maximising the
  expected improvement ratio EI = p(x | good) / p(x | bad). This is
  mathematically equivalent to Bayesian optimisation with a non-parametric
  surrogate, without the O(n³) cost of GP inference. It is the algorithm
  used by Stable-Baselines3, CleanRL, and RLlib for RL HPO sweeps.

  Pruner — MedianPruner:
    After a warmup phase, prune any trial whose intermediate score falls
    below the median of all trials at the same step. This cuts wall time
    by terminating clearly bad configs early, equivalent to successive
    halving without a fixed budget assumption.

Usage::

    from tensor_optix.orchestrator import TrialOrchestrator

    def make_agent(params: dict) -> BaseAgent:
        net = tf.keras.Sequential([...])
        return TFPPOAgent(
            actor=net, critic=...,
            optimizer=tf.keras.optimizers.Adam(params["learning_rate"]),
            hyperparams=HyperparamSet(params=params, episode_id=0),
        )

    def make_pipeline() -> BasePipeline:
        return GymPipeline(env_id="LunarLander-v3", n_steps=2048)

    orchestrator = TrialOrchestrator(
        agent_factory=make_agent,
        pipeline_factory=make_pipeline,
        param_space={
            "learning_rate": ("log_float", 1e-4, 3e-3),
            "clip_ratio":    ("float",     0.1,  0.3),
            "entropy_coef":  ("float",     0.0,  0.05),
        },
        n_trials=20,
        trial_steps=50_000,
        direction="maximize",
    )
    best_params, best_score = orchestrator.run()
    print(f"Best params: {best_params}  score: {best_score:.4f}")

    # Now run final full-budget training with best params
    agent = make_agent(best_params)
    pipeline = make_pipeline()
    optimizer = RLOptimizer(agent=agent, pipeline=pipeline, max_episodes=500)
    optimizer.run()

Parameter space specification:
    Each entry in param_space is a tuple describing how to sample the param:

    ("float",      lo, hi)           — uniform float in [lo, hi]
    ("log_float",  lo, hi)           — log-uniform float (good for lr)
    ("int",        lo, hi)           — uniform int in [lo, hi]
    ("log_int",    lo, hi)           — log-uniform int
    ("categorical", val1, val2, ...) — one of the listed values

Requires: optuna (optional dependency).
    Install with:  pip install optuna  or  uv add optuna
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from typing import Any, Callable, Dict, List, Optional, Tuple

from tensor_optix.core.base_agent import BaseAgent
from tensor_optix.core.base_pipeline import BasePipeline
from tensor_optix.optimizer import RLOptimizer
from tensor_optix.core.loop_controller import LoopCallback
from tensor_optix.core.types import EvalMetrics

logger = logging.getLogger(__name__)

# Optuna is an optional dependency
try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
    _OPTUNA_AVAILABLE = True
except ImportError:
    _OPTUNA_AVAILABLE = False


ParamSpace = Dict[str, Tuple]
"""
Dict mapping param name → sampling spec tuple.
See module docstring for supported specs.
"""


def _sample_params(trial: "optuna.Trial", param_space: ParamSpace) -> Dict[str, Any]:
    """
    Convert a param_space spec dict into a concrete dict of values
    by querying the Optuna trial's suggest_* methods.
    """
    params: Dict[str, Any] = {}
    for name, spec in param_space.items():
        kind = spec[0]
        if kind == "float":
            _, lo, hi = spec
            params[name] = trial.suggest_float(name, lo, hi)
        elif kind == "log_float":
            _, lo, hi = spec
            params[name] = trial.suggest_float(name, lo, hi, log=True)
        elif kind == "int":
            _, lo, hi = spec
            params[name] = trial.suggest_int(name, lo, hi)
        elif kind == "log_int":
            _, lo, hi = spec
            params[name] = trial.suggest_int(name, lo, hi, log=True)
        elif kind == "categorical":
            choices = list(spec[1:])
            params[name] = trial.suggest_categorical(name, choices)
        else:
            raise ValueError(
                f"Unknown param_space kind '{kind}' for param '{name}'. "
                "Expected: float, log_float, int, log_int, categorical."
            )
    return params


class _IntermediateReporter(LoopCallback):
    """
    Bridges RLOptimizer's LoopCallback into Optuna's intermediate reporting.

    Reports the smoothed score to Optuna after each eval episode, enabling
    MedianPruner to terminate poor trials early. If Optuna raises
    TrialPruned, we translate that into a stop() on the RLOptimizer.
    """

    def __init__(self, trial: "optuna.Trial", optimizer_ref: "RLOptimizer"):
        self._trial = trial
        self._optimizer_ref = optimizer_ref

    def on_episode_end(self, episode_id: int, eval_metrics: Optional[EvalMetrics]) -> None:
        if eval_metrics is None:
            return  # eval not scheduled this episode
        score = eval_metrics.primary_score
        self._trial.report(score, step=episode_id)
        if self._trial.should_prune():
            logger.info(
                "Trial %d pruned at episode %d (score=%.4f)",
                self._trial.number, episode_id, score,
            )
            self._optimizer_ref.stop()


class TrialOrchestrator:
    """
    Optuna-based trial-level hyperparameter optimizer for RLOptimizer.

    Each trial is a fully independent RLOptimizer run with a different
    hyperparameter configuration. Optuna's TPE sampler uses results from
    completed trials to choose better configurations for subsequent ones.

    Parameters
    ----------
    agent_factory : Callable[[dict], BaseAgent]
        Called once per trial with the sampled param dict.
        Must return a freshly initialised agent (no shared state between trials).
    pipeline_factory : Callable[[], BasePipeline]
        Called once per trial. Must return a fresh pipeline instance.
    param_space : dict
        Maps param names to sampling specs. See module docstring.
    n_trials : int
        Number of independent trials to run.
    trial_steps : int
        Step budget per trial. Short enough to rank configs quickly,
        long enough that the signal is meaningful. A common heuristic:
        10–20% of the final training budget.
    direction : str
        "maximize" (default) or "minimize" for the primary score.
    n_startup_trials : int
        Number of random trials before TPE's surrogate model kicks in.
        Default 10 — matches Optuna's recommended minimum.
    pruner_warmup_steps : int
        MedianPruner ignores scores before this many episodes.
        Prevents pruning before the agent has had time to learn anything.
        Default 5.
    pruner_interval : int
        MedianPruner checks every this many episodes. Default 1.
    optuna_verbosity : int
        Optuna log level (optuna.logging.WARNING by default to keep
        output clean). Set to optuna.logging.INFO for more detail.
    rloptimizer_kwargs : dict
        Extra kwargs forwarded to RLOptimizer for every trial
        (e.g. checkpoint_dir, rollback_on_degradation, verbose).
        Do NOT pass max_episodes here — use trial_steps instead.
    val_pipeline_factory : Callable[[], BasePipeline], optional
        If provided, called once per trial to create a validation pipeline.
    checkpoint_score_fn : Callable[[BaseAgent], float], optional
        Forwarded to each trial's RLOptimizer.
    study_name : str, optional
        Optuna study name. Auto-generated if not set.
    storage : str, optional
        Optuna storage URL (e.g. "sqlite:///optuna.db") for distributed
        or persistent studies. Default: in-memory (no persistence).
    """

    def __init__(
        self,
        agent_factory: Callable[[Dict[str, Any]], BaseAgent],
        pipeline_factory: Callable[[], BasePipeline],
        param_space: Optional[ParamSpace] = None,
        n_trials: int = 20,
        trial_steps: int = 50_000,
        direction: str = "maximize",
        n_startup_trials: int = 10,
        pruner_warmup_steps: int = 5,
        pruner_interval: int = 1,
        optuna_verbosity: Optional[int] = None,
        rloptimizer_kwargs: Optional[Dict[str, Any]] = None,
        val_pipeline_factory: Optional[Callable[[], BasePipeline]] = None,
        checkpoint_score_fn: Optional[Callable[[BaseAgent], float]] = None,
        study_name: Optional[str] = None,
        storage: Optional[str] = None,
    ):
        if not _OPTUNA_AVAILABLE:
            raise ImportError(
                "Optuna is required for TrialOrchestrator.\n"
                "Install it with:  pip install optuna\n"
                "or (uv):          uv add optuna"
            )
        self._agent_factory = agent_factory
        self._pipeline_factory = pipeline_factory

        # Auto-derive param_space from agent's default_param_bounds if not provided.
        if param_space is None:
            try:
                probe_agent = agent_factory({})
                raw_bounds = getattr(probe_agent, "default_param_bounds", {})
                log_ps = set(getattr(probe_agent, "default_log_params", []))
                param_space = {
                    k: ("log_float" if k in log_ps else "float", lo, hi)
                    for k, (lo, hi) in raw_bounds.items()
                }
            except Exception:
                param_space = {}
        self._param_space = param_space
        self._n_trials = n_trials
        self._trial_steps = trial_steps
        self._direction = direction
        self._n_startup_trials = n_startup_trials
        self._pruner_warmup_steps = pruner_warmup_steps
        self._pruner_interval = pruner_interval
        self._rloptimizer_kwargs = rloptimizer_kwargs or {}
        self._val_pipeline_factory = val_pipeline_factory
        self._checkpoint_score_fn = checkpoint_score_fn
        self._study_name = study_name or f"tensor_optix_study"
        self._storage = storage

        # Set Optuna verbosity
        _verbosity = optuna_verbosity if optuna_verbosity is not None else optuna.logging.WARNING
        optuna.logging.set_verbosity(_verbosity)

        # Per-run state (populated in run())
        self._run_ckpt_dir: Optional[str] = None
        self._trial_weights: Dict[int, Optional[str]] = {}
        self._best_weights_path: Optional[str] = None

        # Build study once so results persist across run() calls
        self._study = optuna.create_study(
            study_name=self._study_name,
            direction=self._direction,
            storage=self._storage,
            sampler=TPESampler(n_startup_trials=self._n_startup_trials),
            pruner=MedianPruner(
                n_startup_trials=self._n_startup_trials,
                n_warmup_steps=self._pruner_warmup_steps,
                interval_steps=self._pruner_interval,
            ),
            load_if_exists=(self._storage is not None),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> Tuple[Dict[str, Any], float]:
        """
        Run all trials and return (best_params, best_score).

        Each trial:
          1. TPE suggests a hyperparameter configuration.
          2. A fresh agent and pipeline are created via the factories.
          3. RLOptimizer runs for at most trial_steps environment steps.
          4. The best smoothed score from that run is reported to Optuna.
          5. MedianPruner may stop the trial early if the score is poor.

        After all trials, the configuration that achieved the highest
        (or lowest, if direction="minimize") score is returned.
        The best trial's weights are accessible via best_weights_path.

        Returns
        -------
        best_params : dict
            The hyperparameter configuration that achieved the best score.
        best_score : float
            The best score achieved by that configuration.
        """
        # Persistent dir that survives all trials so we can warm-start from
        # the best trial's weights after the search completes.
        self._run_ckpt_dir = tempfile.mkdtemp(prefix="to_trials_")
        self._trial_weights = {}
        self._best_weights_path = None

        self._study.optimize(
            self._objective,
            n_trials=self._n_trials,
            catch=(Exception,),
        )

        best = self._study.best_trial
        self._best_weights_path = self._trial_weights.get(best.number)
        logger.info(
            "TrialOrchestrator complete — best trial #%d score=%.4f params=%s weights=%s",
            best.number, best.value, best.params, self._best_weights_path,
        )
        return best.params, best.value

    @property
    def best_weights_path(self) -> Optional[str]:
        """
        Path to the best trial's saved weights directory.
        Valid only after run() has been called. The caller is responsible
        for cleaning up self.run_ckpt_dir once weights have been loaded.
        """
        return self._best_weights_path

    @property
    def run_ckpt_dir(self) -> Optional[str]:
        """Root directory holding all trial checkpoints from the last run()."""
        return self._run_ckpt_dir

    @property
    def study(self) -> "optuna.Study":
        """The underlying Optuna study. Inspect trial history, plot, etc."""
        return self._study

    @property
    def best_params(self) -> Optional[Dict[str, Any]]:
        """Best params found so far, or None if no trial has finished."""
        try:
            return self._study.best_params
        except ValueError:
            return None

    @property
    def best_score(self) -> Optional[float]:
        """Best score found so far, or None if no trial has finished."""
        try:
            return self._study.best_value
        except ValueError:
            return None

    def trials_dataframe(self):
        """Return a pandas DataFrame of all trial results (requires pandas)."""
        return self._study.trials_dataframe()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _objective(self, trial: "optuna.Trial") -> float:
        """
        One Optuna trial = one independent RLOptimizer run.

        Steps are bounded by trial_steps. We convert steps → max_episodes
        by asking the fresh pipeline for its episode length hint. If the
        pipeline does not expose step_count, we fall back to 1 episode = 1
        unit (user should pass trial_steps as episode count in that case).
        """
        params = _sample_params(trial, self._param_space)
        logger.info("Trial %d starting — params: %s", trial.number, params)

        agent = self._agent_factory(params)
        pipeline = self._pipeline_factory()
        val_pipeline = self._val_pipeline_factory() if self._val_pipeline_factory else None

        # Estimate max_episodes from trial_steps and pipeline step hint
        steps_per_episode = getattr(pipeline, "n_steps", None) or getattr(pipeline, "steps_per_episode", None)
        if steps_per_episode is not None and steps_per_episode > 0:
            max_episodes = max(1, self._trial_steps // steps_per_episode)
        else:
            # No step hint — interpret trial_steps as episode count directly
            max_episodes = self._trial_steps

        # Persistent per-trial dir inside the run-level dir so the best
        # trial's weights survive until the caller has loaded them.
        ckpt_dir = os.path.join(self._run_ckpt_dir, f"trial_{trial.number}")
        os.makedirs(ckpt_dir, exist_ok=True)

        rl = RLOptimizer(
            agent=agent,
            pipeline=pipeline,
            max_episodes=max_episodes,
            checkpoint_dir=ckpt_dir,
            val_pipeline=val_pipeline,
            checkpoint_score_fn=self._checkpoint_score_fn,
            **self._rloptimizer_kwargs,
        )

        # Attach the intermediate reporter (enables pruning)
        reporter = _IntermediateReporter(trial, rl)
        rl._controller._callbacks.append(reporter)

        try:
            rl.run()
        except Exception as exc:
            logger.warning("Trial %d raised exception: %s", trial.number, exc)
            raise

        best = rl.best_snapshot
        if best is None:
            return float("-inf") if self._direction == "maximize" else float("inf")

        # Record weights path so run() can hand it to the main agent
        self._trial_weights[trial.number] = best.weights_path

        score = best.eval_metrics.primary_score
        logger.info("Trial %d finished — score=%.4f params=%s", trial.number, score, params)
        return score
