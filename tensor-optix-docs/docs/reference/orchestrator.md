# tensor_optix.orchestrator

```
TrialOrchestrator - Optuna-based trial-level hyperparameter optimization.

This is a separate layer *above* RLOptimizer. It is NOT an online optimizer.

Online optimizers (SPSA, BackoffOptimizer) adapt hyperparams *within* a single
training run, episode by episode. Trial-level optimization (this module) runs
N independent training trials, each with a different hyperparameter
configuration drawn from a principled surrogate model (TPE). After all
trials, the best configuration is identified and optionally used for a
final full-budget training run.

When to use each:
  - Use TrialOrchestrator to find a good starting configuration before
    committing to a long training run.
  - Use RLOptimizer (with SPSA) for online adaptation during the actual run.
  - They compose naturally: run TrialOrchestrator first, then pass the best
    params to RLOptimizer for the final run.

Algorithm - Optuna TPE (Tree-structured Parzen Estimator):
  TPE models p(x | good) and p(x | bad) as kernel density estimates over
  past trial results, selecting the next configuration by maximising the
  expected improvement ratio. Mathematically equivalent to Bayesian
  optimisation with a non-parametric surrogate, without the O(n³) cost of
  GP inference. Used by Stable-Baselines3, CleanRL, and RLlib for RL HPO.

  Pruner - MedianPruner: after a warmup phase, prunes any trial whose
  intermediate score falls below the median of all trials at the same
  step - successive halving without a fixed budget assumption.

Requires: optuna (optional dependency). pip install optuna
```

## ParamSpace

```python
ParamSpace = Dict[str, Tuple]
```

| Spec | Meaning |
|---|---|
| `("float", lo, hi)` | uniform float in `[lo, hi]` |
| `("log_float", lo, hi)` | log-uniform float (good for learning rate) |
| `("int", lo, hi)` | uniform int in `[lo, hi]` |
| `("log_int", lo, hi)` | log-uniform int |
| `("categorical", v1, v2, ...)` | one of the listed values |

## TrialOrchestrator

```python
class TrialOrchestrator:
    """
    Optuna-based trial-level hyperparameter optimizer for RLOptimizer.

    Each trial is a fully independent RLOptimizer run with a different
    hyperparameter configuration. Optuna's TPE sampler uses results from
    completed trials to choose better configurations for subsequent ones.

    Parameters
    ----------
    agent_factory : Callable[[dict], BaseAgent]
        Called once per trial with the sampled param dict. Must return a
        freshly initialised agent (no shared state between trials).
    pipeline_factory : Callable[[], BasePipeline]
        Called once per trial. Must return a fresh pipeline instance.
    param_space : dict
        Maps param names to sampling specs (see ParamSpace).
    n_trials : int
        Number of independent trials to run.
    trial_steps : int
        Step budget per trial. Common heuristic: 10–20% of final training budget.
    direction : str
        "maximize" (default) or "minimize" for the primary score.
    n_startup_trials : int
        Random trials before TPE's surrogate model kicks in. Default 10.
    pruner_warmup_steps : int
        MedianPruner ignores scores before this many episodes. Default 5.
    pruner_interval : int
        MedianPruner checks every this many episodes. Default 1.
    optuna_verbosity : int
        Optuna log level. Default optuna.logging.WARNING.
    rloptimizer_kwargs : dict
        Extra kwargs forwarded to RLOptimizer for every trial. Do NOT pass
        max_episodes here - use trial_steps instead.
    val_pipeline_factory : Callable[[], BasePipeline], optional
        If provided, called once per trial to create a validation pipeline.
    checkpoint_score_fn : Callable[[BaseAgent], float], optional
        Forwarded to each trial's RLOptimizer.
    study_name, storage : str, optional
        Optuna study persistence (e.g. storage="sqlite:///optuna.db").
        Default: in-memory (no persistence).
    """

    def run(self) -> Tuple[Dict[str, Any], float]:
        """
        Run all trials and return (best_params, best_score).

        Each trial: TPE suggests a config → fresh agent/pipeline built via
        factories → RLOptimizer runs for at most trial_steps env steps →
        best smoothed score reported to Optuna → MedianPruner may stop the
        trial early if the score is poor.

        The best trial's weights are accessible via best_weights_path.
        """

    @property
    def best_weights_path(self) -> Optional[str]:
        """
        Path to the best trial's saved weights directory.
        The caller is responsible for cleaning up run_ckpt_dir once weights
        have been loaded.
        """

    @property
    def run_ckpt_dir(self) -> Optional[str]:
        """Root directory holding all trial checkpoints from the last run()."""

    @property
    def study(self) -> "optuna.Study":
        """The underlying Optuna study. Inspect trial history, plot, etc."""

    @property
    def best_params(self) -> Optional[Dict[str, Any]]: ...
    @property
    def best_score(self) -> Optional[float]: ...

    def trials_dataframe(self):
        """Return a pandas DataFrame of all trial results (requires pandas)."""
```

`_IntermediateReporter` (internal `LoopCallback`) bridges `RLOptimizer`'s `on_episode_end` into Optuna's intermediate reporting - when Optuna's pruner raises `TrialPruned`, it's translated into a `stop()` call on the trial's `RLOptimizer`.

See [Run trial-level search](../guides/trial-search.md) for usage, including composing with `RLOptimizer`'s `agent_factory`/`pipeline_factory`/`param_space` constructor path.
