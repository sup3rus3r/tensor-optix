# Tune hyperparameters online

All online optimizers operate in normalised `[0, 1]` parameter space and update every episode - no restarts required. They implement `BaseOptimizer.suggest()` and see only an opaque `HyperparamSet` plus score history; they never assume specific param names.

## Choosing an optimizer

```python
from tensor_optix.optimizers import SPSAOptimizer, AdaptiveOptimizer

# SPSA: Rademacher perturbation vector, two-episode gradient estimate
optimizer = SPSAOptimizer(
    param_bounds={"learning_rate": (1e-4, 3e-3), "clip_ratio": (0.1, 0.3)},
    log_params={"learning_rate"},   # log-space normalisation for params spanning orders of magnitude
)

# AdaptiveOptimizer: routes between SPSA, Momentum, Backoff, and PBT
# based on lag-1 autocorrelation of the score stream and relative performance gap
optimizer = AdaptiveOptimizer(param_bounds={...})

opt = RLOptimizer(agent=agent, pipeline=pipeline, optimizer=optimizer)
```

| Optimizer | Routing condition |
|---|---|
| `SPSAOptimizer` | i.i.d. score noise, no autocorrelation structure - the default |
| `MomentumOptimizer` | Positive lag-1 autocorrelation (smooth landscape) |
| `BackoffOptimizer` | Negative lag-1 autocorrelation (oscillating landscape, sign-only updates) |
| `PBTOptimizer` | Score below 20th percentile of history (exploit checkpoint population) |
| `AdaptiveOptimizer` | Routes automatically based on the two signals above |

If you don't pass `optimizer=` at all, `RLOptimizer` builds an `SPSAOptimizer` from the agent's `default_param_bounds`/`default_log_params` class attributes automatically.

## Why SPSA by default

SPSA updates all N bounded params in exactly 2 episodes regardless of N (one `+` probe, one `−` probe, then commit), versus `BackoffOptimizer`'s round-robin `2N` episodes. For an agent with 3-5 tunable params that's a 3-5x speedup with an unbiased gradient estimate (Spall, 1992). Use `AdaptiveOptimizer` once you suspect the score landscape isn't i.i.d. - e.g. it's visibly oscillating between hyperparameter updates, or stuck below its own historical best.

## Suppressing false degradation during probes

Every probing optimizer exposes `is_probing` (`True` during a probe episode). `LoopController` checks this before firing degradation handling - a score drop caused by a deliberate finite-difference probe is not a policy collapse, and treating it as one would trigger unnecessary rollback and reset the convergence scheduler.

## Combining with DiagnosticController

Online optimizers are slow by design - a meaningful SPSA update takes several episodes. For acute, single-episode failures (loss spikes, PPO entropy collapse, KL blowups, exhausted DQN epsilon), `RLOptimizer` always wires in a `DiagnosticController` that reacts immediately and independently. Tune its thresholds via the `diag_*` kwargs on `RLOptimizer` if the defaults don't suit your environment. See [DiagnosticController reference](../reference/core/diagnostic_controller.md).

## Reference

Full algorithm details (math, parameters, routing logic) are in the [Optimizers reference](../reference/optimizers.md).
