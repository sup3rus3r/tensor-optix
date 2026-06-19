# Ensembles and policy evolution

`PolicyManager` runs as a loop callback and is responsible for *model* evolution - separate from the optimizer's job of tuning hyperparameters. On each `DORMANT` event, `MetaController` evaluates the generalization gap (level and slope) and the validation improvement rate, then issues one of `SPAWN`, `PRUNE`, or `STOP`. Spawned variants are cloned from the best checkpoint with perturbed hyperparameters and (optionally) perturbed weights. `EnsembleAgent` wraps all active variants behind the `BaseAgent` interface, combining their actions via weighted averaging.

## Minimal: rollback only

```python
from tensor_optix import PolicyManager

pm = PolicyManager(registry)
cb = pm.as_callback(agent)
opt = RLOptimizer(..., callbacks=[cb])
```

With no `agent_factory`, `DORMANT` rolls back to the best checkpoint and otherwise leaves the loop's own convergence/stopping logic in control.

## Spawning variants

```python
pm = PolicyManager(registry, max_spawns=4)
cb = pm.as_callback(agent, agent_factory=make_agent)
cb.set_stop_fn(opt.stop)
opt.add_callback(cb)
opt.run()
```

`set_stop_fn` is required when spawning is enabled - when the spawn budget is exhausted, `PolicyManagerCallback` calls it to terminate the loop cleanly with the best-known weights already restored.

## Letting MetaController decide

Without a `MetaController`, `PolicyManagerCallback` spawns whenever budget allows. With one, the spawn/prune/stop decision is signal-driven:

```python
from tensor_optix import MetaController

mc = MetaController(gap_threshold=0.2, gap_slope_threshold=0.02, improvement_threshold=0.01)
cb = pm.as_callback(agent, agent_factory=make_agent, meta_controller=mc)
cb.set_stop_fn(opt.stop)
```

`MetaController` checks, in order: spawn budget exhausted → `STOP`; generalization gap level too high → `PRUNE`; gap actively widening (even if level is still tolerable) → `PRUNE`; validation improvement too flat → `SPAWN`; otherwise → `NO_OP`. This requires a `val_pipeline` for the gap signals to be meaningful.

## Building an ensemble manually

```python
from tensor_optix import EnsembleAgent

pm = PolicyManager(registry)
pm.add_agent(agent_a, weight=1.0)
pm.add_agent(agent_b, weight=0.5)
ensemble = EnsembleAgent(pm, primary_agent=agent_a)
pipeline.set_agent(ensemble)
opt = RLOptimizer(agent=ensemble, pipeline=pipeline, ...)
```

`EnsembleAgent.learn()` trains **every** registered agent on the same episode data - this is mathematically necessary; non-primary agents that don't update will drift away from the primary's action distribution as training progresses, degrading the ensemble. `get_hyperparams`/`set_hyperparams`/`save_weights`/`load_weights` all delegate to the primary agent only - it's the authoritative agent for checkpointing.

## Regime-conditional boosting

Outside the spawn/prune pipeline, you can shift ensemble weight toward a specific agent based on detected regime:

```python
from tensor_optix.core import RegimeDetector

detector = RegimeDetector()
regime = detector.detect(metrics_history)
pm.set_regime(regime)
if regime == "volatile":
    pm.boost(agent_volatile, factor=2.0)
```

## Inspecting state

```python
import json
print(json.dumps(pm.status(), indent=2))
# after training:
print(json.dumps(pm.training_report(), indent=2))
```

`PolicyManagerCallback` also prints a human-readable training report to stdout automatically when the loop stops via budget exhaustion or a `MetaController` `STOP` decision.

## Reference

Full method signatures: [PolicyManager](../reference/core/policy_manager.md), [MetaController](../reference/core/meta_controller.md), [EnsembleAgent](../reference/core/ensemble_agent.md).
