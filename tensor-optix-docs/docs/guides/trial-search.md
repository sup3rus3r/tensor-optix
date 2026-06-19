# Run trial-level search

Online optimizers (SPSA, etc. - see [Tune hyperparameters online](hyperparameter-optimization.md)) adapt hyperparameters *within* one training run. They're good at tracking a non-stationary score landscape but can't efficiently explore the full hyperparameter space from a cold start. `TrialOrchestrator` runs N independent short trials via Optuna's TPE (Tree-structured Parzen Estimator) sampler *before* the main run, then identifies the best configuration.

## Standalone usage

```python
from tensor_optix import TrialOrchestrator

orch = TrialOrchestrator(
    agent_factory=make_agent,
    pipeline_factory=make_pipeline,
    param_space={
        "learning_rate": ("log_float", 1e-4, 3e-3),
        "clip_ratio":    ("float",     0.1,  0.3),
        "batch_size":    ("int",       32,   512),
    },
    n_trials=20,
    trial_steps=50_000,
)
best_params, best_score = orch.run()
```

`param_space` entries are sampling specs:

| Spec | Meaning |
|---|---|
| `("float", lo, hi)` | uniform float |
| `("log_float", lo, hi)` | log-uniform float - use for learning rates |
| `("int", lo, hi)` | uniform int |
| `("log_int", lo, hi)` | log-uniform int |
| `("categorical", v1, v2, ...)` | one of the listed values |

A `MedianPruner` terminates clearly bad trials early (after a warmup phase, any trial whose intermediate score falls below the median of trials at the same step is killed) - this is successive-halving without needing to pre-commit to a fixed budget split.

## Composing with RLOptimizer

`RLOptimizer` can run trial search automatically as a precursor to the main run - pass factories instead of constructed objects:

```python
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
```

`run()` performs trial search, builds the main agent via `agent_factory(best_params)`, **warm-starts it from the best trial's saved weights** (so trial compute isn't wasted on a cold re-init), deletes all trial checkpoint directories, and then runs the full loop with SPSA online adaptation on top of the trial-discovered configuration.

Use `trial_agent_factory` instead of (or in addition to) `agent_factory` if trial-run agent construction needs to differ from the main-run construction (e.g. skip callback registration during trials).

## Reference

Full constructor and method signatures: [Trial search reference](../reference/orchestrator.md).
