# tensor_optix.core.meta_controller

## MetaAction

```python
class MetaAction(Enum):
    """Actions the MetaController can instruct the system to take on DORMANT."""
    NO_OP = auto()   # system is healthy - do nothing
    SPAWN = auto()   # exploration needed - clone and mutate a new variant
    PRUNE = auto()   # ensemble too large or most overfit agent should be removed
    STOP  = auto()   # spawn budget exhausted or convergence confirmed - halt
```

## MetaController

```python
class MetaController:
    """
    Decides what the system should do when it reaches DORMANT state.

    Observes three signals derived from EvalMetrics history and pm.status():

    1. Generalization gap level: mean (train - val) / |val|
       Large gap → model is overfitting → PRUNE

    2. Generalization gap slope: is the gap widening?
       A model where train improves but val stagnates is actively overfitting
       even if the current gap is below the level threshold. Slope of the
       normalized gap series catches this early.

       This replaces the former Pearson correlation signal. Pearson measured
       whether train and val move *together in shape*, not whether they
       diverge in *level*. A perfectly correlated (r=1.0) pair like
       train=[0.9,0.91,0.92] vs val=[0.3,0.31,0.32] is catastrophically
       overfit. Gap slope is the correct overfitting-progression signal.

    3. Normalized val slope (improvement rate)
       Flat or declining val performance → genuine plateau → SPAWN

    Priority: gap level → gap slope → improvement rate. If the budget
    is exhausted, STOP is returned regardless of other signals.

    This is a rule-based controller. It implements no learning of its own.
    The interface is intentionally minimal so it can be swapped for a learned
    policy (e.g. an RL agent whose observation is pm.status() + metrics
    features and whose action space is MetaAction) without any API change.

    Parameters:
        gap_threshold:       normalized gap above which PRUNE fires (default 0.2)
        gap_slope_threshold: normalized gap slope above which PRUNE fires (default 0.02)
                             i.e. gap is widening by >2% of |val| per episode
        improvement_threshold: normalized val slope below which SPAWN fires (default 0.01)
        window:              number of recent EvalMetrics to consider (default 10)
    """

    def __init__(
        self,
        gap_threshold: float = 0.2,
        gap_slope_threshold: float = 0.02,
        improvement_threshold: float = 0.01,
        window: int = 10,
    ): ...

    def decide(self, metrics_history: List[EvalMetrics], pm_status: dict) -> MetaAction:
        """
        Return a MetaAction based on current system state.

        metrics_history: full history of EvalMetrics from the loop
        pm_status: output of PolicyManager.status()
        """
```

### Decision order

1. `pm_status["budget_exhausted"]` → `STOP`, unconditionally.
2. Fewer than 3 metrics recorded → `NO_OP`.
3. Normalized generalization gap > `gap_threshold` → `PRUNE` (overfitting level).
4. Generalization gap slope > `gap_slope_threshold` → `PRUNE` (gap actively widening, even if level is still tolerable).
5. Normalized val-score slope < `improvement_threshold` → `SPAWN` (genuine plateau).
6. Otherwise → `NO_OP`.

Wire a `MetaController` into `PolicyManager.as_callback(..., meta_controller=...)` to replace the default "spawn whenever budget allows" behavior with this signal-driven decision tree.
