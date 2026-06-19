# tensor_optix.callbacks

## RichDashboardCallback

```python
class RichDashboardCallback(LoopCallback):
    """
    Live terminal dashboard using the ``rich`` library.

    Architecture: LoopCallback hooks write lightweight event dicts into a
    thread-safe deque and return immediately (~μs overhead). A single daemon
    background thread owns the rich.Live context, reads from the deque,
    updates display state, and re-renders the panel at refresh_per_second Hz.

    Latency guarantee: on_episode_end appends a dict to a collections.deque
    and returns - O(1), lock-free on CPython (GIL-protected). No rendering,
    I/O, or allocation beyond the event dict happens in the hot path.
    Measured overhead is < 0.05 ms per episode on all tested hardware.

    Requires: pip install rich

    Parameters
    ----------
    title : str   - Header label shown in the panel.
    history : int - Number of recent episode scores to display.
    refresh_per_second : int - Redraw rate (default 4 - avoids flicker).
    show_hyperparams : bool - Whether to display the hyperparameter column.
    transient : bool - If True, dashboard is erased when training stops.
                       Default False.
    """

    def __init__(
        self, title="tensor-optix", history=50, refresh_per_second=4,
        show_hyperparams=True, transient=False,
    ): ...
```

## WandbCallback

```python
class WandbCallback(LoopCallback):
    """
    Logs tensor-optix loop events to Weights & Biases.

    Requires: pip install wandb

    Signal groups logged every eval:
        score/primary, score/best, score/at_degradation
        metrics/*              - full EvalMetrics.metrics breakdown
        hyperparams/*          - on improvement and every SPSA update
        spsa/step_magnitude    - L2 norm of relative per-param SPSA changes:
                                  ||Δx / |x_old|||₂  across all tuned params
        events/improvement, events/plateau, events/convergence, events/degradation

    Args:
        project, name, config, tags, group, resume - forwarded to wandb.init().
        **init_kwargs - any additional kwargs forwarded to wandb.init().
    """

    def __init__(
        self, project=None, name=None, config=None, tags=None,
        group=None, resume=None, **init_kwargs,
    ): ...
```

`spsa/step_magnitude` is only logged when the accumulated squared step is positive - it's a measure of SPSA aggression: large during plateau (probe scale widens), small during active improvement (probe scale shrinks).

## TensorBoardCallback

```python
class TensorBoardCallback(LoopCallback):
    """
    Logs tensor-optix loop events to TensorBoard via SummaryWriter.

    Requires: pip install torch  (SummaryWriter ships with torch.utils.tensorboard)
              or: pip install tensor-optix[tensorboard]

    Logs the same signal groups as WandbCallback (score/*, metrics/*,
    hyperparams/*, spsa/step_magnitude, events/*) using TensorBoard's '/'
    scalar-tag grouping convention.

    Args:
        log_dir:      Directory for event files. Default ./runs/tensor_optix.
        comment:      Suffix appended to log_dir when log_dir is not set.
        flush_secs:   How often (seconds) the writer flushes to disk. Default 10.
        **writer_kwargs: forwarded to SummaryWriter().
    """

    def __init__(self, log_dir=None, comment="", flush_secs=10, **writer_kwargs): ...
```

See [Logging and dashboards](../guides/logging-callbacks.md) for usage and writing your own `LoopCallback`.
