# Logging and dashboards

Three callbacks ship out of the box, all subclasses of `LoopCallback`.

## Live terminal dashboard

```python
from tensor_optix.callbacks import RichDashboardCallback

opt.add_callback(RichDashboardCallback(
    title="CartPole PPO",
    history=50,
    refresh_per_second=4,
))
```

Requires `pip install rich`. The loop's hot path is never blocked - `on_episode_end` appends a lightweight event dict to a thread-safe deque and returns in well under a millisecond; a background daemon thread owns the `rich.Live` render loop and redraws at `refresh_per_second` Hz. Pass `transient=True` to erase the dashboard from the terminal when training stops, leaving a clean prompt.

## Weights & Biases

```python
from tensor_optix.callbacks import WandbCallback

opt.add_callback(WandbCallback(project="my-project", name="run-1", tags=["ppo"]))
```

Requires `pip install wandb`. Logs four signal groups every eval: `score/*` (primary, best, at-degradation), `metrics/*` (everything in `EvalMetrics.metrics` - algorithm diagnostics, reward stats, `generalization_gap` when a val pipeline is active), `hyperparams/*` (current values, logged on improvement and on every SPSA update), and `spsa/step_magnitude` (L2 norm of relative per-param SPSA changes - large during plateau exploration, small while actively improving). Plus binary `events/*` flags for improvement, plateau, convergence, and degradation.

## TensorBoard

```python
from tensor_optix.callbacks import TensorBoardCallback

opt.add_callback(TensorBoardCallback(log_dir="./runs/cartpole"))
```

Requires `pip install torch` (ships with `torch.utils.tensorboard`) or `pip install tensor-optix[tensorboard]`. Logs the same signal groups as `WandbCallback`, using TensorBoard's `/`-separated scalar tag convention for grouped panels.

## Writing your own

```python
from tensor_optix import LoopCallback

class MyLogger(LoopCallback):
    def on_improvement(self, snapshot):
        print(f"New best: {snapshot.eval_metrics.primary_score:.4f}")

    def on_dormant(self, episode_id):
        print(f"Converged at episode {episode_id}")
```

Override any subset of: `on_loop_start`, `on_loop_stop`, `on_episode_end(episode_id, eval_metrics)`, `on_improvement(snapshot)`, `on_plateau(episode_id, state)`, `on_dormant(episode_id)`, `on_degradation(episode_id, eval_metrics)`, `on_hyperparam_update(old_params, new_params)`. A callback that raises inside any hook is caught and logged as a warning by `LoopController` - it will not crash the training run.

`HebbianHook` and `NeuromodulatorSignal` (neuroevo subsystem) are also `LoopCallback` subclasses and can be passed alongside the above via `callbacks=`.

## Reference

Full constructor parameters: [Callbacks reference](../reference/callbacks.md).
