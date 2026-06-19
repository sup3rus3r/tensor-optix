# Train an RL agent

This guide covers the full `RLOptimizer` path - validation, checkpointing, and callbacks - beyond what [Quickstart](../getting-started/quickstart.md) shows with the simplified `Optimizer` wrapper.

## Pipelines

A pipeline steps an environment (or data source), collects `EpisodeData`, and yields it to the agent.

```python
from tensor_optix import BatchPipeline, LivePipeline, VectorBatchPipeline
import gymnasium as gym

# Gymnasium env: steps continuously, no reset between windows
pipeline = BatchPipeline(env=gym.make("CartPole-v1"), agent=agent, window_size=200)

# External data stream: background thread with bounded queue, configurable episode boundaries
pipeline = LivePipeline(
    data_source=MyFeed(),
    agent=agent,
    episode_boundary_fn=LivePipeline.every_n_seconds(300),
)

# N parallel envs via gymnasium.vector, sync or async subprocess
pipeline = VectorBatchPipeline(
    env_fns=[lambda: gym.make("CartPole-v1")] * 8,
    agent=agent,
    window_size=200,
)
```

`BatchPipeline` does **not** reset the environment per window - it steps continuously and resets automatically only on `terminated`/`truncated`. `window_size` is the unit of training, not an environment episode. See [Pipelines](../reference/pipelines.md) for the full reference, including the `gym.Env` method-name collision warning (don't name an env attribute `close`, `step`, `reset`, `render`, or `seed`).

`optimal_window_size(env, algorithm)` computes the formula `Optimizer` uses internally: `clip(k * mean_episode_steps, 512, 8192)`, with `k=4.0` for on-policy (PPO) and `k=1.0` for off-policy (SAC/TD3/DQN).

```python
from tensor_optix import optimal_window_size
window = optimal_window_size(env, "PPO")  # e.g. 2000 for CartPole
```

## The full loop

```python
from tensor_optix import RLOptimizer

opt = RLOptimizer(
    agent=agent,
    pipeline=pipeline,

    # Separate validation pipeline. All checkpoint and rollback decisions use val score only.
    val_pipeline=val_pipeline,
    rollback_on_degradation=True,

    # Optional external scorer run at checkpoint evaluation (e.g. held-out backtest)
    checkpoint_score_fn=lambda a: evaluate(a, held_out_env),

    # Convergence parameters
    dormant_threshold=10,            # consecutive non-improving evals -> DORMANT
    min_episodes_before_dormant=50,  # statistical warmup before convergence detection activates
)

opt.run()
opt.best_snapshot   # -> PolicySnapshot: best weights + EvalMetrics + HyperparamSet
```

Loop states: `ACTIVE → COOLING → DORMANT → watchdog shutdown or policy spawn`. On shutdown the loop restores best-known weights, not the final checkpoint. See [Concepts](../getting-started/concepts.md) for why, and [Loop controller reference](../reference/core/loop_controller.md) for the full constructor and degradation-handling details.

## Validation pipelines

When `val_pipeline` is set, `primary_score` becomes the validation score and `EvalMetrics.generalization_gap` (train − val) becomes available. The validation pipeline's `act()` calls populate the agent's on-policy rollout cache; `LoopController` calls `agent.reset_cache()` (if the agent defines it) immediately after scoring validation, so that data is never accidentally consumed by the next training `learn()` call.

## Callbacks

```python
from tensor_optix.callbacks import RichDashboardCallback, WandbCallback, TensorBoardCallback

opt.add_callback(RichDashboardCallback())        # Rich live terminal panel
opt.add_callback(WandbCallback(project="run"))
opt.add_callback(TensorBoardCallback(log_dir="./tb"))
```

Custom callbacks subclass `LoopCallback` and override any of `on_loop_start`, `on_loop_stop`, `on_episode_end`, `on_improvement`, `on_plateau`, `on_dormant`, `on_degradation`, `on_hyperparam_update`. See [Logging and dashboards](logging-callbacks.md) and the [Callbacks reference](../reference/callbacks.md).

## Verbose mode

`verbose=True` (optionally with `verbose_log_file=...`) prints a per-eval breakdown: raw/smoothed/best score, trend slope vs. adaptive floor, loop state, and any hyperparameter changes from the active optimizer - useful when tuning convergence thresholds for a new environment.
