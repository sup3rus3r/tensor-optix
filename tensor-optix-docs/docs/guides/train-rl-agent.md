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

## Avoiding unicorn checkpoints

By default, "best" means the single episode with the highest raw
`primary_score`. That's fine most of the time, but it has a failure mode
worth knowing about: a lucky episode during early, high-entropy exploration
- reward concentrated on one favorable step rather than earned consistently
- can score higher than anything that follows, even once the policy is
genuinely, durably better. That lucky episode ("unicorn") then becomes a
permanent, unbeatable "best," because nothing revisits whether it was ever a
reliable measurement in the first place.

Two opt-in `RLOptimizer` parameters address this:

```python
opt = RLOptimizer(
    agent=agent,
    pipeline=pipeline,
    criteria_mode="auto",              # or "manual", or "both"
    checkpoint_confirm_window=3,       # consecutive evals needed to corroborate
)
```

- **`criteria_mode="auto"`** - zero extra code. tensor-optix measures how
  concentrated vs. distributed each episode's reward was (using the shape of
  the reward stream every episode already has), and compares that shape
  against *this run's own history* - so it never penalizes domains where
  every legitimate win is naturally concentrated (e.g. a sparse terminal
  bonus in a landing task).
- **`criteria_mode="manual"`** - for when you know exactly what "good"
  means for your task and it's more than one number. Override `criteria()`
  on your `BaseEvaluator` subclass:

  ```python
  class RobotTaskEvaluator(TFEvaluator):
      def criteria(self, episode_data, train_diagnostics) -> dict:
          info = episode_data.infos[-1]
          return {
              "reached_goal":       1.0 if info.get("success") else 0.0,
              "avoided_collision":  0.0 if info.get("collided") else 1.0,
              "within_energy_budget": min(1.0, info.get("energy_left", 0) / 10.0),
          }
  ```

  A policy that only satisfies one of these three should not out-rank one
  that satisfies all three, even if its raw reward happened to be higher.
- **`criteria_mode="both"`** combines the two (geometric mean).

`checkpoint_confirmation` (implicitly enabled whenever `criteria_mode` isn't
`"none"`) is a separate, complementary layer: it doesn't judge whether an
improvement is narrow or broad (that's what `criteria_mode` is for) - it
just requires enough evidence that an improvement isn't purely this run's
own noise, either because the measured noise is already small or because
`checkpoint_confirm_window` consecutive evals corroborate it.

Both are opt-in and default to the pre-existing behavior
(`criteria_mode="none"`) so upgrading tensor-optix never silently changes
what gets checkpointed in an existing project. See
[BaseEvaluator](../reference/core/base_evaluator.md#measuring-good-as-more-than-one-number)
and [LoopController](../reference/core/loop_controller.md#avoiding-unicorn-checkpoints)
for the full mechanism and config surface.

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
