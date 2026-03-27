# tensor-optix

Autonomous continuous learning loop for TensorFlow RL agents.

> **We own the loop. You own the model.**

tensor-optix wraps your TensorFlow model and Gymnasium environment and takes full ownership of the training loop — stepping continuously, evaluating performance windows, tuning hyperparameters, checkpointing, and adapting over time without manual intervention.

**No fixed episodes.** Training runs as a continuous stream of steps. The loop determines when training ends — not the environment's `done` flag.

---

## Install

```bash
pip install tensor-optix
```

**Requirements:** Python >= 3.11, TensorFlow >= 2.18, Gymnasium >= 1.0

---

## Quick Start

```python
import tensorflow as tf
import gymnasium as gym
from tensor_optix import RLOptimizer, TFAgent, BatchPipeline, HyperparamSet

# Build your model normally
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(4,)),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(2),
])
optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4)

agent = TFAgent(
    model=model,
    optimizer=optimizer,
    hyperparams=HyperparamSet(
        params={"learning_rate": 3e-4, "gamma": 0.99},
        episode_id=0,
    ),
)

# Continuous stepping — windows of 200 steps, no forced resets
env = gym.make("CartPole-v1")
pipeline = BatchPipeline(env=env, agent=agent, window_size=200)

opt = RLOptimizer(agent=agent, pipeline=pipeline)
opt.run()  # runs until DORMANT (plateau) or max_episodes
```

---

## How It Works

tensor-optix runs an autonomous improvement loop with four states:

```
ACTIVE   → aggressive tuning, evaluates every window
COOLING  → recent improvement, exponential backoff on eval frequency
DORMANT  → plateau reached — model is trained, minimal intervention
WATCHDOG → monitoring for degradation
```

**DORMANT = trained.** The backoff determines when the model can no longer improve, not a fixed episode count.

The loop:
1. Steps continuously through the environment in fixed-size windows
2. Evaluates each window via `primary_score`
3. If improved: saves checkpoint, resets backoff
4. If plateau: backs off evaluation, eventually reaches DORMANT
5. If degraded: optionally rolls back to best checkpoint, re-activates
6. Tunes hyperparameters using two-phase finite difference

---

## Optimizer — Two-Phase Finite Difference

`BackoffOptimizer` uses staggered two-phase finite difference per param:

```
Phase 1 (probe):  apply θᵢ + δᵢ, run one window
Phase 2 (commit): gradient = (score_after - score_before) / δᵢ
                  if gradient > 0: keep θᵢ + δᵢ
                  if gradient < 0: apply θᵢ - δᵢ  (reverse)
```

Params are cycled round-robin. Each param is probed and committed independently. Step size adapts on improvement and plateau.

```python
from tensor_optix import BackoffOptimizer

opt = RLOptimizer(
    agent=agent,
    pipeline=pipeline,
    optimizer=BackoffOptimizer(
        param_bounds={
            "learning_rate": (1e-5, 1e-2),
            "gamma": (0.9, 0.999),
        },
        perturbation_scale=0.05,
    ),
)
```

### PBTOptimizer

Pseudo population-based training. Maintains a history of `(hyperparams, score)` pairs. Exploits top performers when in the bottom 20%, explores otherwise.

```python
from tensor_optix import PBTOptimizer

opt = RLOptimizer(
    agent=agent,
    pipeline=pipeline,
    optimizer=PBTOptimizer(
        param_bounds={"learning_rate": (1e-5, 1e-2)},
        history_size=50,
    ),
)
```

---

## Custom Evaluator

```python
from tensor_optix import BaseEvaluator, EpisodeData, EvalMetrics

class TotalRewardEvaluator(BaseEvaluator):
    def score(self, episode_data: EpisodeData, train_diagnostics: dict) -> EvalMetrics:
        total = sum(episode_data.rewards)
        return EvalMetrics(
            primary_score=total,
            metrics={"total_reward": total},
            episode_id=episode_data.episode_id,
        )

opt = RLOptimizer(agent=agent, pipeline=pipeline, evaluator=TotalRewardEvaluator())
```

---

## Custom Agent (Algorithm-Specific Learning)

`TFAgent` provides a REINFORCE baseline. Subclass and override `learn()` for PPO, SAC, DQN, etc.:

```python
from tensor_optix import TFAgent
from tensor_optix.core.types import EpisodeData
import tensorflow as tf

class PPOAgent(TFAgent):
    def learn(self, episode_data: EpisodeData) -> dict:
        clip_ratio = self._hyperparams.params.get("clip_ratio", 0.2)
        # ... PPO update logic ...
        return {"loss": loss_value, "entropy": entropy_value}
```

---

## Live Pipeline

For real-time data sources (trading, robotics, online environments):

```python
from tensor_optix import LivePipeline

class MyFeed:
    def stream(self):
        while True:
            yield obs, reward, terminated, truncated, info

pipeline = LivePipeline(
    data_source=MyFeed(),
    agent=agent,
    episode_boundary_fn=LivePipeline.every_n_seconds(300),
)
```

---

## Callbacks

```python
from tensor_optix import LoopCallback

class MyLogger(LoopCallback):
    def on_improvement(self, snapshot):
        print(f"New best: {snapshot.eval_metrics.primary_score:.4f}")

    def on_dormant(self, window_id):
        print(f"Training complete at window {window_id}")

opt = RLOptimizer(agent=agent, pipeline=pipeline, callbacks=[MyLogger()])
```

Available hooks: `on_loop_start`, `on_loop_stop`, `on_episode_end`, `on_improvement`, `on_plateau`, `on_dormant`, `on_degradation`, `on_hyperparam_update`.

---

## Full Configuration

```python
opt = RLOptimizer(
    agent=agent,
    pipeline=pipeline,
    evaluator=None,                     # default: TFEvaluator
    optimizer=None,                     # default: BackoffOptimizer
    checkpoint_dir="./checkpoints",
    max_snapshots=10,
    rollback_on_degradation=False,
    improvement_margin=0.0,
    max_episodes=None,                  # None = run until DORMANT
    base_interval=1,
    backoff_factor=2.0,
    max_interval_episodes=100,
    plateau_threshold=5,
    dormant_threshold=20,
    degradation_threshold=0.95,
    callbacks=[],
)
```

---

## Architecture

```
tensor_optix/
├── core/
│   ├── types.py                # EpisodeData, EvalMetrics, HyperparamSet, LoopState
│   ├── base_agent.py           # BaseAgent — 6-method contract
│   ├── base_evaluator.py
│   ├── base_optimizer.py
│   ├── base_pipeline.py
│   ├── loop_controller.py      # State machine + main loop
│   ├── checkpoint_registry.py
│   └── backoff_scheduler.py
├── adapters/tensorflow/
│   ├── tf_agent.py             # TFAgent — Keras model wrapper
│   └── tf_evaluator.py         # TFEvaluator — default scorer
├── pipeline/
│   ├── batch_pipeline.py       # Continuous stepping, fixed windows
│   └── live_pipeline.py        # Real-time streaming
└── optimizers/
    ├── backoff_optimizer.py    # Two-phase finite difference
    └── pbt_optimizer.py        # Pseudo population-based training
```

---

## License

MIT — Copyright (c) 2026 sup3rus3r
