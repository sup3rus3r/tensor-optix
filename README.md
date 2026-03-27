# tensor-optix

Self-evolving autonomous learning loop for TensorFlow RL agents.

---

## About

tensor-optix replaces the conventional reinforcement learning training loop with an autonomous, continuously-learning optimization system. You bring your TensorFlow model and Gymnasium environment — the library owns everything else: stepping, evaluation, hyperparameter tuning, checkpointing, and policy evolution.

The system runs as a continuous stream of steps with no fixed episode count. It detects performance plateaus through exponential backoff, tunes hyperparameters using finite difference estimation, and evolves policies by comparing live performance against its checkpoint history. Multiple agents can run simultaneously as a weighted ensemble — essential for non-stationary environments like financial markets where no single policy dominates all regimes.

**Core philosophy:** We own the loop. You own the model.

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
5. If DORMANT: `PolicyManager` compares current score vs registry best and rolls back if needed
6. If degraded: optionally rolls back to best checkpoint, re-activates
7. Tunes hyperparameters using two-phase finite difference

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

## Policy Evolution

`PolicyManager` handles model evolution — separate from the hyperparameter optimizer.

**Separation of concerns:**
- `BackoffOptimizer` / `PBTOptimizer` → tune hyperparameters
- `PolicyManager` → evolve models (rollback, ensemble)

### Automatic rollback on DORMANT

When the loop reaches DORMANT, `PolicyManager` compares the current score against the best checkpoint. If current < best, it loads the best known weights back into the agent.

```python
from tensor_optix import PolicyManager, RLOptimizer
from tensor_optix.core.checkpoint_registry import CheckpointRegistry

registry = CheckpointRegistry("./checkpoints")
pm = PolicyManager(registry)

opt = RLOptimizer(
    agent=agent,
    pipeline=pipeline,
    checkpoint_dir="./checkpoints",
    callbacks=[pm.as_callback(agent)],
)
opt.run()
```

### Ensemble — multiple policies

Run multiple agents simultaneously. Actions are combined as a weighted average.

```python
from tensor_optix import PolicyManager, EnsembleAgent

pm = PolicyManager(registry)
pm.add_agent(agent_trending,  weight=1.0)   # strong in trending markets
pm.add_agent(agent_ranging,   weight=1.0)   # strong in sideways markets
pm.add_agent(agent_volatile,  weight=1.0)   # strong in high-volatility markets

ensemble = EnsembleAgent(pm, primary_agent=agent_trending)

opt = RLOptimizer(
    agent=ensemble,
    pipeline=BatchPipeline(env=env, agent=ensemble, window_size=200),
    callbacks=[pm.as_callback(agent_trending)],
)
opt.run()
```

**Weight updates** — adjust ensemble weights based on recent regime performance:

```python
# After evaluating each agent separately:
pm.update_weights({0: sharpe_trending, 1: sharpe_ranging, 2: sharpe_volatile})
```

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
│   ├── backoff_scheduler.py
│   ├── policy_manager.py       # PolicyManager + PolicyManagerCallback
│   └── ensemble_agent.py       # EnsembleAgent — multi-policy BaseAgent wrapper
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

### Component responsibilities

| Component | Responsibility |
|-----------|---------------|
| `LoopController` | State machine, episode orchestration |
| `BackoffScheduler` | Adaptation interval + state transitions |
| `CheckpointRegistry` | Snapshot storage and manifest |
| `BaseOptimizer` | Hyperparameter tuning |
| `PolicyManager` | Model evolution (rollback, ensemble weights) |
| `EnsembleAgent` | Multi-policy action combining |

---

## License

MIT — Copyright (c) 2026 sup3rus3r
