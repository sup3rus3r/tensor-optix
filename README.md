# tensor-optix

Self-evolving autonomous reinforcement learning loop , algorithm-agnostic, framework-optional.

---

## About

tensor-optix replaces the conventional RL training loop with an autonomous system that owns evaluation, checkpointing, hyperparameter tuning, policy evolution, and ensemble management. You bring your agent and environment. The library does everything else.

**The system never stops at a fixed episode count.** It detects convergence through exponential backoff, spawns policy variants when it plateaus, weights an ensemble by rolling performance, and uses both training and validation signals to drive every decision , not training alone.

**Core philosophy:** We own the loop. You own the model.

---

## Install

```bash
pip install tensor-optix
```

**Requirements:** Python >= 3.11, Gymnasium >= 1.0
TensorFlow >= 2.18 is required only when using `TFAgent` or `TFEvaluator`. The core loop, PolicyManager, and all ensemble/evolution logic are framework-free.

---

## Quick Start

```python
import tensorflow as tf
import gymnasium as gym
from tensor_optix import RLOptimizer, TFAgent, BatchPipeline, HyperparamSet

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(4,)),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(2),
])

agent = TFAgent(
    model=model,
    optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
    hyperparams=HyperparamSet(params={"learning_rate": 3e-4, "gamma": 0.99}, episode_id=0),
)

env = gym.make("CartPole-v1")
pipeline = BatchPipeline(env=env, agent=agent, window_size=200)

opt = RLOptimizer(agent=agent, pipeline=pipeline)
opt.run()  # runs until convergence (DORMANT state)
```

---

## Algorithm Support

The core loop calls exactly **six methods** on any agent. Nothing else is assumed , no network architecture, no action space shape, no gradient-based learning, no framework.

```python
class BaseAgent(ABC):
    def act(self, observation) -> any: ...         # any action type
    def learn(self, episode_data) -> dict: ...     # any algorithm
    def get_hyperparams(self) -> HyperparamSet: ...
    def set_hyperparams(self, hp: HyperparamSet): ...
    def save_weights(self, path: str): ...
    def load_weights(self, path: str): ...
```

This is the only coupling point between your algorithm and the framework.

### Using PPO

```python
from tensor_optix import TFAgent

class PPOAgent(TFAgent):
    def act(self, observation):
        obs = tf.expand_dims(tf.cast(observation, tf.float32), 0)
        logits, _ = self.model(obs, training=False)
        return int(tf.random.categorical(logits, 1).numpy()[0, 0])

    def learn(self, episode_data):
        # PPO clip update, advantage estimation, entropy bonus
        # ...
        return {"loss": loss, "entropy": entropy, "kl": kl}

    def set_hyperparams(self, hp):
        super().set_hyperparams(hp)
        self._clip_ratio = hp.params.get("clip_ratio", 0.2)
        self._entropy_coeff = hp.params.get("entropy_coeff", 0.01)
```

```python
agent = PPOAgent(
    model=actor_critic_model,
    optimizer=tf.keras.optimizers.Adam(3e-4),
    hyperparams=HyperparamSet(
        params={"learning_rate": 3e-4, "clip_ratio": 0.2, "entropy_coeff": 0.01, "gamma": 0.99},
        episode_id=0,
    ),
)
```

### Using DQN

```python
from tensor_optix.core.base_agent import BaseAgent

class DQNAgent(BaseAgent):
    def act(self, observation):
        if np.random.random() < self._epsilon:
            return self._env.action_space.sample()
        q_values = self.q_network(tf.expand_dims(observation, 0), training=False)
        return int(tf.argmax(q_values, axis=-1).numpy()[0])

    def learn(self, episode_data):
        # Add to replay buffer, sample batch, Bellman update
        # ...
        return {"td_loss": loss, "q_mean": q_mean}

    def set_hyperparams(self, hp):
        self._hyperparams = hp.copy()
        self._epsilon = hp.params.get("epsilon", 0.1)
        if "learning_rate" in hp.params:
            self.optimizer.learning_rate.assign(hp.params["learning_rate"])
```

### Using SAC / TD3 / DDPG

Same pattern , implement `BaseAgent`, override `act()` for continuous action sampling and `learn()` for your update rule. Hyperparams are an open dict; no key names are hardcoded anywhere in the framework.

### Using PyTorch or JAX

```python
import torch

class TorchPPOAgent(BaseAgent):
    def act(self, observation):
        obs = torch.FloatTensor(observation).unsqueeze(0)
        with torch.no_grad():
            logits = self.policy(obs)
        return torch.distributions.Categorical(logits=logits).sample().item()

    def learn(self, episode_data):
        # Standard PyTorch training loop
        return {"loss": loss.item()}

    def save_weights(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.policy.state_dict(), os.path.join(path, "policy.pt"))

    def load_weights(self, path):
        self.policy.load_state_dict(torch.load(os.path.join(path, "policy.pt")))
```

The loop, hyperparameter optimizer, checkpointing, and ensemble logic all work identically. No TensorFlow required.

---

## How It Works

### The Loop States

```
ACTIVE   → aggressive tuning, evaluates every window
COOLING  → recent improvement, exponential backoff on eval frequency
DORMANT  → plateau reached , model is trained, minimal intervention
WATCHDOG → monitoring for degradation
```

**DORMANT = trained.** Not a fixed episode count , the system backs off evaluation geometrically until improvement stops, then declares convergence.

### Backoff Schedule

```
interval₀ = base_interval
intervalₙ = min(intervalₙ₋₁ × backoff_factor, max_interval_episodes)

Plateau detected when:  consecutive_no_improvement ≥ plateau_threshold
DORMANT declared when:  consecutive_no_improvement ≥ dormant_threshold
```

Every improvement resets the backoff counter. The system accelerates evaluation when learning is happening, backs off when it isn't.

### Hyperparameter Optimizer , Two-Phase Finite Difference

`BackoffOptimizer` cycles through hyperparameters using staggered two-phase finite difference:

```
For each param θᵢ:
  Phase 1 (probe):   apply θᵢ + δᵢ, run one window, record score s₊
  Phase 2 (commit):  gradient ĝᵢ = (s₊ − s₀) / δᵢ
                     if ĝᵢ > 0: keep θᵢ + δᵢ
                     if ĝᵢ ≤ 0: apply θᵢ − δᵢ  (reverse direction)
```

Step size `δᵢ` adapts: shrinks on improvement, grows on plateau. Params cycle round-robin , each is probed and committed independently.

`PBTOptimizer` maintains a history of `(hyperparams, score)` pairs and exploits top performers when in the bottom 20%, otherwise explores with Gaussian perturbation.

---

## The Science: Train + Val Together

Without validation, every decision , checkpoint saves, rollbacks, spawn triggers , is made on training data alone. That is overfitting disguised as improvement.

### Validation Pipeline

```python
opt = RLOptimizer(
    agent=agent,
    pipeline=train_pipeline,
    val_pipeline=val_pipeline,   # held-out , agent acts, never learns
)
```

On every eval window, the loop runs one val episode (`act()` only, no `learn()`), then calls `evaluator.combine(train_metrics, val_metrics)`:

```
primary_score        = val_score          ← drives ALL checkpoint and rollback decisions
generalization_gap   = train_score − val  ← surfaced in every EvalMetrics
```

Every adaptation decision in the system , rollback, spawn, noise scale, MetaController , is driven by out-of-sample performance, not training performance.

### Three-Signal Adaptive Noise

When spawning a policy variant, the mutation intensity is computed from three signals:

**Signal 1 , Val slope (improvement rate)**

```
scores = [primary_score₁, ..., primary_scoreₙ]
slope  = linear_regression_slope(scores)
t      = clip(slope / max_slope, 0, 1)
```

`t → 1` when val is improving strongly. `t → 0` on plateau.

**Signal 2 , Generalization gap**

```
gap_penalty = clip(mean(train − val) / |mean(val)|, 0, 1)
```

Large gap means the model fits training data but not held-out data , explore different solutions.

**Signal 3 , Train/val correlation (Pearson)**

```
corr         = Pearson(train_scores, val_scores)
corr_penalty = clip(1 − corr, 0, 1)
```

`corr → 1` means train and val are moving together (healthy). `corr → 0` or negative means train is moving but val isn't following , a signal to explore.

**Combined formula:**

```
effective_t = t × (1 − 0.5 × gap_penalty) × (1 − 0.5 × corr_penalty)
noise_scale = max_scale − effective_t × (max_scale − min_scale)
```

When the system is healthy (val improving, low gap, high correlation): `effective_t → 1`, `noise_scale → min_scale`. When overfitting or diverging: `effective_t → 0`, `noise_scale → max_scale`. The mutation intensity is automatically calibrated to the health of the system.

---

## Policy Evolution

### Separation of Concerns

```
BackoffOptimizer / PBTOptimizer  →  tunes hyperparameters
PolicyManager                    →  evolves models (rollback, spawn, prune, ensemble)
```

### Automatic Rollback

When the loop reaches DORMANT, `PolicyManager` compares the current score against the best checkpoint. If current < best, it loads the best known weights back into the agent automatically.

```python
from tensor_optix import PolicyManager
from tensor_optix.core.checkpoint_registry import CheckpointRegistry

registry = CheckpointRegistry("./checkpoints")
pm = PolicyManager(registry)

opt = RLOptimizer(
    agent=agent,
    pipeline=pipeline,
    callbacks=[pm.as_callback(agent)],
)
opt.run()
```

### Spawn Budget , When Is Training Done?

Without a budget, DORMANT → spawn → new training → DORMANT → spawn → forever. `max_spawns` defines termination:

```python
pm = PolicyManager(registry, max_spawns=3)
cb = pm.as_callback(agent)
cb.set_stop_fn(opt.stop)   # called automatically when budget exhausted

opt.run()  # returns cleanly when budget is exhausted
```

When budget is exhausted, a training report is printed automatically:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Training Complete
  Reason           : Spawn budget exhausted
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Best score       : 0.8732
  Val score        : 0.8612
  Generalization   : 0.0120  (train − val)
  Spawns used      : 3 / 3
  Pruned agents    : 1
  Ensemble size    : 3
  Regime           : trending
  Agents           :
    [0] weight=2.4100  mean_score=0.8710
    [1] weight=1.0300  mean_score=0.7240
    [2] weight=0.5600  mean_score=0.6120
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### Autonomous Spawning

Provide an `agent_factory` and the system spawns variants on every DORMANT without manual intervention:

```python
def make_agent():
    return PPOAgent(model=build_model(), optimizer=..., hyperparams=...)

pm = PolicyManager(registry, max_spawns=5, max_ensemble_size=4)
cb = pm.as_callback(agent, agent_factory=make_agent)
cb.set_stop_fn(opt.stop)
```

On DORMANT:
1. Rebalance ensemble weights from rolling score history
2. Rollback to best checkpoint if current < best
3. Call `agent_factory()` to create a fresh shell
4. Compute adaptive noise from three signals
5. Clone best checkpoint into shell, perturb hyperparams
6. Add to ensemble, prune if over `max_ensemble_size`
7. When budget exhausted → call `stop_fn()` → print report

### MetaController , Autonomous Decisions

`MetaController` observes the full metrics history and decides what to do on each DORMANT:

```
STOP   → budget exhausted
PRUNE  → generalization gap > gap_threshold (overfitting)
SPAWN  → low train/val correlation or improvement plateau
NO_OP  → system is healthy, let it run
```

```python
from tensor_optix import MetaController

cb = pm.as_callback(
    agent,
    agent_factory=make_agent,
    meta_controller=MetaController(
        gap_threshold=0.3,          # normalized gap level above this → PRUNE
        gap_slope_threshold=0.02,   # gap widening rate above this → PRUNE
        improvement_threshold=0.05, # normalized val slope below this → SPAWN
    ),
)
```

The MetaController interface is identical to any learned policy , swap it for a neural network decision maker without changing anything else.

### Ensemble , Multiple Policies

Actions are combined as a weighted average: `a = Σ(wᵢ × aᵢ) / Σ(wᵢ)`

```python
from tensor_optix import PolicyManager, EnsembleAgent

pm = PolicyManager(registry)
pm.add_agent(agent_trending,  weight=1.0)
pm.add_agent(agent_ranging,   weight=1.0)
pm.add_agent(agent_volatile,  weight=1.0)

ensemble = EnsembleAgent(pm, primary_agent=agent_trending)

opt = RLOptimizer(
    agent=ensemble,
    pipeline=BatchPipeline(env=env, agent=ensemble, window_size=200),
    callbacks=[pm.as_callback(agent_trending)],
)
```

### Autonomous Weight Rebalancing

```python
# Record per-agent scores , happens every evaluation window
pm.record_agent_score(0, sharpe_trending)
pm.record_agent_score(1, sharpe_ranging)

# auto_update_weights() is called automatically on DORMANT
# Weights shift proportionally to rolling mean score
pm.auto_update_weights()
```

Scores tracked in a rolling window (`score_window=10`). Higher mean score → proportionally higher weight.

### Population Control

```python
# Prune the lowest-weight agent when ensemble grows too large
pm.prune(bottom_k=1)   # removes lowest-weight agent, remaps score history indices

# Boost a specific agent's weight after regime detection
pm.boost(agent_trending, factor=2.0)  # others proportionally reduced at action time
```

### Regime Detection

```python
from tensor_optix import RegimeDetector

detector = RegimeDetector(
    volatility_threshold=0.2,   # CV above this → "volatile"
    trend_threshold=0.05,       # normalized slope above this → "trending"
    window=10,
)

regime = detector.detect(metrics_history)  # "trending" | "ranging" | "volatile"
pm.set_regime(regime)
pm.boost(regime_agents[regime], factor=2.0)
```

For domain-specific signals (VIX, ATR, Sharpe), subclass and override `detect()`.

### Observability

```python
import json
print(json.dumps(pm.status(), indent=2))
# {
#   "ensemble_size": 3,
#   "agents": [
#     {"index": 0, "weight": 2.41, "mean_score": 0.871, "recent_scores": [...]},
#     ...
#   ],
#   "regime": "trending",
#   "spawn_count": 2,
#   "prune_count": 1,
#   "max_spawns": 5,
#   "spawns_remaining": 3,
#   "budget_exhausted": false
# }
```

---

## Custom Evaluator

```python
from tensor_optix import BaseEvaluator, EpisodeData, EvalMetrics

class SharpeEvaluator(BaseEvaluator):
    def score(self, episode_data: EpisodeData, train_diagnostics: dict) -> EvalMetrics:
        rewards = np.array(episode_data.rewards)
        sharpe = rewards.mean() / (rewards.std() + 1e-8)
        return EvalMetrics(
            primary_score=float(sharpe),
            metrics={"sharpe": float(sharpe), "mean_reward": float(rewards.mean())},
            episode_id=episode_data.episode_id,
        )

opt = RLOptimizer(agent=agent, pipeline=pipeline, evaluator=SharpeEvaluator())
```

For train+val combined scoring, override `combine()`:

```python
class ConservativeEvaluator(BaseEvaluator):
    def combine(self, train: EvalMetrics, val: EvalMetrics) -> EvalMetrics:
        score = min(train.primary_score, val.primary_score)  # must be good on both
        return EvalMetrics(
            primary_score=score,
            metrics={
                "train_score": train.primary_score,
                "val_score": val.primary_score,
                "generalization_gap": train.primary_score - val.primary_score,
            },
            episode_id=train.episode_id,
        )
```

---

## Live Pipeline

For real-time data sources (trading, robotics, online environments):

```python
from tensor_optix import LivePipeline

class MarketFeed:
    def stream(self):
        while True:
            yield obs, reward, terminated, truncated, info

pipeline = LivePipeline(
    data_source=MarketFeed(),
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
        print(f"Converged at window {window_id}")

opt = RLOptimizer(agent=agent, pipeline=pipeline, callbacks=[MyLogger()])
```

Available hooks: `on_loop_start`, `on_loop_stop`, `on_episode_end`, `on_improvement`, `on_plateau`, `on_dormant`, `on_degradation`, `on_hyperparam_update`.

---

## Full Configuration

```python
opt = RLOptimizer(
    agent=agent,
    pipeline=pipeline,
    val_pipeline=val_pipeline,              # optional held-out pipeline
    evaluator=None,                         # default: TFEvaluator
    optimizer=None,                         # default: BackoffOptimizer
    checkpoint_dir="./checkpoints",
    max_snapshots=10,
    rollback_on_degradation=False,
    improvement_margin=0.0,
    max_episodes=None,                      # None = run until DORMANT
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
│   ├── base_agent.py           # BaseAgent , 6-method contract
│   ├── base_evaluator.py       # BaseEvaluator , score, combine, compare
│   ├── base_optimizer.py       # BaseOptimizer , suggest, on_improvement, on_plateau
│   ├── base_pipeline.py        # BasePipeline , episodes() generator
│   ├── loop_controller.py      # State machine + main loop
│   ├── backoff_scheduler.py    # Convergence detection + state transitions
│   ├── checkpoint_registry.py  # Snapshot storage and manifest
│   ├── policy_manager.py       # PolicyManager + PolicyManagerCallback
│   ├── ensemble_agent.py       # EnsembleAgent , multi-policy BaseAgent wrapper
│   ├── regime_detector.py      # RegimeDetector , score-based regime classification
│   └── meta_controller.py      # MetaController , SPAWN/PRUNE/STOP/NO_OP decisions
├── adapters/tensorflow/
│   ├── tf_agent.py             # TFAgent , Keras model wrapper (A2C advantage or REINFORCE)
│   └── tf_evaluator.py         # TFEvaluator , default scorer
├── pipeline/
│   ├── batch_pipeline.py       # Continuous stepping, fixed windows
│   └── live_pipeline.py        # Real-time streaming
└── optimizers/
    ├── backoff_optimizer.py    # Two-phase finite difference
    └── pbt_optimizer.py        # Pseudo population-based training
```

| Component | Responsibility |
|-----------|---------------|
| `LoopController` | State machine, episode orchestration, eval, checkpoint |
| `BackoffScheduler` | Convergence detection via exponential backoff |
| `CheckpointRegistry` | Snapshot storage, best-checkpoint manifest |
| `BackoffOptimizer` | Two-phase finite difference hyperparameter tuning |
| `PBTOptimizer` | Population-based exploit/explore hyperparameter tuning |
| `PolicyManager` | Rollback, spawn, prune, boost, ensemble weights, adaptive noise |
| `PolicyManagerCallback` | Autonomous evolution on every DORMANT event |
| `MetaController` | Rule-based (or learned) SPAWN/PRUNE/STOP/NO_OP decisions |
| `EnsembleAgent` | Weighted-average action combining across multiple agents |
| `RegimeDetector` | Score-based regime classification (trending / ranging / volatile) |

---

## Math & Science Reference

This section documents the mathematical decisions behind each component.

### A2C Advantage Baseline (`TFAgent`)

The base `TFAgent.learn()` supports two gradient estimators:

**REINFORCE (fallback, no `episode_data.values`):**
```
∇J ≈ Σ ∇log π(aₜ|sₜ) · Ĝₜ        where Ĝₜ = normalized discounted return
```
Variance is O(T²). Converges but slowly.

**Actor-Critic advantage (when `episode_data.values` is set):**
```
Aₜ = Gₜ − V(sₜ)                    advantage = return − critic estimate
∇J ≈ Σ ∇log π(aₜ|sₜ) · Âₜ         where Âₜ = normalized advantage
```
By the policy gradient theorem, subtracting V(sₜ) does not bias the gradient while dramatically reducing variance. The explained variance diagnostic in `learn()` diagnostics measures critic quality: 1.0 = perfect baseline, 0.0 = useless, <0 = harmful.

To use A2C, populate `episode_data.values` with your critic's V(sₜ) estimates before calling `learn()`.

### PBT Perturbation Modes (`PBTOptimizer`)

Parameters that span orders of magnitude (learning rates, weight decay) require multiplicative perturbation, not additive. Additive noise on `[1e-4, 1e-1]` would oversample near the top and undersample the critical lower end.

**Linear (default for bounded params):**
```
δ = scale × (high − low)
θ' = clip(θ + Uniform(−δ, +δ), low, high)
```

**Log-scale (for `learning_rate`, `lr`, `alpha`, `epsilon`, `weight_decay`):**
```
δ_log = scale × log(high / low)
θ' = clip(θ × exp(Uniform(−δ_log, +δ_log)), low, high)
```
Equal probability mass per decade, following Jaderberg et al. 2017 (PBT). Custom param names can be passed via `log_scale_params` constructor arg.

### Detrended Volatility (`RegimeDetector`)

Raw CV (`std/|mean|`) conflates volatility with trend direction — a steadily declining score has low CV but is not "ranging". The corrected metric:

```
trend_line  = linear_regression(scores)
residuals   = scores − trend_line
CV_detrended = std(residuals) / (|mean(scores)| + ε)
```

A single `np.polyfit` call produces both the slope (used for trend classification) and the residuals (used for volatility), with no redundant computation.

### Degradation Floor (`BackoffScheduler`)

The watchdog threshold:
```
allowed_drop = max(
    |best_score| × (1 − degradation_threshold),   # relative
    min_degradation_drop,                           # absolute floor
)
degraded = score < best_score − allowed_drop
```
The floor prevents spurious resets when `best_score ≈ 0`, where the relative term collapses to near-zero and any noise fires the watchdog. Default `min_degradation_drop=1e-4` suits normalized score ranges. Increase for raw reward scales.

### Gap-Slope Overfitting Signal (`MetaController`)

The former Signal 2 (Pearson correlation) was replaced with gap slope. Pearson measures whether train and val move together in shape — not whether they diverge in level. A pair like `train=[0.9, 0.91, 0.92]` and `val=[0.3, 0.31, 0.32]` has r=1.0 but is catastrophically overfit.

Gap slope detects active overfitting progression:
```
gap_t = (train_score_t − val_score_t) / |val_score_t|   # normalized gap at t
slope = linear_regression_slope(gap_t)                    # is it widening?
```
`slope > gap_slope_threshold` → PRUNE, even if the current gap level is below threshold.

### Ensemble Multi-Agent Learning (`EnsembleAgent`)

`EnsembleAgent.learn()` trains all registered agents on the same `EpisodeData`. Without this, non-primary agents diverge from the primary as training progresses — their action distributions become stale while the primary improves. The ensemble then degrades: you pay the cost of N agents but only get 1 improving policy. Primary agent diagnostics are returned to the evaluator; per-agent diagnostics are logged at DEBUG level.

---

## License

MIT , Copyright (c) 2026 sup3rus3r
