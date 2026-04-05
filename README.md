# tensor-optix

Autonomous training loop for any sequential learning model — built-in PPO, DQN, and SAC for TensorFlow and PyTorch.

---

## About

tensor-optix is a framework-agnostic autonomous training loop. It owns evaluation, checkpointing, hyperparameter tuning, policy evolution, and ensemble management for **any model that can act and learn from sequential data** — reinforcement learning agents, online forecasters, trading systems, robotics controllers, or any custom architecture that fits the six-method `BaseAgent` interface.

The framework has zero assumptions about your model, algorithm, or framework. No RL-specific logic exists in the core loop — it works equally well with PPO, a custom evolutionary strategy, a supervised sequence model, or anything else. For RL specifically, it ships with production-ready implementations of **PPO, DQN, and SAC** for both TensorFlow and PyTorch so you can start training without writing a single algorithm line.

**The system never stops at a fixed episode count.** It detects convergence through exponential backoff, spawns policy variants when it plateaus, weights an ensemble by rolling performance, and uses both training and validation signals to drive every decision — not training alone.

**Core philosophy:** We own the loop. You own the model.

---

## Install

```bash
# TensorFlow (default — includes PPO, DQN, SAC for TF)
pip install tensor-optix

# PyTorch support
pip install tensor-optix[torch]

# Both + Atari + MuJoCo
pip install tensor-optix[all]
```

**Requirements:** Python >= 3.11, Gymnasium >= 1.0, NumPy >= 1.24.
TensorFlow >= 2.18 is required for TF algorithms. PyTorch >= 2.0 is required for Torch algorithms. The core loop, PolicyManager, and all ensemble/evolution logic are framework-free.

---

## Quick Start

### TensorFlow PPO

```python
import tensorflow as tf
import gymnasium as gym
from tensor_optix import RLOptimizer, TFPPOAgent, BatchPipeline, HyperparamSet

actor = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(4,)),
    tf.keras.layers.Dense(64, activation="tanh"),
    tf.keras.layers.Dense(64, activation="tanh"),
    tf.keras.layers.Dense(2),                    # logits, one per action
])
critic = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(4,)),
    tf.keras.layers.Dense(64, activation="tanh"),
    tf.keras.layers.Dense(64, activation="tanh"),
    tf.keras.layers.Dense(1),                    # scalar value estimate
])

agent = TFPPOAgent(
    actor=actor,
    critic=critic,
    optimizer=tf.keras.optimizers.Adam(3e-4),
    hyperparams=HyperparamSet(params={
        "learning_rate": 3e-4,
        "clip_ratio":    0.2,
        "entropy_coef":  0.01,
        "vf_coef":       0.5,
        "gamma":         0.99,
        "gae_lambda":    0.95,
        "n_epochs":      10,
        "minibatch_size": 64,
    }, episode_id=0),
)

env      = gym.make("CartPole-v1")
pipeline = BatchPipeline(env=env, agent=agent, window_size=2048)

opt = RLOptimizer(agent=agent, pipeline=pipeline)
opt.run()  # runs until convergence, auto-tunes hyperparams, saves best checkpoint
```

### PyTorch PPO

```python
import torch
import torch.nn as nn
import gymnasium as gym
from tensor_optix import RLOptimizer, BatchPipeline, HyperparamSet
from tensor_optix.algorithms.torch_ppo import TorchPPOAgent

obs_dim, n_actions = 4, 2

actor  = nn.Sequential(nn.Linear(obs_dim, 64), nn.Tanh(), nn.Linear(64, n_actions))
critic = nn.Sequential(nn.Linear(obs_dim, 64), nn.Tanh(), nn.Linear(64, 1))

agent = TorchPPOAgent(
    actor=actor,
    critic=critic,
    optimizer=torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()), lr=3e-4
    ),
    hyperparams=HyperparamSet(params={
        "learning_rate": 3e-4, "clip_ratio": 0.2, "entropy_coef": 0.01,
        "gamma": 0.99, "gae_lambda": 0.95, "n_epochs": 10, "minibatch_size": 64,
    }, episode_id=0),
)

pipeline = BatchPipeline(env=gym.make("CartPole-v1"), agent=agent, window_size=2048)
opt = RLOptimizer(agent=agent, pipeline=pipeline)
opt.run()
```

---

## Built-in Algorithms

### PPO — Proximal Policy Optimization

Discrete action spaces. Actor + critic are separate models.

```python
from tensor_optix.algorithms.tf_ppo import TFPPOAgent   # TensorFlow
from tensor_optix.algorithms.torch_ppo import TorchPPOAgent  # PyTorch
```

**What it implements:**
- GAE-λ advantage estimation with episode-boundary handling
- Clipped surrogate objective: `L = min(r·A, clip(r, 1−ε, 1+ε)·A)`
- Value function loss (MSE) with configurable coefficient
- Entropy bonus for exploration regularization
- n epochs of shuffled minibatch gradient descent per rollout
- Global gradient norm clipping

**Hyperparams exposed to the tuner:**

| Key | Default | Description |
|-----|---------|-------------|
| `learning_rate` | `3e-4` | Adam learning rate |
| `clip_ratio` | `0.2` | PPO clipping epsilon (ε) |
| `entropy_coef` | `0.01` | Entropy bonus weight |
| `vf_coef` | `0.5` | Value loss weight |
| `gamma` | `0.99` | Discount factor |
| `gae_lambda` | `0.95` | GAE smoothing (0 = TD, 1 = MC) |
| `n_epochs` | `10` | Update epochs per rollout |
| `minibatch_size` | `64` | Minibatch size |
| `max_grad_norm` | `0.5` | Gradient clipping norm |

---

### DQN — Deep Q-Network

Discrete action spaces. Single Q-network; target network updated periodically.

```python
from tensor_optix.algorithms.tf_dqn import TFDQNAgent     # TensorFlow
from tensor_optix.algorithms.torch_dqn import TorchDQNAgent  # PyTorch
```

**What it implements:**
- Experience replay buffer (circular)
- Target network with hard periodic updates
- Epsilon-greedy exploration with multiplicative decay
- TD loss: `MSE(Q(s,a), r + γ max Q_target(s',·))`

```python
q_net = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(obs_dim,)),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(n_actions),    # Q-values for all actions
])

agent = TFDQNAgent(
    q_network=q_net,
    n_actions=n_actions,
    optimizer=tf.keras.optimizers.Adam(1e-3),
    hyperparams=HyperparamSet(params={
        "learning_rate": 1e-3, "gamma": 0.99,
        "epsilon": 1.0, "epsilon_min": 0.05, "epsilon_decay": 0.995,
        "batch_size": 64, "target_update_freq": 100,
        "replay_capacity": 100_000,
    }, episode_id=0),
)
```

---

### SAC — Soft Actor-Critic

Continuous action spaces. Squashed Gaussian actor, twin critics, auto-entropy tuning.

```python
from tensor_optix.algorithms.tf_sac import TFSACAgent     # TensorFlow
from tensor_optix.algorithms.torch_sac import TorchSACAgent  # PyTorch
```

**What it implements:**
- Reparameterized squashed Gaussian: `a = tanh(μ + σε)`, actions ∈ (−1, 1)
- Clipped double-Q (twin critics) to reduce overestimation bias
- Soft Polyak target network updates: `θ_tgt ← τθ + (1−τ)θ_tgt`
- Auto-entropy temperature: learnable `log_α`, target entropy = `−dim(A)`
- Off-policy experience replay

```python
action_dim = 6   # e.g. MuJoCo Ant

actor   = tf.keras.Sequential([        # obs → [mean || log_std], shape [batch, 2*action_dim]
    tf.keras.layers.Input(shape=(obs_dim,)),
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dense(action_dim * 2),
])
critic1 = tf.keras.Sequential([        # [obs || action] → Q-value
    tf.keras.layers.Input(shape=(obs_dim + action_dim,)),
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dense(1),
])
critic2 = tf.keras.Sequential([...])   # independent weights (twin-Q)

agent = TFSACAgent(
    actor=actor, critic1=critic1, critic2=critic2,
    action_dim=action_dim,
    actor_optimizer=tf.keras.optimizers.Adam(3e-4),
    critic_optimizer=tf.keras.optimizers.Adam(3e-4),
    alpha_optimizer=tf.keras.optimizers.Adam(3e-4),
    hyperparams=HyperparamSet(params={
        "learning_rate": 3e-4, "gamma": 0.99,
        "tau": 0.005, "batch_size": 256,
        "replay_capacity": 1_000_000,
    }, episode_id=0),
)
```

---

### Parallel Environments — VectorBatchPipeline

Run N environments simultaneously. Increases sample throughput N× over a single `BatchPipeline`.

```python
import gymnasium as gym
from tensor_optix.pipeline.vector_pipeline import VectorBatchPipeline

env_fns  = [lambda: gym.make("CartPole-v1")] * 8
pipeline = VectorBatchPipeline(env_fns=env_fns, window_size=256)

# Async (subprocess-based) for CPU-bound envs:
pipeline = VectorBatchPipeline(env_fns=env_fns, window_size=256, async_envs=True)
```

---

### Observation & Reward Normalization

```python
from tensor_optix.core.normalizers import ObsNormalizer, RewardNormalizer

obs_norm    = ObsNormalizer(obs_shape=(obs_dim,), clip=10.0)
reward_norm = RewardNormalizer(gamma=0.99, clip=10.0)

# Update stats from a collected rollout, then normalize in act():
obs_norm.update(obs_batch)
normed_obs = obs_norm.normalize(raw_obs)

# Scale rewards using running return std:
reward_norm.step(raw_reward)
scaled_reward = reward_norm.normalize([raw_reward])[0]
```

---

### Extend Any Algorithm

All algorithms implement `BaseAgent`. Subclass to add domain-specific logic without reimplementing PPO:

```python
class TradingPPO(TFPPOAgent):
    """Adds action distribution logging to standard PPO."""

    def learn(self, episode_data):
        diag = super().learn(episode_data)           # full PPO update
        obs  = tf.cast(episode_data.observations[:256], tf.float32)
        probs = tf.nn.softmax(self._actor(obs, training=False))
        probs_mean = tf.reduce_mean(probs, axis=0).numpy()
        diag["p_flat"]  = float(probs_mean[0])
        diag["p_long"]  = float(probs_mean[1])
        diag["p_short"] = float(probs_mean[2])
        return diag
```

---

## The Loop Interface

The core loop calls exactly **six methods** on any agent. Nothing else is assumed.

```python
class BaseAgent(ABC):
    def act(self, observation) -> any: ...         # any action type
    def learn(self, episode_data) -> dict: ...     # any algorithm
    def get_hyperparams(self) -> HyperparamSet: ...
    def set_hyperparams(self, hp: HyperparamSet): ...
    def save_weights(self, path: str): ...
    def load_weights(self, path: str): ...
```

Any model — RL algorithms, evolutionary strategies, supervised sequence models, online forecasters — plugs in by implementing these six methods. No specific framework, action space, or learning paradigm is assumed.

---

## How It Works

### The Loop States

```
ACTIVE   → aggressive tuning, evaluates every window
COOLING  → recent improvement, exponential backoff on eval frequency
DORMANT  → plateau reached — model is trained, minimal intervention
WATCHDOG → monitoring for degradation
```

**DORMANT = trained.** Not a fixed episode count — the system backs off evaluation geometrically until improvement stops, then declares convergence.

### Backoff Schedule

```
interval₀ = base_interval
intervalₙ = min(intervalₙ₋₁ × backoff_factor, max_interval_episodes)

Plateau detected when:  consecutive_no_improvement ≥ plateau_threshold
DORMANT declared when:  consecutive_no_improvement ≥ dormant_threshold
```

Every improvement resets the backoff counter. The system accelerates evaluation when learning is happening, backs off when it isn't.

### Hyperparameter Optimizer — Two-Phase Finite Difference

`BackoffOptimizer` cycles through hyperparameters using staggered two-phase finite difference:

```
For each param θᵢ:
  Phase 1 (probe):   apply θᵢ + δᵢ, run one window, record score s₊
  Phase 2 (commit):  gradient ĝᵢ = (s₊ − s₀) / δᵢ
                     if ĝᵢ > 0: keep θᵢ + δᵢ
                     if ĝᵢ ≤ 0: apply θᵢ − δᵢ  (reverse direction)
```

Step size `δᵢ` adapts: shrinks on improvement, grows on plateau. Params cycle round-robin — each is probed and committed independently.

`PBTOptimizer` maintains a history of `(hyperparams, score)` pairs and exploits top performers when in the bottom 20%, otherwise explores with Gaussian perturbation.

---

## The Science: Train + Val Together

Without validation, every decision — checkpoint saves, rollbacks, spawn triggers — is made on training data alone. That is overfitting disguised as improvement.

### Validation Pipeline

```python
opt = RLOptimizer(
    agent=agent,
    pipeline=train_pipeline,
    val_pipeline=val_pipeline,   # held-out — agent acts, never learns
)
```

On every eval window, the loop runs one val episode (`act()` only, no `learn()`), then calls `evaluator.combine(train_metrics, val_metrics)`:

```
primary_score        = val_score          ← drives ALL checkpoint and rollback decisions
generalization_gap   = train_score − val  ← surfaced in every EvalMetrics
```

Every adaptation decision — rollback, spawn, noise scale, MetaController — is driven by out-of-sample performance, not training performance.

### Three-Signal Adaptive Noise

When spawning a policy variant, the mutation intensity is computed from three signals:

**Signal 1 — Val slope (improvement rate)**
```
t = clip(slope(val_scores) / max_slope, 0, 1)
```
`t → 1` when val is improving strongly. `t → 0` on plateau.

**Signal 2 — Generalization gap**
```
gap_penalty = clip(mean(train − val) / |mean(val)|, 0, 1)
```
Large gap means the model fits training data but not held-out data — explore different solutions.

**Signal 3 — Train/val correlation (Pearson)**
```
corr_penalty = clip(1 − Pearson(train_scores, val_scores), 0, 1)
```
`corr → 1` means train and val move together (healthy). `corr → 0` means train is moving but val isn't — a signal to explore.

**Combined formula:**
```
effective_t = t × (1 − 0.5 × gap_penalty) × (1 − 0.5 × corr_penalty)
noise_scale = max_scale − effective_t × (max_scale − min_scale)
```

---

## Policy Evolution

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

### Spawn Budget — When Is Training Done?

```python
pm = PolicyManager(registry, max_spawns=3)
cb = pm.as_callback(agent)
cb.set_stop_fn(opt.stop)

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

```python
def make_agent():
    return TFPPOAgent(actor=build_actor(), critic=build_critic(),
                      optimizer=tf.keras.optimizers.Adam(3e-4),
                      hyperparams=initial_hp)

pm = PolicyManager(registry, max_spawns=5, max_ensemble_size=4)
cb = pm.as_callback(agent, agent_factory=make_agent)
cb.set_stop_fn(opt.stop)
```

On DORMANT: rebalance ensemble weights → rollback if degraded → clone best checkpoint → perturb hyperparams → add to ensemble → prune if over limit → stop when budget exhausted.

### MetaController — Autonomous Decisions

```python
from tensor_optix import MetaController

cb = pm.as_callback(
    agent,
    agent_factory=make_agent,
    meta_controller=MetaController(
        gap_threshold=0.3,
        gap_slope_threshold=0.02,
        improvement_threshold=0.05,
    ),
)
```

`MetaController` decides `SPAWN / PRUNE / STOP / NO_OP` on each DORMANT event based on generalization gap level, gap slope (overfitting progression), and val improvement rate.

### Ensemble

```python
from tensor_optix import PolicyManager, EnsembleAgent

pm = PolicyManager(registry)
pm.add_agent(agent_a, weight=1.0)
pm.add_agent(agent_b, weight=1.0)

ensemble = EnsembleAgent(pm, primary_agent=agent_a)
# Actions: a = Σ(wᵢ × aᵢ) / Σ(wᵢ)

opt = RLOptimizer(
    agent=ensemble,
    pipeline=BatchPipeline(env=env, agent=ensemble, window_size=2048),
    callbacks=[pm.as_callback(agent_a)],
)
```

---

## Custom Evaluator

```python
from tensor_optix import BaseEvaluator, EpisodeData, EvalMetrics
import numpy as np

class SharpeEvaluator(BaseEvaluator):
    def score(self, episode_data: EpisodeData, train_diagnostics: dict) -> EvalMetrics:
        rewards = np.array(episode_data.rewards)
        sharpe  = rewards.mean() / (rewards.std() + 1e-8)
        return EvalMetrics(
            primary_score=float(sharpe),
            metrics={"sharpe": float(sharpe), "mean_reward": float(rewards.mean())},
            episode_id=episode_data.episode_id,
        )

opt = RLOptimizer(agent=agent, pipeline=pipeline, evaluator=SharpeEvaluator())
```

---

## Live Pipeline

```python
from tensor_optix import LivePipeline

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
│   ├── base_agent.py           # BaseAgent — 6-method contract
│   ├── base_evaluator.py       # BaseEvaluator — score, combine, compare
│   ├── base_optimizer.py       # BaseOptimizer — suggest, on_improvement, on_plateau
│   ├── base_pipeline.py        # BasePipeline — episodes() generator
│   ├── loop_controller.py      # State machine + main loop
│   ├── backoff_scheduler.py    # Convergence detection + state transitions
│   ├── checkpoint_registry.py  # Snapshot storage and manifest
│   ├── normalizers.py          # RunningMeanStd, ObsNormalizer, RewardNormalizer
│   ├── trajectory_buffer.py    # compute_gae(), make_minibatches()
│   ├── policy_manager.py       # PolicyManager + PolicyManagerCallback
│   ├── ensemble_agent.py       # EnsembleAgent — multi-policy BaseAgent wrapper
│   ├── regime_detector.py      # RegimeDetector — score-based regime classification
│   └── meta_controller.py      # MetaController — SPAWN/PRUNE/STOP/NO_OP decisions
├── adapters/
│   ├── tensorflow/
│   │   ├── tf_agent.py         # TFAgent — generic REINFORCE / A2C base
│   │   └── tf_evaluator.py     # TFEvaluator — default scorer
│   └── pytorch/
│       ├── torch_agent.py      # TorchAgent — generic REINFORCE / A2C base
│       └── torch_evaluator.py  # TorchEvaluator — default scorer (no TF dep)
├── algorithms/
│   ├── tf_ppo.py               # TFPPOAgent — clipped surrogate, GAE-λ, n-epoch minibatch
│   ├── tf_dqn.py               # TFDQNAgent — replay buffer, target net, ε-greedy
│   ├── tf_sac.py               # TFSACAgent — twin critics, squashed Gaussian, auto-α
│   ├── torch_ppo.py            # TorchPPOAgent
│   ├── torch_dqn.py            # TorchDQNAgent
│   └── torch_sac.py            # TorchSACAgent
└── pipeline/
    ├── batch_pipeline.py       # Continuous stepping, fixed windows
    ├── live_pipeline.py        # Real-time streaming with reconnect
    └── vector_pipeline.py      # Parallel envs via gymnasium.vector
```

| Component | Responsibility |
|-----------|---------------|
| `TFPPOAgent` / `TorchPPOAgent` | PPO with GAE, clipping, entropy, n-epoch minibatch |
| `TFDQNAgent` / `TorchDQNAgent` | DQN with replay buffer, target net, ε-greedy |
| `TFSACAgent` / `TorchSACAgent` | SAC with twin critics, auto-entropy, soft updates |
| `LoopController` | State machine, episode orchestration, eval, checkpoint |
| `BackoffScheduler` | Convergence detection via exponential backoff |
| `CheckpointRegistry` | Snapshot storage, best-checkpoint manifest |
| `BackoffOptimizer` | Two-phase finite difference hyperparameter tuning |
| `PBTOptimizer` | Population-based exploit/explore hyperparameter tuning |
| `PolicyManager` | Rollback, spawn, prune, boost, ensemble weights, adaptive noise |
| `MetaController` | Rule-based SPAWN/PRUNE/STOP/NO_OP decisions |
| `EnsembleAgent` | Weighted-average action combining across multiple agents |
| `RegimeDetector` | Score-based regime classification (trending / ranging / volatile) |
| `VectorBatchPipeline` | Parallel environment rollouts via gymnasium.vector |
| `ObsNormalizer` | Online running mean/std observation normalization |
| `RewardNormalizer` | Return-std reward scaling |

---

## Math & Science Reference

### GAE-λ (`trajectory_buffer.compute_gae`)

```
δₜ = rₜ + γ·V(sₜ₊₁)·(1−dₜ) − V(sₜ)
Aₜ = δₜ + γλ·(1−dₜ)·Aₜ₊₁
```

The `(1−dₜ)` mask zeros both the bootstrap from V(sₜ₊₁) and the GAE propagation from Aₜ₊₁ simultaneously, so a single window containing multiple episode fragments is handled correctly without splitting by episode.

### PPO Clipped Surrogate (`TFPPOAgent` / `TorchPPOAgent`)

```
rₜ(θ) = π_θ(aₜ|sₜ) / π_θ_old(aₜ|sₜ)

L_clip = E[ min(rₜ·Âₜ, clip(rₜ, 1−ε, 1+ε)·Âₜ) ]
L_vf   = E[ (V_θ(sₜ) − Rₜ)² ]
L_ent  = E[ H(π_θ(·|sₜ)) ]

L = −L_clip + c₁·L_vf − c₂·L_ent
```

### SAC Squashed Gaussian (`TFSACAgent` / `TorchSACAgent`)

```
u ~ N(μ_θ(s), σ_θ(s))
a = tanh(u)

log π(a|s) = log N(u; μ, σ) − Σᵢ log(1 − tanh²(uᵢ) + ε)
```

Auto-entropy: `L_α = −α · (log π(aₜ|sₜ) + H_target)` where `H_target = −dim(A)`.

### A2C Advantage Baseline (`TFAgent` / `TorchAgent`)

```
Aₜ = Gₜ − V(sₜ)
∇J ≈ Σ ∇log π(aₜ|sₜ) · Âₜ
```

By the policy gradient theorem, subtracting V(sₜ) does not bias the gradient while reducing variance. The `explained_variance` diagnostic in `learn()` measures critic quality: 1.0 = perfect, 0.0 = useless, <0 = harmful.

### PBT Perturbation Modes (`PBTOptimizer`)

**Log-scale** for `learning_rate`, `epsilon`, `weight_decay`:
```
δ_log = scale × log(high / low)
θ' = clip(θ × exp(Uniform(−δ_log, +δ_log)), low, high)
```
Equal probability mass per decade, following Jaderberg et al. 2017.

### Detrended Volatility (`RegimeDetector`)

```
residuals    = scores − linear_regression(scores)
CV_detrended = std(residuals) / (|mean(scores)| + ε)
```
Avoids conflating a steadily improving score with stability.

### Gap-Slope Overfitting Signal (`MetaController`)

```
gapₜ = (train_scoreₜ − val_scoreₜ) / |val_scoreₜ|
slope = linear_regression_slope(gap₀, ..., gapₙ)
slope > threshold → PRUNE
```
Detects active overfitting progression before gap level alone triggers.

---

## License

MIT — Copyright (c) 2026 sup3rus3r
