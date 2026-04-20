# tensor-optix

Autonomous training loop for any sequential learning model — built-in PPO, DQN, SAC, TD3, Rainbow DQN, and Recurrent PPO for TensorFlow, PyTorch, and JAX/Flax.

---

## About

tensor-optix is a framework-agnostic autonomous training loop. It owns evaluation, checkpointing, hyperparameter tuning, policy evolution, and ensemble management for **any model that can act and learn from sequential data** — reinforcement learning agents, online forecasters, trading systems, robotics controllers, or any custom architecture that fits the six-method `BaseAgent` interface.

The framework has zero assumptions about your model, algorithm, or framework. No RL-specific logic exists in the core loop — it works equally well with PPO, a custom evolutionary strategy, a supervised sequence model, or anything else. For RL specifically, it ships with production-ready implementations of **PPO, DQN, SAC, TD3, Rainbow DQN, and Recurrent PPO** for TensorFlow and PyTorch, plus a **JAX/Flax PPO adapter**, so you can start training without writing a single algorithm line.

**The system never stops at a fixed episode count.** It detects convergence through exponential backoff, spawns policy variants when it plateaus, and uses both training and validation signals to drive every decision — not training alone.

**New in 1.9.0:**
- **Distributed async actor-learner** (IMPALA + V-trace) — N actors run in parallel POSIX shared-memory processes, learner applies V-trace off-policy correction, 4× sample throughput on CPU
- **JAX/Flax PPO adapter** — `FlaxPPOAgent` and `FlaxAgent` base class; XLA-compiled updates, optax optimisers, pickle-safe weight serialisation; convergence-parity with `TorchPPOAgent`
- **Live terminal dashboard** — `RichDashboardCallback` renders a real-time Rich panel (score sparkline, hyperparams, loop state) with zero extra dependencies

**Core philosophy:** We own the loop. You own the model.

---

## Install

```bash
# TensorFlow (default — includes PPO, DQN, SAC for TF)
pip install tensor-optix

# PyTorch support (CUDA wheel must be installed separately — see below)
pip install tensor-optix[torch]

# JAX/Flax support (FlaxAgent, FlaxPPOAgent)
pip install tensor-optix[jax]

# Everything: PyTorch + JAX + Atari + MuJoCo
pip install tensor-optix[all]

# Box2D environments (requires swig: apt install swig / brew install swig)
pip install tensor-optix[box2d]

# CUDA compiler tools (NVIDIA GPU only)
pip install tensor-optix[cuda]
```

**PyTorch + CUDA:** `pip install tensor-optix[torch]` installs the CPU build of PyTorch from PyPI. For a CUDA-enabled build, install PyTorch first from the [official index](https://pytorch.org/get-started/locally/) before installing tensor-optix:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install tensor-optix
```

**Requirements:** Python >= 3.11, Gymnasium >= 1.0, NumPy >= 1.24.
TensorFlow >= 2.18 is required for TF algorithms. PyTorch >= 2.0 is required for Torch algorithms. JAX >= 0.10 + Flax >= 0.12.6 + optax >= 0.2.8 required for JAX algorithms. The core loop, PolicyManager, and all ensemble/evolution logic are framework-free.

---

## Benchmarks

### Feature showcase (~5 min, CPU)

Demonstrates the JAX/Flax adapter (convergence parity with PyTorch PPO) and the async actor-learner throughput (4× actors vs 1):

```bash
uv run python benchmarks/benchmark.py --demo
```

### Core benchmark — tensor-optix vs fixed loop (fast, 1 seed)

CartPole + LunarLander, 1 seed each:

```bash
uv run python benchmarks/benchmark.py --envs cartpole lunarlander --seeds 0
```

### Full benchmark — all 5 environments, 3 seeds

CartPole (DQN), LunarLander (PPO), Acrobot (PPO), LunarLanderContinuous (SAC), BipedalWalker (SAC):

```bash
uv run python benchmarks/benchmark.py
```

Additional flags:
```
--optix-only       skip baseline, run tensor-optix only
--no-plot          skip matplotlib chart
--verbose          print per-episode output
--seeds 0 1 2      override seeds
```

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
from tensor_optix.algorithms.tf_ppo import TFPPOAgent        # TensorFlow
from tensor_optix.algorithms.torch_ppo import TorchPPOAgent  # PyTorch (auto-detects CUDA)
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

### PPO Continuous — Gaussian Policy (Continuous Actions)

Continuous action spaces. Squashed Gaussian actor with learned log-std.

```python
from tensor_optix.algorithms.tf_ppo_continuous import TFGaussianPPOAgent        # TensorFlow
from tensor_optix.algorithms.torch_ppo_continuous import TorchGaussianPPOAgent  # PyTorch
```

**What it adds over discrete PPO:**
- Gaussian policy head outputs `[mean || log_std]` — `action_dim * 2` outputs
- Actions sampled as `a = tanh(mean + std * ε)`, squashed to `(−1, 1)`
- Log-prob corrected for tanh squashing (numerically stable)
- `action_dim` required at construction

```python
import torch.nn as nn
from tensor_optix.algorithms.torch_ppo_continuous import TorchGaussianPPOAgent

obs_dim, act_dim = 8, 2   # e.g. LunarLanderContinuous-v3

actor  = nn.Sequential(nn.Linear(obs_dim, 64), nn.Tanh(), nn.Linear(64, act_dim * 2))
critic = nn.Sequential(nn.Linear(obs_dim, 64), nn.Tanh(), nn.Linear(64, 1))

agent = TorchGaussianPPOAgent(
    actor=actor, critic=critic,
    optimizer=torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()), lr=3e-4
    ),
    action_dim=act_dim,
    hyperparams=HyperparamSet(params={
        "learning_rate": 3e-4, "clip_ratio": 0.2, "entropy_coef": 0.01,
        "gamma": 0.99, "gae_lambda": 0.95, "n_epochs": 10, "minibatch_size": 64,
    }, episode_id=0),
)
```

---

### DQN — Deep Q-Network

Discrete action spaces. Single Q-network; target network updated periodically.

```python
from tensor_optix.algorithms.tf_dqn import TFDQNAgent        # TensorFlow
from tensor_optix.algorithms.torch_dqn import TorchDQNAgent  # PyTorch (auto-detects CUDA)
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
    optimizer=tf.keras.optimizers.Adam(3e-4),
    hyperparams=HyperparamSet(params={
        "learning_rate": 3e-4, "gamma": 0.99,
        "epsilon": 1.0, "epsilon_min": 0.05, "epsilon_decay": 0.995,
        "batch_size": 64, "target_update_freq": 100,
        "replay_capacity": 100_000,
        "per_alpha": 0.0,        # 0 = uniform replay; 1 = full prioritized
        "per_beta":  0.4,        # IS correction exponent
        "n_step":    1,          # multi-step TD return length
    }, episode_id=0),
)
```

---

### SAC — Soft Actor-Critic

Continuous action spaces. Squashed Gaussian actor, twin critics, auto-entropy tuning.

```python
from tensor_optix.algorithms.tf_sac import TFSACAgent        # TensorFlow
from tensor_optix.algorithms.torch_sac import TorchSACAgent  # PyTorch (auto-detects CUDA)
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

### TD3 — Twin Delayed Deep Deterministic Policy Gradient

Continuous action spaces. Deterministic actor, twin critics, target policy smoothing, delayed actor updates.

```python
from tensor_optix.algorithms.torch_td3 import TorchTD3Agent  # PyTorch
```

**What it adds over SAC:**
- Deterministic policy (no entropy term) — lower variance, better on dense-reward locomotion
- Target policy smoothing: adds clipped Gaussian noise to target actions to prevent value over-fitting to narrow peaks
- Delayed actor updates: critic updated every step, actor updated every `policy_delay` steps (default 2)
- Twin-Q minimization over both critics (same as SAC)

```python
import torch.nn as nn
from tensor_optix.algorithms.torch_td3 import TorchTD3Agent

obs_dim, act_dim = 24, 4   # e.g. BipedalWalker-v3

actor   = nn.Sequential(nn.Linear(obs_dim, 256), nn.ReLU(), nn.Linear(256, act_dim), nn.Tanh())
critic1 = nn.Sequential(nn.Linear(obs_dim + act_dim, 256), nn.ReLU(), nn.Linear(256, 1))
critic2 = nn.Sequential(nn.Linear(obs_dim + act_dim, 256), nn.ReLU(), nn.Linear(256, 1))

agent = TorchTD3Agent(
    actor=actor, critic1=critic1, critic2=critic2,
    action_dim=act_dim,
    actor_optimizer=torch.optim.Adam(actor.parameters(), lr=3e-4),
    critic_optimizer=torch.optim.Adam(
        list(critic1.parameters()) + list(critic2.parameters()), lr=3e-4
    ),
    hyperparams=HyperparamSet(params={
        "learning_rate": 3e-4, "gamma": 0.99, "tau": 0.005,
        "batch_size": 256, "policy_delay": 2,
        "target_noise": 0.2, "noise_clip": 0.5,
        "expl_noise": 0.1, "replay_capacity": 1_000_000,
    }, episode_id=0),
)
```

---

### Rainbow DQN

Discrete action spaces. Combines six DQN improvements: Double Q, Dueling networks, Prioritized Experience Replay, n-step returns, Noisy Networks, and Distributional RL.

```python
from tensor_optix.algorithms.torch_rainbow_dqn import TorchRainbowDQNAgent, RainbowQNetwork
```

**What it implements (all six Rainbow components):**
- **Double Q** — target actions from online net, values from target net (reduces overestimation)
- **Dueling** — separate value and advantage streams; advantage-mean subtraction prevents identifiability issues
- **PER** — SumTree prioritized replay with IS-weight correction; priority exponent `α`, correction exponent `β`
- **n-step** — multi-step bootstrap targets; trades variance for bias, dramatically speeds up sparse-reward environments
- **Noisy Nets** — factorized Gaussian noise on linear layers replaces ε-greedy; exploration driven by learned uncertainty
- **Categorical / C51** — distributional Q as a 51-atom categorical distribution; KL loss instead of MSE

```python
obs_dim, n_actions = 8, 4

q_net = RainbowQNetwork(obs_dim=obs_dim, n_actions=n_actions,
                         hidden_size=256, n_atoms=51,
                         v_min=-10.0, v_max=10.0)

agent = TorchRainbowDQNAgent(
    q_network=q_net, n_actions=n_actions,
    optimizer=torch.optim.Adam(q_net.parameters(), lr=6.25e-5),
    hyperparams=HyperparamSet(params={
        "learning_rate": 6.25e-5, "gamma": 0.99,
        "n_step": 3,            # multi-step return length
        "per_alpha": 0.5,       # PER priority exponent
        "per_beta": 0.4,        # IS-weight correction exponent
        "batch_size": 32,
        "target_update_freq": 1000,
        "replay_capacity": 100_000,
    }, episode_id=0),
    n_atoms=51, v_min=-10.0, v_max=10.0,
)
```

---

### Recurrent PPO — LSTM Actor-Critic

Discrete action spaces. Hidden state carried across episode steps; handles partial observability and sequence-structured tasks.

```python
from tensor_optix.algorithms.torch_recurrent_ppo import TorchRecurrentPPOAgent
```

**What it adds over standard PPO:**
- `LSTMActorCritic` module — shared LSTM trunk, separate actor/critic heads
- Hidden state `(h, c)` carried across episode steps and reset at episode boundaries
- Rollout buffer stores per-step hidden states — no BPTT truncation artifacts
- Same clipped surrogate objective as PPO; hidden state dimension configurable

```python
from tensor_optix.algorithms.torch_recurrent_ppo import TorchRecurrentPPOAgent

obs_dim, n_actions, hidden_size = 4, 2, 64

agent = TorchRecurrentPPOAgent(
    obs_dim=obs_dim,
    n_actions=n_actions,
    hidden_size=hidden_size,
    hyperparams=HyperparamSet(params={
        "learning_rate": 3e-4, "clip_ratio": 0.2, "entropy_coef": 0.01,
        "vf_coef": 0.5, "gamma": 0.99, "gae_lambda": 0.95,
        "n_epochs": 4, "minibatch_size": 64,
    }, episode_id=0),
    device="auto",
)
# act() carries hidden state internally — same BaseAgent interface
action = agent.act(obs)
```

---

### JAX / Flax Adapter

Discrete action spaces. `FlaxAgent` base and `FlaxPPOAgent` implement the same `BaseAgent` interface as all PyTorch/TF algorithms using Flax NNX + optax.

```python
# pip install tensor-optix[jax]
from tensor_optix.adapters.jax.flax_agent    import FlaxAgent
from tensor_optix.adapters.jax.flax_evaluator import FlaxEvaluator
from tensor_optix.algorithms.flax_ppo         import FlaxPPOAgent
```

**Key properties:**
- `nnx.value_and_grad` differentiates through the combined actor+critic in one pass — no separate backward calls per head
- `nnx.Optimizer(model, optax.adam(lr), wrt=nnx.Param)` — only trainable `Param` variables are updated
- Weights serialised as a plain nested `dict` via `nnx.to_pure_dict(nnx.state(model))` and restored with `nnx.replace_by_pure_dict` + `nnx.update` — pickle-safe, version-stable
- Convergence-parity with `TorchPPOAgent` on CartPole-v1 (within 10%, verified by test suite)

```python
import gymnasium as gym
from tensor_optix import HyperparamSet
from tensor_optix.algorithms.flax_ppo import FlaxPPOAgent

env = gym.make("CartPole-v1")
obs_dim  = env.observation_space.shape[0]   # 4
n_actions = env.action_space.n              # 2

agent = FlaxPPOAgent(
    obs_dim=obs_dim,
    n_actions=n_actions,
    hyperparams=HyperparamSet(params={
        "learning_rate": 3e-4, "clip_ratio": 0.2, "entropy_coef": 0.01,
        "vf_coef": 0.5, "gamma": 0.99, "gae_lambda": 0.95,
        "n_epochs": 10, "minibatch_size": 64,
    }, episode_id=0),
    hidden_size=64,
    seed=0,
)

# Drop-in with any existing tensor-optix loop:
from tensor_optix import BatchPipeline, RLOptimizer
from tensor_optix.adapters.jax.flax_evaluator import FlaxEvaluator

pipeline = BatchPipeline(env=env, agent=agent, window_size=2048)
opt = RLOptimizer(agent=agent, pipeline=pipeline, evaluator=FlaxEvaluator())
opt.run()
```

---

### Distributed Training — Async Actor-Learner (IMPALA + V-trace)

IMPALA-style asynchronous actor-learner for PyTorch discrete policies. N actor processes collect trajectories in parallel using POSIX shared memory — learner weight updates are instantly visible to all actors with zero serialization overhead.

```python
# pip install tensor-optix[torch]
from tensor_optix.distributed import AsyncActorLearner, compute_vtrace_targets
```

**Architecture:**
```
shared memory   ┌─────────────────────────────────────────┐
                │  actor nn.Module  (share_memory=True)   │
                │  critic nn.Module (share_memory=True)   │
                └─────────────────────────────────────────┘
                     ↑ reads (lock-free)  ↑ optimizer.step()
  ┌─────────┐        │               ┌───────────────────┐
  │ Actor 0 │────────┘               │ Learner (main)    │
  │ Actor 1 │──── traj_queue ───────►│ V-trace targets   │
  │ ...     │                        │ gradient update   │
  │ Actor N │                        └───────────────────┘
  └─────────┘
```

**Off-policy correction (V-trace):**
```
ρ̄_t = min(ρ̄, π_θ(a_t|s_t) / π_μ(a_t|s_t))   ← clipped IS weight
v_t  = V(s_t) + δ_t + γ c̄_t (v_{t+1} − V(s_{t+1}))
```
Actors may lag behind the current learner policy (μ ≠ θ). V-trace corrects for this staleness so the learner can safely train on trajectories from any actor generation.

**Throughput:** 4 actors achieve ≥ 2× single-actor sample throughput in tests on CartPole-v1.

```python
import torch.nn as nn
import gymnasium as gym
from tensor_optix.distributed import AsyncActorLearner

obs_dim, n_actions = 4, 2

actor  = nn.Sequential(nn.Linear(obs_dim, 64), nn.Tanh(), nn.Linear(64, n_actions))
critic = nn.Sequential(nn.Linear(obs_dim, 64), nn.Tanh(), nn.Linear(64, 1))
optimizer = torch.optim.Adam(
    list(actor.parameters()) + list(critic.parameters()), lr=3e-4
)

learner = AsyncActorLearner(
    actor=actor,
    critic=critic,
    optimizer=optimizer,
    env_factory=lambda: gym.make("CartPole-v1"),
    n_actors=4,
    trajectory_len=64,
    gamma=0.99,
    rho_bar=1.0,   # V-trace IS clip ρ̄
    c_bar=1.0,     # V-trace trace clip c̄
    entropy_coef=0.01,
    seed=0,
)

stats = learner.run(max_steps=500_000)
# stats: {total_steps, total_updates, steps_per_second, elapsed}
print(f"Throughput: {stats['steps_per_second']:.0f} steps/s")
```

**Platform note:** uses `mp.get_context("fork")` — designed for Linux. On macOS or Windows (spawn start method), pass picklable factory callables.

---

### Live Terminal Dashboard

`RichDashboardCallback` renders a live terminal panel during training using the [Rich](https://github.com/Textualize/rich) library.

```python
# pip install rich   (optional dependency)
from tensor_optix.callbacks import RichDashboardCallback

opt = RLOptimizer(
    agent=agent,
    pipeline=pipeline,
    callbacks=[RichDashboardCallback(refresh_rate=2)],
)
opt.run()
```

The dashboard shows:
- Real-time score sparkline (last 20 eval points)
- Current loop state (ACTIVE / COOLING / DORMANT)
- Live hyperparameter values
- Smoothed score, best score, episode count
- Estimated convergence progress

No extra configuration required — pass it as a callback and it wires itself to the `LoopCallback` hooks automatically.

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

# Update stats from a collected rollout, then normalize in act():
obs_norm.update(obs_batch)
normed_obs = obs_norm.normalize(raw_obs)
```

`RewardNormalizer` is most useful with on-policy agents. Pass it directly to the agent — it handles the step/reset cycle at episode boundaries internally, so you don't have to track done flags manually:

```python
from tensor_optix.core.normalizers import RewardNormalizer

reward_norm = RewardNormalizer(gamma=0.99, clip=10.0)

agent = TFPPOAgent(
    actor=actor, critic=critic, optimizer=optimizer, hyperparams=hp,
    reward_normalizer=reward_norm,   # resets at done, normalizes before GAE
)
```

The agent calls `step(r)` and `reset()` for each reward in the window during `learn()`, then normalizes the full batch before GAE. This is a correctness requirement for multi-episode windows — forgetting to reset the running return across episode boundaries biases the return variance estimate.

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

### Adaptive Eval Scheduling

One of tensor-optix's core strengths is that **it decides when to evaluate** — not you, and not a fixed schedule.

The `BackoffScheduler` dynamically controls eval frequency based on learning progress:

```
  [loop] ep=   1  raw=  -67.2  smoothed= -150.6  state=ACTIVE  interval=1
  [loop] ep=   2  raw= -191.9  smoothed= -129.5  state=ACTIVE  interval=1
  [loop] ep=   4  raw= -106.8  smoothed= -100.5  state=ACTIVE  interval=2
  [loop] ep=   6  raw= -258.8  smoothed= -178.6  state=ACTIVE  interval=2
  [loop] ep=   8  raw=   65.7  smoothed=  -96.6  state=ACTIVE  interval=4
  [loop] ep=  16  raw=  208.1  smoothed=  136.9  state=ACTIVE  interval=8
  [loop] ep=  17  raw=  194.8  smoothed=  201.5  state=ACTIVE  interval=1  ← improvement detected, reset
```

- **No improvement** → interval doubles (2, 4, 8...) — more training time between evals, budget spent on learning
- **Improvement detected** → interval snaps back to 1 — dense eval to track the breakthrough
- **DORMANT** → loop stops, best weights restored automatically

Vanilla frameworks evaluate on a fixed schedule regardless of progress. tensor-optix spends the budget where it matters.

Enable verbose output to see this in action:

```python
opt = RLOptimizer(agent=agent, pipeline=pipeline, verbose=True)
```

### The Loop States

```
ACTIVE   → learning detected, evaluates frequently
COOLING  → no recent improvement, exponential backoff on eval frequency
DORMANT  → plateau confirmed — training is done, loop stops cleanly
```

**DORMANT = trained.** Not a fixed episode count — the system backs off evaluation geometrically until improvement stops, then declares convergence and restores the best known weights.

### Backoff Schedule

```
interval₀ = base_interval
intervalₙ = min(intervalₙ₋₁ × backoff_factor, max_interval_episodes)

Plateau detected when:  consecutive_no_improvement ≥ plateau_threshold
DORMANT declared when:  consecutive_no_improvement ≥ dormant_threshold
```

Every improvement resets the counter. The system accelerates evaluation when learning is happening, backs off when it isn't.

### Hyperparameter Optimization — Two Layers

tensor-optix provides two complementary levels of hyperparameter optimization. They are designed to compose: run `TrialOrchestrator` first to find a good starting configuration, then hand those params to `RLOptimizer` for the final full-budget run with SPSA online adaptation.

#### Layer 1 — Online Adaptation (SPSA, within a single run)

The default optimizer is `SPSAOptimizer` (Simultaneous Perturbation Stochastic Approximation). It adapts hyperparameters *during* a training run, episode by episode, responding to non-stationarity in real time.

```
Episode 1 (probe+):  sample Δ ∈ {−1, +1}ᴺ  (Rademacher vector)
                     apply θ⁺ = θ + c·Δ,  record score f⁺

Episode 2 (probe−):  apply θ⁻ = θ − c·Δ,  record score f⁻

Gradient estimate:   ĝᵢ = (f⁺ − f⁻) / (2·c·Δᵢ)   ← unbiased for all i simultaneously

Update:              x_new = clip(x + α·ĝ, 0, 1)   ← in normalized [0,1] param space
                     θ_new = denormalize(x_new)
```

All probing is done in a normalized `[0, 1]` parameter space — a fixed perturbation scale applies equally to a learning rate of `3e-4` and a clip ratio of `0.2` without manual tuning.

**Probe-aware degradation gating:** during probe episodes, score drops are self-inflicted perturbations, not genuine policy collapses. The loop skips degradation checks while the optimizer is probing.

`BackoffOptimizer` is also available for single-parameter cycling via two-phase finite difference:

```python
from tensor_optix.optimizers.backoff_optimizer import BackoffOptimizer

opt = RLOptimizer(
    agent=agent,
    pipeline=pipeline,
    optimizer=BackoffOptimizer(param_bounds={
        "learning_rate": (1e-5, 1e-2),
        "clip_ratio":    (0.05, 0.4),
    }),
)
```

#### Layer 2 — Trial-Level Search (TrialOrchestrator, across independent runs)

`TrialOrchestrator` runs N fully independent `RLOptimizer` trials, each with a different hyperparameter configuration, and uses **Optuna TPE** (Tree-structured Parzen Estimator) to select configurations. It is mathematically the same algorithm used by Stable-Baselines3, CleanRL, and RLlib for RL HPO sweeps.

**When to use it:** before committing to a long training run. Give each trial 10–20% of your final budget — enough to rank configurations, not enough for full training.

```python
from tensor_optix import TrialOrchestrator, RLOptimizer
# pip install optuna   (optional dependency)

def make_agent(params: dict) -> BaseAgent:
    net = tf.keras.Sequential([...])
    return TFPPOAgent(
        actor=net, critic=critic_net,
        optimizer=tf.keras.optimizers.Adam(params["learning_rate"]),
        hyperparams=HyperparamSet(params=params, episode_id=0),
    )

def make_pipeline() -> BasePipeline:
    return BatchPipeline(env_id="LunarLander-v3", n_steps=2048)

orchestrator = TrialOrchestrator(
    agent_factory=make_agent,
    pipeline_factory=make_pipeline,
    param_space={
        "learning_rate": ("log_float", 1e-4, 3e-3),   # log-uniform (good for lr)
        "clip_ratio":    ("float",     0.1,  0.3),
        "entropy_coef":  ("float",     0.001, 0.05),  # 0.001 floor prevents entropy collapse
        "gamma":         ("float",     0.95, 0.999),
    },
    n_trials=20,          # number of independent runs
    trial_steps=50_000,   # step budget per trial
)
best_params, best_score = orchestrator.run()

# Final full-budget run with the best configuration found
agent = make_agent(best_params)
optimizer = RLOptimizer(agent=agent, pipeline=make_pipeline(), max_episodes=500)
optimizer.run()
```

**Parameter space spec:**

| Spec | Description |
|---|---|
| `("float", lo, hi)` | Uniform float in `[lo, hi]` |
| `("log_float", lo, hi)` | Log-uniform float — use for learning rate, α |
| `("int", lo, hi)` | Uniform integer |
| `("log_int", lo, hi)` | Log-uniform integer — use for batch size, buffer size |
| `("categorical", v1, v2, ...)` | One of the listed values |

**How TPE works:**

TPE fits two kernel density estimates — `p(x | good)` over the top-k% of trials and `p(x | bad)` over the rest. The next configuration is chosen by maximising the acquisition ratio `p(x | good) / p(x | bad)`. This is equivalent to Bayesian optimisation with a non-parametric surrogate, without the O(n³) cost of Gaussian Process inference.

**MedianPruner:** after a short warmup, any trial whose score falls below the median of all trials at the same episode is terminated early. This cuts wall time by stopping clearly bad configurations without committing to a fixed bracket schedule.

Each trial gets an isolated checkpoint directory — no cross-trial interference. The underlying `Optuna` study is accessible via `orchestrator.study` for inspection, plotting, and storage to SQLite for distributed sweeps.

### Adaptive Improvement Margin

A gain of +0.001 on a signal with ±5 noise is not a real improvement. The loop computes an adaptive floor before crediting any score as a new best:

```
effective_margin = max(user_margin, noise_k × std(recent_scores))
```

`noise_k = 2.0` by default. Gains below 2σ of recent score noise do not reset backoff, preventing noise-level fluctuations from blocking convergence detection.

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

### External Checkpoint Scoring

The training window mean is a noisy signal — it diverges from the true policy quality that matters at deployment. Pass `checkpoint_score_fn` to decouple checkpoint saving from training noise:

```python
def external_eval(agent) -> float:
    # deterministic, fixed seed — measures true policy quality
    return eval_policy(agent, env_id="LunarLander-v3", seed=42, n_episodes=5)

opt = RLOptimizer(
    agent=agent,
    pipeline=pipeline,
    checkpoint_score_fn=external_eval,
)
```

The best checkpoint is selected by the external score. The training signal still drives convergence detection — only checkpoint saving uses the external eval.

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

Without a val pipeline, spawn noise adapts to the training improvement ratio:
```
σ = base_noise / (1 + improvement_ratio)
```
Exploit when improving, explore when stuck.

---

## Policy Evolution

### Automatic Rollback

When `run()` returns, the agent **always holds the best known weights** — whether stopped by convergence, budget, or manual `stop()`. This is unconditional and requires no configuration.

For mid-training rollback on degradation:

```python
opt = RLOptimizer(agent=agent, pipeline=pipeline, rollback_on_degradation=True)
```

### Spawn Budget — When Is Training Done?

```python
from tensor_optix import PolicyManager

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

### Built-in callbacks

```python
from tensor_optix.callbacks import WandbCallback, TensorBoardCallback, RichDashboardCallback

# Weights & Biases — logs every on_episode_end and on_improvement event
opt = RLOptimizer(agent=agent, pipeline=pipeline,
                  callbacks=[WandbCallback(project="my-project")])

# TensorBoard — writes to ./runs/ by default
opt = RLOptimizer(agent=agent, pipeline=pipeline,
                  callbacks=[TensorBoardCallback(log_dir="./runs/cartpole")])

# Live terminal dashboard (requires `pip install rich`)
opt = RLOptimizer(agent=agent, pipeline=pipeline,
                  callbacks=[RichDashboardCallback(refresh_rate=2)])
```

### Custom callbacks

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

    # ── Pipelines & components ────────────────────────────────────────────
    val_pipeline=val_pipeline,              # held-out pipeline; agent acts only, never learns
    evaluator=None,                         # default: TFEvaluator (or TorchEvaluator for PyTorch)
    optimizer=None,                         # default: SPSAOptimizer (all params in 2 episodes)

    # ── Checkpointing ─────────────────────────────────────────────────────
    checkpoint_dir="./checkpoints",         # where snapshots are saved
    max_snapshots=10,                       # keep only the N best snapshots on disk
    checkpoint_score_fn=None,               # callable(agent) → float; when set, checkpoint
                                            # selection uses this external eval instead of the
                                            # noisy training signal. Use for deterministic evals.

    # ── Budget ────────────────────────────────────────────────────────────
    max_episodes=None,                      # hard cap; None = run until DORMANT

    # ── Convergence detection (BackoffScheduler) ──────────────────────────
    base_interval=1,                        # eval every N episodes at the start
    backoff_factor=2.0,                     # multiply interval by this on each non-improvement
    max_interval_episodes=100,              # interval cap — never wait more than this
    plateau_threshold=5,                    # consecutive non-improvements → COOLING
    dormant_threshold=20,                   # consecutive non-improvements → DORMANT (stop)
                                            # off-policy agents (DQN, SAC) auto-set to 15 —
                                            # replay buffer scores are noisier than on-policy
    score_smoothing=2,                      # rolling mean window applied before comparing scores;
                                            # filters single-episode noise from convergence signals

    # ── Improvement margin ────────────────────────────────────────────────
    improvement_margin=0.0,                 # fixed minimum gain to count as improvement
    noise_k=2.0,                            # adaptive margin multiplier: effective_margin =
                                            # max(improvement_margin, noise_k × std(recent_scores))
                                            # increase if the loop stops too early on noisy envs
    score_window=20,                        # rolling window size for computing the noise floor

    # ── Degradation detection ─────────────────────────────────────────────
    rollback_on_degradation=False,          # restore best weights when degradation is detected
                                            # safe for on-policy (PPO); skip for off-policy (DQN/SAC)
    degradation_threshold=0.95,             # score must drop below best × threshold to fire
    min_episodes_before_dormant=0,          # don't declare DORMANT before this many evals;
                                            # auto-set per agent class when left at 0:
                                            # DQN=60 (epsilon decay floor timing),
                                            # SAC=30 (Q-function stabilisation)
    min_episodes_before_degradation=5,      # don't fire degradation before this many evals;
                                            # prevents false alarms during chaotic early training

    # ── Misc ──────────────────────────────────────────────────────────────
    callbacks=[],
    verbose=False,                          # print per-episode eval output (raw, smoothed, state)
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
│   ├── noisy_linear.py         # NoisyLinear — factorized Gaussian noise (Rainbow)
│   ├── policy_manager.py       # PolicyManager + PolicyManagerCallback
│   ├── ensemble_agent.py       # EnsembleAgent — multi-policy BaseAgent wrapper
│   ├── regime_detector.py      # RegimeDetector — score-based regime classification
│   ├── meta_controller.py      # MetaController — SPAWN/PRUNE/STOP/NO_OP decisions
│   ├── her_buffer.py           # HERReplayBuffer — Hindsight Experience Replay
│   └── diagnostic_controller.py# DiagnosticController — per-episode metric aggregation
├── optimizers/
│   ├── spsa_optimizer.py       # SPSAOptimizer — default, all params in 2 episodes
│   ├── backoff_optimizer.py    # BackoffOptimizer — single-param finite difference
│   ├── momentum_optimizer.py   # MomentumOptimizer — gradient-signed momentum
│   └── pbt_optimizer.py        # PBTOptimizer — population-based exploit/explore
├── adapters/
│   ├── tensorflow/
│   │   ├── tf_agent.py         # TFAgent — generic REINFORCE / A2C base
│   │   └── tf_evaluator.py     # TFEvaluator — default scorer
│   ├── pytorch/
│   │   ├── torch_agent.py      # TorchAgent — generic REINFORCE / A2C base
│   │   └── torch_evaluator.py  # TorchEvaluator — default scorer (no TF dep)
│   └── jax/                    # NEW — JAX/Flax adapter
│       ├── flax_agent.py       # FlaxAgent — REINFORCE base, nnx.Optimizer, pickle weights
│       └── flax_evaluator.py   # FlaxEvaluator — total reward scorer
├── algorithms/
│   ├── tf_ppo.py               # TFPPOAgent — clipped surrogate, GAE-λ, n-epoch minibatch
│   ├── tf_ppo_continuous.py    # TFGaussianPPOAgent — continuous PPO, squashed Gaussian
│   ├── tf_dqn.py               # TFDQNAgent — PER replay, target net, ε-greedy
│   ├── tf_sac.py               # TFSACAgent — twin critics, squashed Gaussian, auto-α
│   ├── tf_td3.py               # TFTDDAgent — deterministic policy, delayed actor, target smoothing
│   ├── torch_ppo.py            # TorchPPOAgent (CUDA auto-detect)
│   ├── torch_ppo_continuous.py # TorchGaussianPPOAgent — continuous PPO (CUDA auto-detect)
│   ├── torch_dqn.py            # TorchDQNAgent (CUDA auto-detect)
│   ├── torch_sac.py            # TorchSACAgent (CUDA auto-detect)
│   ├── torch_td3.py            # TorchTD3Agent — TD3: twin critics, delayed actor, noise smoothing
│   ├── torch_recurrent_ppo.py  # TorchRecurrentPPOAgent — LSTM actor-critic, stateful act()
│   ├── torch_rainbow_dqn.py    # TorchRainbowDQNAgent — Double/Dueling/PER/n-step/Noisy/C51
│   └── flax_ppo.py             # FlaxPPOAgent — PPO via flax.nnx + optax (JAX backend)
├── distributed/                # NEW — async actor-learner
│   ├── async_learner.py        # AsyncActorLearner — IMPALA-style, POSIX shared memory
│   └── vtrace.py               # compute_vtrace_targets — pure-numpy V-trace IS correction
├── callbacks/
│   ├── wandb_callback.py       # WandbCallback — logs to Weights & Biases
│   ├── tensorboard_callback.py # TensorBoardCallback — logs to TensorBoard
│   └── rich_dashboard.py       # RichDashboardCallback — live terminal panel (requires rich)
└── pipeline/
    ├── batch_pipeline.py       # Continuous stepping, fixed windows
    ├── live_pipeline.py        # Real-time streaming with reconnect
    └── vector_pipeline.py      # Parallel envs via gymnasium.vector
```

| Component | Responsibility |
|-----------|---------------|
| `TFPPOAgent` / `TorchPPOAgent` | Discrete PPO — GAE, clipping, entropy, n-epoch minibatch |
| `TFGaussianPPOAgent` / `TorchGaussianPPOAgent` | Continuous PPO — squashed Gaussian actor, same PPO core |
| `TFDQNAgent` / `TorchDQNAgent` | DQN with PER replay, n-step returns, target net, ε-greedy |
| `TFSACAgent` / `TorchSACAgent` | SAC — twin critics, auto-entropy, soft target updates |
| `TFTDDAgent` / `TorchTD3Agent` | TD3 — deterministic actor, twin critics, delayed updates, target smoothing |
| `TorchRecurrentPPOAgent` | LSTM actor-critic PPO — hidden state carried across steps, handles partial observability |
| `TorchRainbowDQNAgent` | Rainbow DQN — Double/Dueling/PER/n-step/Noisy Nets/C51 (all six improvements) |
| `FlaxPPOAgent` | PPO via Flax NNX + optax — XLA-compiled, convergence-parity with TorchPPOAgent |
| `FlaxAgent` | Base agent for any `nnx.Module` — REINFORCE update, pickle-safe weight I/O |
| `AsyncActorLearner` | IMPALA-style N-actor learner — POSIX shared memory, V-trace off-policy correction |
| `compute_vtrace_targets` | Pure-numpy V-trace IS correction (Espeholt et al. 2018) |
| `LoopController` | State machine, episode orchestration, eval, checkpoint |
| `BackoffScheduler` | Adaptive eval scheduling + convergence detection |
| `CheckpointRegistry` | Snapshot storage, best-checkpoint manifest |
| `SPSAOptimizer` | SPSA — all N params in 2 episodes, normalized space (online, within a run) |
| `BackoffOptimizer` | Two-phase finite difference hyperparameter tuning |
| `PBTOptimizer` | Population-based exploit/explore hyperparameter tuning |
| `TrialOrchestrator` | Optuna TPE trial-level HPO — N independent runs, MedianPruner (requires `optuna`) |
| `PolicyManager` | Rollback, spawn, prune, boost, ensemble weights, adaptive noise |
| `MetaController` | Rule-based SPAWN/PRUNE/STOP/NO_OP decisions |
| `EnsembleAgent` | Weighted-average action combining across multiple agents |
| `RegimeDetector` | Score-based regime classification (trending / ranging / volatile) |
| `WandbCallback` | Logs scores, hyperparams, and diagnostics to Weights & Biases |
| `TensorBoardCallback` | Logs to TensorBoard SummaryWriter |
| `RichDashboardCallback` | Live terminal panel — sparkline, state, hyperparams (requires `rich`) |
| `VectorBatchPipeline` | Parallel environment rollouts via gymnasium.vector |
| `ObsNormalizer` | Online running mean/std observation normalization |
| `RewardNormalizer` | Return-std reward scaling |

---

## Common Pitfalls & Best Practices

### Device management

Every built-in Torch agent (`TorchPPOAgent`, `TorchGaussianPPOAgent`, `TorchDQNAgent`, `TorchSACAgent`) accepts a `device` parameter and moves its networks there on construction. The default is `"auto"`, which selects CUDA if available.

```python
agent = TorchPPOAgent(actor=actor, critic=critic, optimizer=opt,
                      hyperparams=hp, device="cuda")   # or "cpu", "auto"
```

The base `TorchAgent` adapter now also accepts `device="auto"` and applies it consistently in `act()` and `load_weights()`. If you subclass `TorchAgent` directly, pass `device` to `super().__init__()` — otherwise obs tensors and loaded checkpoints default to CPU even on a CUDA machine.

**Watch out:** constructing the optimizer _before_ calling `.to(device)` on the model is safe because optimizers hold references to parameter tensors, not copies. But creating the optimizer _after_ `agent.load_weights()` restores weights to the wrong device can leave parameters split between CPU and GPU, which causes a silent slowdown rather than an error.

### Ensemble memory on GPU

Spawning agents with `PolicyManager.spawn_variant()` or `agent_factory` mode creates new networks on the target device. Calling `prune()` removes agents from the ensemble and automatically calls `agent.teardown()` on each removed agent, which moves its networks to CPU and calls `torch.cuda.empty_cache()`.

If you remove agents from the ensemble by any other means (e.g., rebuilding `_ensemble` manually), call `teardown()` yourself:

```python
removed = pm.prune(bottom_k=2)   # teardown() is called automatically

# If removing manually:
agent.teardown()
```

For long PBT-style runs with frequent spawning, monitor GPU memory with `torch.cuda.memory_allocated()`. If memory grows despite pruning, the likely cause is optimizer state — gradient moments accumulate per parameter. Re-creating the optimizer on each spawn (as the built-in `agent_factory` pattern does) avoids this.

### On-policy vs. off-policy rollback

`rollback_on_degradation=True` is safe for PPO but harmful for DQN and SAC. Off-policy agents accumulate experience in a replay buffer across many policies. Rolling back weights without clearing the buffer means the restored policy immediately trains on transitions it never generated — corrupted Bellman targets drag it back down.

The framework handles this automatically: any agent where `is_on_policy` returns `False` skips the weight rollback even when `rollback_on_degradation=True`. If you write a custom off-policy agent, override the property:

```python
@property
def is_on_policy(self) -> bool:
    return False
```

### Wiring PolicyManager early stopping

`PolicyManager.as_callback()` returns a `PolicyManagerCallback` that stops training when the spawn budget is exhausted — but only if you wire the stop function:

```python
pm_cb = pm.as_callback(agent, agent_factory=my_factory)
rl_opt = RLOptimizer(...)
pm_cb.set_stop_fn(rl_opt.stop)   # required — without this, training runs the full budget
rl_opt.add_callback(pm_cb)
rl_opt.run()
```

Without `set_stop_fn`, the callback prints the training report when the budget runs out but cannot halt the loop. Training continues until `max_episodes` is reached.

For the factory-mode PPO path (where `agent_factory` is passed to `RLOptimizer` and `pm_cb` is created inside the factory), wire the stop function inside the factory — `rl_opt` is already bound in the enclosing scope by the time the factory is called:

```python
def agent_factory_full(params):
    agent = make_agent(params)
    pm_cb = pm.as_callback(agent, agent_factory=lambda: make_agent(params))
    pm_cb.set_stop_fn(rl_opt.stop)   # rl_opt is bound before run() calls this factory
    rl_opt.add_callback(pm_cb)
    return agent

rl_opt = RLOptimizer(agent_factory=agent_factory_full, ...)
rl_opt.run()
```

### Checkpoint directory hygiene

Each run writes checkpoints to `checkpoint_dir`. If you reuse the same directory across restarts without clearing it, `CheckpointRegistry` will load stale snapshots from a previous run and roll back to them during training. Either pass a unique directory per run (include seed and timestamp) or call `shutil.rmtree(ckpt_dir, ignore_errors=True)` at the start of each run.

### State dict key mismatches during weight averaging / spawning

`average_weights()` and `load_weights()` use PyTorch `state_dict` keys. If the architecture passed to a spawned agent shell differs from the one that was checkpointed (different layer names, sizes, or number of layers), `load_state_dict()` will raise a `RuntimeError` with a key mismatch message. The framework does not catch this — it is user responsibility to pass a compatible shell. The safest pattern is to use the same `agent_factory` for both the primary agent and all spawned variants.

---

## Math & Science Reference

### SPSA Gradient Estimate (`SPSAOptimizer`)

```
Δ ~ Rademacher({−1, +1}ᴺ)          ← simultaneous perturbation vector

θ⁺ = θ + c·Δ   →   f⁺ = score(θ⁺)
θ⁻ = θ − c·Δ   →   f⁻ = score(θ⁻)

ĝᵢ = (f⁺ − f⁻) / (2·c·Δᵢ)         ← unbiased estimator for all i

x  = (θ − lo) / (hi − lo)           ← normalize to [0,1]
x' = clip(x + α·ĝ, 0, 1)
θ' = lo + x' · (hi − lo)            ← denormalize
```

Spall (1992) proved that this two-measurement estimator is unbiased and converges at the same asymptotic rate as finite difference with N measurements. Normalizing to `[0,1]` before updating ensures a fixed perturbation scale `c` applies equally to parameters of any magnitude.

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

### V-trace Off-Policy Correction (`compute_vtrace_targets` / `AsyncActorLearner`)

Espeholt et al. 2018 ([IMPALA](https://arxiv.org/abs/1802.01561)). Corrects actor-lag bias when actors run an older policy μ while the learner updates to θ.

```
ρ_t   = π_θ(a_t|s_t) / π_μ(a_t|s_t)          ← IS ratio
ρ̄_t  = min(ρ̄, ρ_t)                            ← clipped IS weight
c̄_t  = min(c̄, ρ_t)                            ← trace coefficient

δ_t   = ρ̄_t · (r_t + γ·(1−done_t)·V(s_{t+1}) − V(s_t))

v_t   = V(s_t) + δ_t + γ·(1−done_t)·c̄_t·(v_{t+1} − V(s_{t+1}))
                                               ← backward recursion

A_t   = ρ̄_t · (r_t + γ·(1−done_t)·v_{t+1} − V(s_t))
                                               ← policy gradient advantage
```

With ρ̄ = c̄ = 1 and synchronous actors (μ = θ), reduces to standard on-policy GAE. The clip ρ̄ bounds the maximum IS weight, preventing gradient spikes from stale trajectories.

### TD3 Target Policy Smoothing (`TorchTD3Agent` / `TFTDDAgent`)

```
a_noise = clip(N(0, σ), −c, c)              ← clipped Gaussian noise
a_target = clip(π_θ_tgt(s') + a_noise, a_lo, a_hi)

y = r + γ · min(Q_tgt1(s', a_target), Q_tgt2(s', a_target))
```

Smoothing prevents the critic from over-fitting to narrow deterministic action peaks. The actor is updated every `policy_delay` critic steps (default 2), giving the critic time to stabilise before actor gradients are computed.

### Rainbow Categorical Distribution (`TorchRainbowDQNAgent`)

Distributional Bellman target (Bellemare et al. 2017):
```
atoms: z_i ∈ {v_min + i·Δz},   i = 0..N-1,   Δz = (v_max − v_min) / (N-1)

y     = r + γ · z_i   (projected onto support)

L     = KL(target_dist ‖ Q_dist(s, a))       ← cross-entropy loss
```
The 51-atom categorical distribution represents the full return distribution rather than its mean, enabling value-based agents to reason about risk.

---

## License

MIT — Copyright (c) 2026 sup3rus3r
