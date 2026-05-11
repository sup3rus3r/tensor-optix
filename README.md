# tensor-optix

tensor-optix is a training loop framework with statistical convergence control, online hyperparameter optimisation, and an optional neuroevolution subsystem for dynamic topology.

The core loop runs your agent against a pipeline, maintains a separate validation signal, and manages four states: ACTIVE, COOLING, DORMANT, and watchdog shutdown. Convergence is detected using a corrected t-test on the smoothed score slope plus lag-1 autocorrelation, not a fixed patience counter. Hyperparameters are updated every episode via SPSA gradient estimates, with automatic routing to momentum-based or sign-only updates depending on the autocorrelation structure of the score landscape. Checkpointing and rollback are driven by the validation signal only, never training score. On DORMANT, a MetaController evaluates the generalization gap and its slope to decide between spawning a policy variant, pruning the ensemble, or stopping.

The neuroevo subsystem (`pip install tensor-optix[neuroevo]`) represents the policy as a `NeuronGraph`: a mutable directed graph of heterogeneous scalar neurons (point, GRU, LSTM, trainable-GRU, trainable-LSTM) with variable-delay edges. Weights for excitatory and inhibitory neurons follow softplus Dale's Law: raw parameter θ maps to `softplus(θ)` (excitatory) or `-softplus(θ)` (inhibitory), eliminating gradient dead zones at weight boundaries. `TopologyController` runs as a loop callback and evaluates three independent signals per episode: improvement slope significance, residual autocorrelation structure, and gradient utilization across hidden neurons. All three must cross their thresholds before a grow operation fires. Pruning is by importance score (incident edge weight magnitude times mean absolute activation). Merging is by Pearson correlation of per-episode activation histories. After every structural mutation the controller calls `graph.invalidate_compile()` to reset for the new topology. `TopologyAwareAdam` resets momentum state for parameters affected by any structural change. `BrainNetwork` composes multiple named `NeuronGraph` regions with sparse learnable inter-region edges. `HebbianHook` accumulates co-activation products across each episode and applies an Oja-style weight update after the PPO gradient step. `NeuromodulatorSignal` maps a `RegimeDetector` output (trending / ranging / volatile) to simultaneous changes in Hebbian learning rate, entropy coefficient, and topology grow/prune thresholds.

The entire system, including neuroevo, is accessed through a six-method `BaseAgent` interface:

```python
class BaseAgent(ABC):
    def act(self, observation) -> any: ...
    def learn(self, episode_data: EpisodeData) -> dict: ...
    def get_hyperparams(self) -> HyperparamSet: ...
    def set_hyperparams(self, hyperparams: HyperparamSet) -> None: ...
    def save_weights(self, path: str) -> None: ...
    def load_weights(self, path: str) -> None: ...
```

Fifteen agents are included across PyTorch, TensorFlow, and JAX/Flax. Bring your own by implementing the interface above.

---

## Install

```bash
# Core loop only, no algorithm implementations
pip install tensor-optix

# PyTorch algorithms
pip install tensor-optix[torch]

# TensorFlow algorithms
pip install tensor-optix[tensorflow]

# JAX/Flax
pip install tensor-optix[jax]

# Neuroevo (requires torch)
pip install tensor-optix[neuroevo]

# GPU (Linux/WSL2, CUDA 12.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install tensor-optix[torch]

# All frameworks and environment extras
pip install tensor-optix[all]
```

Environment extras: `[box2d]`, `[atari]`, `[mujoco]`  
Logging extras: `[wandb]`, `[tensorboard]`  
Export: `[onnx]`

---

## Algorithms

All agents implement `BaseAgent` and are interchangeable with `RLOptimizer`.

### PyTorch

| Agent | Algorithm | Action Space |
|---|---|---|
| `TorchPPOAgent` | PPO + GAE-λ | Discrete |
| `TorchGaussianPPOAgent` | PPO | Continuous |
| `TorchRecurrentPPOAgent` | PPO + GRU/LSTM hidden state | Discrete |
| `TorchDQNAgent` | DQN + PER + n-step returns | Discrete |
| `TorchRainbowDQNAgent` | Rainbow DQN (NoisyNet, distributional, PER, n-step, dueling, double) | Discrete |
| `TorchSACAgent` | SAC, twin Q-critics, automatic entropy tuning | Continuous |
| `TorchTD3Agent` | TD3 | Continuous |

### TensorFlow

`TFPPOAgent`, `TFGaussianPPOAgent`, `TFDQNAgent`, `TFSACAgent`, `TFTDDAgent`

### JAX/Flax

`FlaxPPOAgent`

### Auto-selection

`make_agent` inspects the environment action space and returns a fully constructed agent. Pass an algorithm name as the first argument, or let it be inferred.

```python
from tensor_optix import make_agent
import gymnasium as gym

env = gym.make("LunarLanderContinuous-v3")
agent = make_agent(env)                         # -> TorchSACAgent
agent = make_agent("SAC", env)                  # same
agent = make_agent(env, framework="tf")         # -> TFSACAgent
agent = make_agent(env, deterministic=True)     # -> TorchTD3Agent

# Neuroevo path: NeuronGraph + GraphAgent with Hebbian + TopologyController
agent = make_agent("SAC", env, neuroevo=True)
agent = make_agent("PPO", env, neuroevo=True, graph_hidden=16, hebbian_lr=1e-3)
```

Neuroevo options: `graph_in` (input neurons, default `min(obs_dim, 16)`), `graph_hidden` (GRU neurons, default 8), `graph_out` (output neurons, default `act_dim + 1`), `hebbian_lr`, `hebbian_decay`, `grow_cooldown`.

---

## One-line training with Optimizer

`Optimizer` wraps `RLOptimizer` with sensible defaults. It auto-computes `window_size`, wires neuroevo callbacks, and activates SPSA when the agent has `default_param_bounds`.

```python
from tensor_optix import make_agent, Optimizer
import gymnasium as gym

env   = gym.make("HumanoidStandup-v5")
agent = make_agent("SAC", env, neuroevo=True)
opt   = Optimizer(agent, env)
opt.run()

# Vectorized: 8 parallel envs
opt = Optimizer(agent, lambda: gym.make("CartPole-v1"), n_envs=8)
opt.run()
```

`optimal_window_size(env, algorithm)` computes the window size formula used internally: `clip(k * mean_episode_steps, 512, 8192)` where k=4.0 for on-policy (PPO) and k=1.0 for off-policy (SAC/TD3).

```python
from tensor_optix import optimal_window_size
window = optimal_window_size(env, "PPO")  # e.g. 2000 for CartPole
```

---

## Pipelines

A pipeline steps an environment (or data source), collects `EpisodeData`, and yields it to the agent. Three implementations are provided.

```python
from tensor_optix import BatchPipeline, LivePipeline, VectorBatchPipeline

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

---

## The loop

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
    dormant_threshold=10,            # consecutive episodes without improvement -> DORMANT
    min_episodes_before_dormant=50,  # statistical warmup before convergence detection activates
)

opt.run()
opt.best_snapshot   # -> PolicySnapshot: best weights + EvalMetrics + HyperparamSet
```

Loop state transitions: `ACTIVE` -> `COOLING` -> `DORMANT` -> watchdog shutdown or policy spawn.

On shutdown the loop restores best-known weights, not the final checkpoint.

---

## Hyperparameter optimisation

All optimisers operate in normalised [0, 1] parameter space and update every episode. No restarts required.

```python
from tensor_optix.optimizers import SPSAOptimizer, AdaptiveOptimizer

# SPSA: Rademacher perturbation vector, two-episode gradient estimate
optimizer = SPSAOptimizer(
    param_bounds={"learning_rate": (1e-4, 3e-3), "clip_ratio": (0.1, 0.3)},
    log_params={"learning_rate"},   # log-space normalisation for params spanning orders of magnitude
)

# AdaptiveOptimizer: routes between SPSA, Momentum, Backoff, and PBT
# based on lag-1 autocorrelation of the score stream and relative performance gap
optimizer = AdaptiveOptimizer(param_bounds={...})

opt = RLOptimizer(agent=agent, pipeline=pipeline, optimizer=optimizer)
```

| Optimizer | Routing condition |
|---|---|
| `SPSAOptimizer` | i.i.d. score noise, no autocorrelation structure |
| `MomentumOptimizer` | Positive lag-1 autocorrelation (smooth landscape) |
| `BackoffOptimizer` | Negative lag-1 autocorrelation (oscillating landscape, sign-only updates) |
| `PBTOptimizer` | Score below 20th percentile of history (exploit checkpoint population) |
| `AdaptiveOptimizer` | Routes automatically based on the two signals above |

### Trial-level search

`TrialOrchestrator` runs N independent short trials via Optuna TPE before the main run, then warm-starts from the best trial's weights and config.

```python
from tensor_optix import TrialOrchestrator

orch = TrialOrchestrator(
    agent_factory=make_agent,
    pipeline_factory=make_pipeline,
    param_space={
        "learning_rate": ("log_float", 1e-4, 3e-3),
        "clip_ratio":    ("float",     0.1,  0.3),
        "batch_size":    ("int",       32,   512),
    },
    n_trials=20,
    trial_episodes=50,
)
best_config = orch.run()
```

---

## Ensemble and policy evolution

`PolicyManager` runs as a loop callback. On each DORMANT event, `MetaController` evaluates the generalization gap (train minus val, normalised) and its slope, and the validation improvement rate, then issues one of: SPAWN, PRUNE, or STOP. Spawned variants are cloned from the best checkpoint with perturbed hyperparameters. `EnsembleAgent` wraps all active variants behind the `BaseAgent` interface, with weighted action averaging.

```python
from tensor_optix import PolicyManager

pm = PolicyManager(registry, max_spawns=4)
cb = pm.as_callback(agent, agent_factory=make_agent)
cb.set_stop_fn(opt.stop)
opt.add_callback(cb)
opt.run()
```

---

## Callbacks

```python
from tensor_optix.callbacks import RichDashboardCallback, WandbCallback, TensorBoardCallback

opt.add_callback(RichDashboardCallback())        # Rich live terminal panel
opt.add_callback(WandbCallback(project="run"))
opt.add_callback(TensorBoardCallback(log_dir="./tb"))
```

Custom callbacks subclass `LoopCallback` and override any of:

```python
class LoopCallback:
    def on_loop_start(self) -> None: ...
    def on_loop_stop(self) -> None: ...
    def on_episode_end(self, episode_id: int, eval_metrics) -> None: ...
    def on_improvement(self, snapshot) -> None: ...
    def on_plateau(self, episode_id: int, state) -> None: ...
    def on_dormant(self, episode_id: int) -> None: ...
    def on_degradation(self, episode_id: int, eval_metrics) -> None: ...
    def on_hyperparam_update(self, old: dict, new: dict) -> None: ...
```

---

## Distributed training (IMPALA + V-trace)

`AsyncActorLearner` implements IMPALA-style async actor-learner. N actor subprocesses read weights from shared memory (lock-free), collect trajectories, and push them to a queue. The learner dequeues trajectories, applies V-trace importance-sampling correction, and writes updated weights back to shared memory.

```python
from tensor_optix.distributed import AsyncActorLearner

learner = AsyncActorLearner(
    actor=actor,
    critic=critic,
    optimizer=optimizer,
    env_factory=lambda: gym.make("ALE/Pong-v5"),
    n_actors=8,
    trajectory_len=64,
)
stats = learner.run(max_steps=10_000_000)
# stats["steps_per_second"] -> ~4x single-process throughput on CPU
```

---

## Neuroevo

`NeuronGraph` is a mutable directed graph of scalar neurons with variable-delay edges. `GraphAgent` wraps it as a `BaseAgent` with PPO-style weight learning. `TopologyController` mutates the graph live during the training loop.

```bash
pip install tensor-optix[neuroevo]
```

### Graph construction

```python
from tensor_optix.neuroevo import NeuronGraph, GraphAgent, GRUNeuron, LSTMNeuron

graph = NeuronGraph()

for _ in range(4):
    graph.add_neuron(role="input", activation="linear")
for _ in range(8):
    graph.add_neuron(role="hidden", activation="tanh")
    # or: graph.add_neuron(role="hidden", neuron=GRUNeuron())
    # or: graph.add_neuron(role="hidden", neuron=LSTMNeuron())
graph.add_neuron(role="output", activation="linear")  # last output neuron is the value head

graph.add_edge(src_id, dst_id, weight=0.0, delay=0)   # feedforward (d=0)
graph.add_edge(src_id, dst_id, weight=0.0, delay=1)   # recurrent (d>=1, reads from history buffer)

agent = GraphAgent(graph, obs_dim=4, n_actions=2)
```

All edges initialise at weight=0.0, which is function-preserving at insertion time.

### Neuron types

| Type | Hidden state | Gradient through state |
|---|---|---|
| `Neuron` | None (point neuron) | N/A |
| `GRUNeuron` | Scalar h, detached | No |
| `LSTMNeuron` | Scalar h and c, detached | No |
| `TrainableGRUNeuron` | Scalar h, not detached | Yes, up to chunk_len steps |
| `TrainableLSTMNeuron` | Scalar h and c, not detached | Yes, up to chunk_len steps |

All types implement the same protocol: `step()`, `importance()`, `can_merge_with()`, `make_relay()`, `split_copy()`. `NeuronGraph` and `TopologyController` are type-blind.

### Trainable recurrent neurons

`TrainableGRUNeuron` and `TrainableLSTMNeuron` set `is_recurrent = True`. `RecurrentGraphAgent` detects this flag and switches from shuffled-minibatch PPO to sequential chunk training with truncated BPTT.

```python
from tensor_optix.neuroevo import TrainableGRUNeuron, TrainableLSTMNeuron, RecurrentGraphAgent

graph = NeuronGraph()
# ... input and output neurons ...
graph.add_neuron(role="hidden", neuron=TrainableGRUNeuron())
graph.add_neuron(role="hidden", neuron=TrainableLSTMNeuron())

agent = RecurrentGraphAgent(
    graph, obs_dim=4, n_actions=2,
    hyperparams=HyperparamSet(params={"chunk_len": 64}),
)
# Falls back to standard shuffled-minibatch PPO if no recurrent neurons are present
```

### Topology controller

```python
from tensor_optix.neuroevo import TopologyController

controller = TopologyController.for_graph(
    graph=graph,
    scheduler=opt._scheduler,
    grow_grad_threshold=0.7,         # fraction of hidden neurons with |grad| > eps required to grow
    prune_neuron_threshold=1e-4,     # importance score below this -> prune candidate
    prune_edge_threshold=1e-3,       # |weight| below this for prune_edge_patience episodes -> prune
    merge_similarity_threshold=0.95, # Pearson correlation threshold for merge
)
opt.add_callback(controller)
opt.run()
```

Grow fires only when all three signals agree:
1. Improvement slope t-test is not significant (gradient updates are not making progress)
2. Score residuals have significant autocorrelation (capacity is underutilised)
3. Gradient utilization exceeds `grow_grad_threshold` (existing neurons are saturated)

For multi-region graphs, use `TopologyController.for_brain(brain, scheduler=...)`. Each region gets independent signal buffers and cooldown timers.

### BrainNetwork

```python
from tensor_optix.neuroevo import BrainNetwork, TopologyController

brain = BrainNetwork()
brain.add_region("sensory",   sensory_graph)
brain.add_region("memory",    memory_graph)
brain.add_region("executive", executive_graph)

brain.add_pathway("sensory",  "memory",    n_connections=8, delay=1)
brain.add_pathway("memory",   "executive", n_connections=8, delay=0)

controller = TopologyController.for_brain(brain, scheduler=opt._scheduler)
```

Inter-region edges are learnable parameters. Regions are executed in topological order each forward pass.

### Hebbian learning

`HebbianHook` applies an Oja-style local weight update after each episode. The rule is: `dw = eta * mean_t(h_pre * h_post) - lambda * w`. Call `record()` after each `act()` to accumulate co-activation products, then `apply()` after the PPO gradient step.

```python
from tensor_optix.neuroevo import HebbianHook

hook = HebbianHook(graph, hebbian_lr=1e-3, weight_decay=1e-4)

for step in episode:
    action = agent.act(obs)
    hook.record()
    obs, reward, done, _ = env.step(action)

agent.learn(episode_data)
hook.apply()
hook.reset()
```

Use `HebbianHook.from_brain(brain, ...)` for `BrainNetwork` graphs.

### Neuromodulation

`NeuromodulatorSignal` takes a `RegimeDetector` classification and applies coordinated parameter changes across `HebbianHook`, `GraphAgent`, and `TopologyController` simultaneously.

```python
from tensor_optix.neuroevo import NeuromodulatorSignal
from tensor_optix.core import RegimeDetector

detector = RegimeDetector()
mod = NeuromodulatorSignal(hook=hook, agent=agent, controller=controller)

regime = detector.detect(metrics_history)  # "trending" | "ranging" | "volatile"
mod.apply(regime)
# trending  -> lower entropy coefficient, reduce hebbian_lr (consolidate)
# volatile  -> raise entropy coefficient, raise grow thresholds (explore)
# ranging   -> raise hebbian_lr (local plasticity)
```

### Dale's Law

```python
# clamp mode (default): outgoing weights clamped post-step
graph = NeuronGraph(dale_mode="clamp")
graph.add_neuron(role="hidden", activation="relu", cell_type="excitatory")  # weights >= 0
graph.add_neuron(role="hidden", activation="tanh", cell_type="inhibitory")  # weights <= 0

# softplus mode: raw parameter theta, effective weight = softplus(theta) * sign
# gradient-safe, no dead zone at the clamp boundary
# enforce_dale() is a no-op in this mode
graph = NeuronGraph(dale_mode="softplus")
w = graph.effective_weight(edge_id)  # reads post-softplus value
```

### TopologyAwareAdam

Drop-in Adam replacement that resets (m, v) momentum state for parameters touched by a grow, prune, or merge operation. Stale momentum estimates from before a structural change would otherwise corrupt the first update on modified parameters.

```python
from tensor_optix.neuroevo import TopologyAwareAdam

optimizer = TopologyAwareAdam(graph.parameters(), lr=3e-4)
optimizer.notify_topology_change(new_params)  # call after any topology mutation
```

### Compiled forward

`NeuronGraph` runs in **eager mode by default**. Because `_raw_forward` mutates Python-side neuron state (`neuron._current`, `push_history`), `torch.compile` cannot safely trace it without replaying those side effects. The default eager path is safe for training, recurrent neurons, and dynamic topologies.

If the topology is static and you manage neuron state externally, you can opt in to a compiled forward:

```python
graph.compile_forward()   # one-time call; re-call after any topology mutation
```

The backend is selected automatically: `inductor` on Linux/macOS, `aot_eager` on Windows. Has no effect on PyTorch < 2.0.

`TopologyController` calls `graph.invalidate_compile()` after every grow, prune, and merge. If you mutate the graph directly outside the controller, call it yourself:

```python
graph.add_edge(src_id, dst_id, weight=0.0, delay=1)
graph.invalidate_compile()
```

`invalidate_compile()` rebuilds the matrix cache and resets to eager mode. If `compile_forward()` was previously called, it also evicts stale Dynamo kernels (`torch._dynamo.reset()`) and recompiles — this reset is process-global, so all `NeuronGraph` instances in the process retrace on their next forward call.

---

## Core utilities

### Normalizers

Online Welford mean/variance. `ObsNormalizer` normalises observations. `RewardNormalizer` divides rewards by return standard deviation (not reward std), preserving sign.

```python
from tensor_optix.core.normalizers import ObsNormalizer, RewardNormalizer

obs_norm = ObsNormalizer(shape=(obs_dim,))
obs_norm.update(obs_batch)
normalized = obs_norm.normalize(obs)

rew_norm = RewardNormalizer()
```

### Hindsight Experience Replay

Wraps `PrioritizedReplayBuffer` with episode-level goal relabeling. Supports `future` (default), `final`, and `episode` relabeling strategies.

```python
from tensor_optix.core.her_buffer import HERBuffer

her = HERBuffer(obs_dim=obs_dim, act_dim=act_dim, goal_dim=goal_dim, strategy="future", k=4)
her.store_episode(obs_list, act_list, rew_list, next_obs_list, done_list,
                  achieved_goals, compute_reward_fn)
obs_b, act_b, rew_b, next_b, done_b, weights, idx, n = her.sample(batch_size)
```

### Checkpoint registry

```python
from tensor_optix.core.checkpoint_registry import CheckpointRegistry

registry = CheckpointRegistry(checkpoint_dir="./checkpoints", max_snapshots=10)
registry.save(agent, eval_metrics, hyperparams)
registry.load_best(agent)
registry.load_ensemble(agent, top_k=3)   # stochastic weight averaging over top-k snapshots
```

### Regime detection

Classifies score history into one of three regimes using detrended coefficient of variation. Detrended CV measures noise around the trend, not raw score variance.

```python
from tensor_optix.core import RegimeDetector

detector = RegimeDetector()
regime = detector.detect(metrics_history)   # "trending" | "ranging" | "volatile"
```

---

## Requirements

- Python >= 3.11
- gymnasium >= 1.0
- numpy >= 1.24

The core loop, `PolicyManager`, and all ensemble and evolution logic have no framework dependency. Framework installs are opt-in via extras.
