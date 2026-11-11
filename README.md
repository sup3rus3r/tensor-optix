# tensor-optix

[See Documentation](https://sup3rus3r.github.io/tensor-optix/)

tensor-optix is a training framework for reinforcement learning and for supervised/unsupervised machine learning. Both have statistical convergence control, online hyperparameter optimisation, and real checkpointing.

It has two parallel cores:

- **RL core**. An episode-driven loop (`LoopController`/`RLOptimizer`) built around `BaseAgent`'s `act()`/`learn()` contract. Statistical convergence control, SPSA-based online hyperparameter tuning, and an optional neuroevolution subsystem for dynamic topology.
- **ML core** (`tensor_optix.ml_core`). An epoch/batch-driven loop (`Trainer`, wrapped by `BaseMLModel.fit()`) built around a sklearn-style `fit()`/`predict()`/`evaluate()` contract. Real classification metrics (accuracy, precision/recall/F1, confusion matrix) and regression metrics (MAE, RMSE, R²), validation-based early stopping, and checkpointing on the best metric.

Both cores reuse the same lower-level, algorithm-agnostic machinery: `CheckpointRegistry`, `BackoffScheduler`'s convergence/plateau detection, `SPSAOptimizer`. Each core has its own contract on top because an RL episode and a training epoch aren't the same shape. See [Concepts](https://sup3rus3r.github.io/tensor-optix/getting-started/concepts/).

## RL core

The RL loop runs your agent against a pipeline, maintains a separate validation signal, and manages four states: ACTIVE, COOLING, DORMANT, and watchdog shutdown. Convergence is detected via a linear trend fit over the recent score window compared against an adaptive, noise-scaled floor, not a fixed patience counter (see [Concepts](https://sup3rus3r.github.io/tensor-optix/getting-started/concepts/) for the exact math; `BackoffScheduler` is the implementation). Hyperparameters are updated every episode via SPSA gradient estimates, with automatic routing to momentum-based or sign-only updates depending on the autocorrelation structure of the score landscape (`AdaptiveOptimizer`). Checkpointing and rollback are driven by the validation signal only, never training score. On DORMANT, a MetaController evaluates the generalization gap and its slope to decide between spawning a policy variant, pruning the ensemble, or stopping.

The neuroevo subsystem (`pip install tensor-optix[neuroevo]`) represents the policy as a `NeuronGraph`: a mutable directed graph of heterogeneous scalar neurons (point, GRU, LSTM, trainable-GRU, trainable-LSTM) with variable-delay edges. Weights for excitatory and inhibitory neurons follow softplus Dale's Law: raw parameter θ maps to `softplus(θ)` (excitatory) or `-softplus(θ)` (inhibitory), eliminating gradient dead zones at weight boundaries. `TopologyController` runs as a loop callback and evaluates three independent signals per episode: improvement slope significance, residual autocorrelation structure, and gradient utilization across hidden neurons. All three must cross their thresholds before a grow operation fires. Pruning is by importance score (incident edge weight magnitude times mean absolute activation). Merging is by Pearson correlation of per-episode activation histories. After every structural mutation the controller calls `graph.invalidate_compile()` to reset for the new topology. `TopologyAwareAdam` resets momentum state for parameters affected by any structural change. `BrainNetwork` composes multiple named `NeuronGraph` regions with sparse learnable inter-region edges. When a neuron is pruned from a region, all inter-region edges referencing it are automatically cleaned up. `HebbianHook` accumulates co-activation products across each episode and applies an Oja-style weight update after the PPO gradient step. `NeuromodulatorSignal` maps a `RegimeDetector` output (trending / ranging / volatile) to simultaneous changes in Hebbian learning rate, entropy coefficient, and topology grow/prune thresholds.

The RL core (including neuroevo) is accessed through a six-method `BaseAgent` interface:

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

## ML core

The ML loop trains your model for a number of epochs, evaluates on a held-out validation set each epoch, and stops early once the validation metric plateaus. It restores the best checkpoint automatically. It is accessed through a `BaseMLModel` interface, with `fit()` as the one-call entry point built on top:

```python
class BaseMLModel(ABC):
    def predict(self, X) -> Any: ...
    def train_step(self, batch) -> float: ...
    def evaluate(self, data) -> MLMetrics: ...
    def iter_batches(self, data, batch_size: int): ...
    def save_weights(self, path: str) -> None: ...
    def load_weights(self, path: str) -> None: ...
    # fit(train_data, val_data, epochs, early_stopping_patience, ...) is provided
```

```python
import torch.nn as nn
from tensor_optix.ml_core import TorchMLModel

model = TorchMLModel(
    nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10)),
    loss_fn=nn.CrossEntropyLoss(),
)
history = model.fit(
    (X_train, y_train), (X_val, y_val),
    epochs=50, early_stopping_patience=5, checkpoint_dir="./checkpoints",
)
print(history.val_metrics[-1].metrics)  # {'accuracy': 0.94, 'f1_macro': 0.93, ...}
```

Metrics are auto-detected from the target dtype. Integer/class-index targets get classification metrics (accuracy, per-class and macro precision/recall/F1, confusion matrix). Float targets get regression metrics (MAE, RMSE, R²). Pass `metrics="classification"`, `metrics="regression"`, or a list of custom `(y_true, y_pred) -> float` callables to override. `TorchMLModel`, `TFMLModel`, and `FlaxMLModel` are provided. The Torch and TF adapters accept a raw `(X, y)` array pair or a native `Dataset`/`DataLoader`/`tf.data.Dataset`; the Flax adapter accepts a raw `(X, y)` array pair only. `fit()` also takes `lr_schedule=` (see `tensor_optix.ml_core.lr_schedules`), `augment=` for per-batch data augmentation, and there are `freeze_layers()`/`unfreeze_layers()`/`load_pretrained_backbone()` helpers for transfer learning. See [Train an ML model](https://sup3rus3r.github.io/tensor-optix/guides/ml-core/) for the full guide.

There is also an older, RL-loop-based way to train an arbitrary `nn.Module`: `Optimizer(model, dataset)`, which runs the model through the RL loop as a fake episode. See [ML support](#ml-support) below. It predates `ml_core`, is kept for backward compatibility, and has no real classification/regression metrics and no validation-based early stopping. Use `ml_core` for new code.

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

Environment extras: `[box2d]`, `[atari]`, `[mujoco]`, `[pettingzoo]` (multi-agent)  
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

`make_agent` inspects the environment action space and returns a fully constructed agent. Pass an algorithm name as the first argument, or let it be inferred. Observations may be a flat `Box`, an image `Box` `(H, W, C)` (CNN encoder), or a `Dict` of those (see [Encoders](https://sup3rus3r.github.io/tensor-optix/reference/encoders/)).

```python
from tensor_optix import make_agent
import gymnasium as gym

env = gym.make("LunarLanderContinuous-v3")
agent = make_agent(env)                          # -> TorchSACAgent (continuous)
agent = make_agent("SAC", env)                   # same, explicit
agent = make_agent(env, framework="tf")          # -> TFSACAgent
agent = make_agent(env, deterministic=True)      # -> TorchTD3Agent

# Discrete action spaces
env = gym.make("CartPole-v1")
agent = make_agent(env)                          # -> TorchPPOAgent (default discrete)
agent = make_agent(env, algorithm="DQN")         # -> TorchDQNAgent
agent = make_agent(env, algorithm="RAINBOW")     # -> TorchRainbowDQNAgent

# Neuroevo path: NeuronGraph + GraphAgent with Hebbian + TopologyController
agent = make_agent(env, neuroevo=True)
agent = make_agent("PPO", env, neuroevo=True, graph_hidden=16, hebbian_lr=1e-3)

# Feature-extractor mode: NeuronGraph -> features concatenated to obs, fed into SAC
agent = make_agent("SAC", env, neuroevo=True, neuroevo_mode="feature_extractor")
```

Neuroevo options: `graph_in` (input neurons, default `min(obs_dim, 16)`), `graph_hidden` (GRU neurons, default 8), `graph_out` (output neurons, default `act_dim + 1`), `hebbian_lr`, `hebbian_decay`, `grow_cooldown`.

`neuroevo_mode` controls how the graph is used:
- `"policy"` (default) — graph IS the policy, wrapped as `GraphAgent` (PPO-based)
- `"feature_extractor"` — graph runs in parallel as a feature extractor; its `graph_out`-dim output is concatenated with the raw observation before SAC actor/critic networks. The base SAC layer handles exploration and replay; the graph adds adaptive temporal features via GRU + LSTM neurons with Hebbian learning.

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

Notes:

- `VectorBatchPipeline` works with every built-in agent except `TorchRecurrentPPOAgent`: PPO agents compute GAE per env, and the off-policy agents (DQN, Rainbow, SAC, TD3) build replay transitions per env. Envs are created with Gymnasium's `SAME_STEP` autoreset so no action-ignored "reset" transitions enter the training data.
- `BatchPipeline` resets a recurrent agent's hidden state (`reset_hidden()`) at every episode boundary.
- `LivePipeline` applies backpressure: when the consumer is slower than the source, the producer waits instead of dropping samples or restarting the stream.

### Multi-agent (PettingZoo)

`pip install tensor-optix[torch,pettingzoo]` adds pipelines for PettingZoo's Parallel and AEC (turn-based) APIs. Each seat is an ordinary agent; a window yields one `EpisodeData` per seat.

```python
from pettingzoo.classic import rps_v2
from tensor_optix import make_agent
from tensor_optix.multi_agent import SeatEnvView, AECMultiAgentPipeline, MultiAgentRunner

env = rps_v2.env()
env.reset(seed=0)
views  = {aid: SeatEnvView(env, aid) for aid in env.possible_agents}
agents = {aid: make_agent(views[aid]) for aid in env.possible_agents}

pipeline = AECMultiAgentPipeline(env, agents, window_size=400, views=views)
MultiAgentRunner(pipeline, agents).run(max_windows=500)
```

In turn-based games each reward (including the terminal win/loss) is credited to the action that earned it. `MultiAgentRunner` scores every seat each window, checkpoints each seat's best weights (`checkpoint_dir=`), detects per-seat convergence with the same trend test as the single-agent loop, and stops once every seat has converged (`stop_when="any"` for the first), restoring best weights. See [Multi-agent training](https://sup3rus3r.github.io/tensor-optix/guides/multi-agent/).

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

- Every non-improving eval counts toward `COOLING`/`DORMANT`, for on-policy and off-policy agents alike. While the score is flat the eval interval backs off (`backoff_factor`, default 2.0); set `backoff_factor=1.0` to evaluate every window.
- At `DORMANT` the loop stops by default. A `PolicyManagerCallback` with spawn budget remaining spawns a variant instead and training continues.
- `rollback_on_degradation=True` restores the best checkpoint when a degradation is confirmed (`min_consecutive_degradations` consecutive evals below `degradation_threshold x best`), for on-policy agents only - off-policy agents keep their weights because their replay buffer would immediately pull a restored policy back.
- On shutdown the loop restores best-known weights, not the final checkpoint.
- `verbose=True` prints a per-eval diagnostic block; it is safe on consoles that cannot render the box-drawing glyphs (they degrade to `?`).

### YAML / CLI

```bash
tensor-optix validate config.yaml
tensor-optix train config.yaml agent.learning_rate=1e-3 optimizer.max_episodes=200
```

The `algorithm` key selects the agent class and must be buildable for the env (e.g. `TorchDQNAgent` needs a discrete action space; a mismatch is a build error, not a silent substitution). See [Configure via YAML / CLI](https://sup3rus3r.github.io/tensor-optix/guides/cli-config/).

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
cb = pm.as_callback(agent, agent_factory=lambda: make_agent(env))  # zero-argument factory
cb.set_stop_fn(opt.stop)
opt.add_callback(cb)
opt.run()
```

Each `DORMANT` event with budget remaining spawns a variant and training continues; once the budget is exhausted the callback stops the loop with the best weights restored. A multi-member `EnsembleAgent` combines policies at inference time - for discrete PPO members `act()` returns the averaged probability vector, which you sample or `argmax`.

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

`HebbianHook` and `NeuromodulatorSignal` are both `LoopCallback` subclasses and can be passed directly to `Optimizer` or `RLOptimizer` via `callbacks=`.

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

For multi-region graphs, use `TopologyController.for_brain(brain, scheduler=...)`. Each region gets independent signal buffers and cooldown timers. When a neuron is pruned from a region, all inter-region `BrainNetwork` edges referencing it are automatically removed.

### Save and load

Topology-aware checkpointing works identically to non-evo agents. The topology (neuron types, roles, edges) is saved alongside weights so the graph is fully reconstructed on load — no manual topology reconstruction required.

```python
# Save
agent.save_weights("checkpoint.pt")

# Load into any agent instance — topology is reconstructed automatically
agent2 = make_agent(env, neuroevo=True)
agent2.load_weights("checkpoint.pt")

# Or skip constructing an agent entirely
agent3 = GraphAgent.from_checkpoint("checkpoint.pt")
agent3.act(obs)
```

Rollback on degradation uses the same path internally — when the loop restores a best checkpoint, the topology at the time of that checkpoint is fully restored, even if the graph has grown or been pruned since.

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

`HebbianHook` is a `LoopCallback` — pass it directly to `Optimizer` or `RLOptimizer` via `callbacks=` and it wires itself automatically. Use `HebbianHook.from_brain(brain, ...)` for `BrainNetwork` graphs.

### Neuromodulation

`NeuromodulatorSignal` takes a `RegimeDetector` classification and applies coordinated parameter changes across `HebbianHook`, `GraphAgent`, and `TopologyController` simultaneously. It is a `LoopCallback` and can be passed directly to `Optimizer`.

```python
from tensor_optix.neuroevo import NeuromodulatorSignal
from tensor_optix.core import RegimeDetector

signal = NeuromodulatorSignal(
    detector=RegimeDetector(),
    hebbian_hook=hook,           # optional
    agent=agent,                 # optional — modulates entropy_coef
    topology_controller=tc,      # optional — modulates grow/prune thresholds
)

# In your training loop, after each episode:
signal.step(metrics_history)
# trending  -> lower entropy_coef, lower hebbian_lr (consolidate)
# ranging   -> raise hebbian_lr, lower grow threshold (explore structure)
# volatile  -> lower hebbian_lr, raise entropy_coef (cautious plasticity)

# Or wire as a callback — called automatically each episode
opt = Optimizer(agent, env, callbacks=[hook, signal])
opt.run()
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

## ML support

> **Use [`tensor_optix.ml_core`](#ml-core) for new code.** The `Optimizer(model, dataset)` path below is the original way to train an arbitrary `nn.Module`. It runs the model through the RL loop's `BaseAgent`/`RLOptimizer` machinery by treating each mini-batch as a fake RL episode (`reward = -loss`). It is deprecated but kept working for backward compatibility — `MLAgent` emits a `DeprecationWarning` pointing to `ml_core` on construction, and there is no removal planned. It has no real classification/regression metrics (checkpointing and rollback compare mean "episode return," which is just negative loss) and no validation-split-based early stopping. `ml_core`'s `BaseMLModel`/`Trainer` reuse the same convergence/checkpointing machinery through a real `fit()`/`evaluate()` contract with real metrics.

tensor-optix also supports general ML training via `Optimizer`. Pass any `nn.Module` and a PyTorch `Dataset` or `DataLoader` — the same SPSA hyperparameter tuning, rollback on degradation, convergence detection, and checkpointing apply automatically.

```python
import tensor_optix as optix
import torch.nn as nn

# Supervised — loss auto-detected from dataset
model = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))
opt = optix.Optimizer(model, train_dataset)
opt.run()

# Explicit loss
opt = optix.Optimizer(model, train_dataset, loss="cross_entropy")
opt = optix.Optimizer(model, train_dataset, loss="mse")

# Autoencoder
opt = optix.Optimizer(autoencoder, train_dataset, loss="reconstruction")

# VAE — model must return (reconstruction, mu, logvar)
opt = optix.Optimizer(vae, train_dataset, loss="vae")

# Contrastive (SimCLR) — dataset yields (view1, view2) pairs
opt = optix.Optimizer(encoder, pairs_dataset, loss="contrastive")

# DataLoader works too
opt = optix.Optimizer(model, DataLoader(dataset, batch_size=64))

# Custom loss
opt = optix.Optimizer(model, dataset, loss=nn.HuberLoss())
opt.run()
```

Available `loss=` strings:

| String | Criterion | Use case |
|---|---|---|
| `"auto"` | detected from data | default |
| `"cross_entropy"` | `CrossEntropyLoss` | multi-class classification |
| `"bce"` | `BCEWithLogitsLoss` | binary classification |
| `"mse"` | `MSELoss` | regression |
| `"mae"` | `L1Loss` | regression, outlier-robust |
| `"huber"` | `HuberLoss` | regression, very outlier-robust |
| `"reconstruction"` | MSE(output, input) | autoencoders |
| `"vae"` | ELBO: recon + KL | variational autoencoders |
| `"contrastive"` | NT-Xent | SimCLR-style contrastive learning |

Any `nn.Module` or callable can be passed directly as `loss=`.

Save and load work identically to RL agents:

```python
agent = optix.Optimizer(model, dataset, loss="cross_entropy")
# Checkpoints written automatically to checkpoint_dir
# Load back:
from tensor_optix.ml import MLAgent
ml_agent.load_weights("checkpoint.pt")
```

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

## Learning-to-Learn Engine

tensor-optix also includes an optional Learning-to-Learn Engine: a layer that makes experiment history actionable — recording every trial with enough structure to compare reliably, running experiments concurrently under explicit budgets, learning which training strategies help on which tasks, and (recommend-only, never automatic) proposing interventions for a human to review before anything is promoted to a deployed configuration.

### Experiment ledger

Every experiment — a trial, a strategy candidate, a re-evaluation of a deployed agent — is recorded to a SQLite-backed ledger with full provenance (code revision, library versions, hostname) and a typed lifecycle.

```python
from tensor_optix.ledger import ExperimentStore, ExperimentState

store = ExperimentStore("./experiments.sqlite3")
record = store.create_experiment(
    task_env="CartPole-v1", algorithm="PPO", framework="torch",
    seed=0, checkpoint_dir="./ckpt", solve_threshold=475.0,
)
store.start_experiment(record.experiment_id)
store.record_progress(record.experiment_id, best_primary_score=480.0)
store.finalize_experiment(record.experiment_id, state=ExperimentState.COMPLETED)
```

Writes are idempotent on `experiment_id`, and every method opens a short, single-statement transaction — nothing here ever holds a lock for the duration of training.

To record an ordinary training run, attach `ExperimentLedgerCallback(store, task_env=..., algorithm=..., checkpoint_dir=...)` (from `tensor_optix.ledger.callback`) to `RLOptimizer`. A run that raises is finalized `FAILED` automatically; an interrupted one `CANCELLED`.

### Concurrent trial scheduler

`ConcurrentTrialScheduler` runs multiple Optuna trials as real OS processes, with the parent process as sole owner of the Optuna study and the ledger.

```python
from tensor_optix.scheduler import ConcurrentTrialScheduler

scheduler = ConcurrentTrialScheduler(
    agent_factory=make_agent, pipeline_factory=make_pipeline,
    param_space={"learning_rate": ("log_float", 1e-4, 3e-3)},
    n_trials=20, n_workers=4, ledger_store=store,
    task_env="CartPole-v1", algorithm="PPO", solve_threshold=475.0,
)
best_params, best_score = scheduler.run()
```

A worker that dies without reporting (OOM-kill, crash) is detected, retried up to `max_retries` times, and then recorded as `FAILED` - the scheduler never waits on a dead process.

### Cross-task transfer (Nanonets)

`NanonetsSelector` pools evidence from other tasks' ledger history when deciding how to allocate a new task's trial budget — but only when a conservative abstention gate confirms a source task's data is actually comparable (median performance at or above the target's own 95th percentile, both normalized against each task's own solve threshold so differently-scaled/signed scores are never compared on raw terms). When nothing passes the gate, it falls back to the non-learned baseline rather than guess.

```python
from tensor_optix.nanonets import NanonetsSelector

selector = NanonetsSelector(store, source_task_envs=["Acrobot-v1"])
decision = selector.propose(task_env="CartPole-v1", methods=["TPESampler"], budget=10)
print(decision.rationale)  # states explicitly whether it pooled or fell back
```

### Strategy discovery

A small, closed set of typed training-procedure operations (learning-rate trend adjustment, checkpoint rollback, exploration bounds, optimizer-method choice, budget allocation, schedule changes, policy-variant selection) — never arbitrary generated code. Each candidate is run paired against a no-strategy baseline, compared via a conservative confidence-interval gate, and checked against a 6-criterion promotion policy.

```python
from tensor_optix.strategy import LearningRateTrendOp, StrategyCandidate, StrategyTrialRunner, PromotionPolicy

candidate = StrategyCandidate.from_op(LearningRateTrendOp(window=5, increase_factor=1.2))
runner = StrategyTrialRunner(
    agent_factory=make_agent, pipeline_factory=make_pipeline,
    trial_steps=50_000, ledger_store=store, task_env="CartPole-v1",
    algorithm="PPO", solve_threshold=475.0,
)
results = runner.run_candidate(candidate, seeds=[0, 1, 2])

policy = PromotionPolicy(store, margin=5.0, min_seeds=3, approval_mode="human")
verdict = policy.evaluate(candidate_id=candidate.candidate_id, train_task_envs=["CartPole-v1"], held_out_task_envs=[])
```

`approval_mode="human"` (the default) means `verdict.criteria["human_reviewed"]` can only ever become `True` via a separate, explicit `store.approve_strategy(...)` call — `PromotionPolicy.evaluate()` never sets it itself.

### Live adaptive runtime

`LiveAdaptiveRuntime` runs the full observe → detect → propose → experiment → evaluate → recommend loop against a long-running deployed agent, standing in for live inference feedback. It is **recommend-only**: it issues durable, auditable recommendation records to the ledger, but the loop itself never calls `approve_strategy` and never changes which agent is deployed. Promoting a candidate requires two separate, explicit human actions outside the loop (`store.approve_strategy(...)`, then `promote_candidate_to_baseline(...)`).

```python
from tensor_optix.runtime import LiveAdaptiveRuntime, RuntimeConfig

config = RuntimeConfig(task_env="CartPole-v1", algorithm="PPO", max_cycles=20)
runtime = LiveAdaptiveRuntime(store, config, agent_factory=make_agent, pipeline_factory=make_pipeline)
observations, recommendations = runtime.run()
```

Full guide: [Learning-to-Learn Engine](https://sup3rus3r.github.io/tensor-optix/guides/learning-to-learn/).

---

## Requirements

- Python >= 3.11
- gymnasium >= 1.0
- numpy >= 1.24

The core loop, `PolicyManager`, and all ensemble and evolution logic have no framework dependency. Framework installs are opt-in via extras.
