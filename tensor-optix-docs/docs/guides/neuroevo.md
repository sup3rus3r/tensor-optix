# Build a neuroevo agent

```bash
pip install tensor-optix[neuroevo]
```

The neuroevo subsystem represents a policy as a `NeuronGraph`: a mutable directed graph of heterogeneous scalar neurons (point, GRU, LSTM, trainable-GRU, trainable-LSTM) connected by variable-delay edges. A `TopologyController` runs as a loop callback and mutates this graph live during training - growing capacity when the existing network is saturated, pruning dead neurons and edges, and merging redundant ones.

## Quickest path: make_agent

```python
from tensor_optix import make_agent
import gymnasium as gym

env = gym.make("LunarLanderContinuous-v3")
agent = make_agent(env, neuroevo=True)
agent = make_agent("PPO", env, neuroevo=True, graph_hidden=16, hebbian_lr=1e-3)

# Feature-extractor mode: NeuronGraph -> features concatenated to obs, fed into SAC
agent = make_agent("SAC", env, neuroevo=True, neuroevo_mode="feature_extractor")
```

`neuroevo_mode` controls how the graph is used:

- `"policy"` (default) - the graph *is* the policy, wrapped as `GraphAgent` (PPO-based).
- `"feature_extractor"` - the graph runs in parallel as a feature extractor; its output is concatenated with the raw observation before SAC's actor/critic networks. SAC handles exploration and replay; the graph adds adaptive temporal features via GRU + LSTM neurons with Hebbian learning.

Neuroevo options on `make_agent`: `graph_in` (default `min(obs_dim, 16)`), `graph_hidden` (default 8), `graph_out` (default `act_dim + 1`), `hebbian_lr`, `hebbian_decay`, `grow_cooldown`.

## Building a graph manually

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

All edges initialize at `weight=0.0`, which is function-preserving at insertion time - this is the same Net2Net-style principle behind every topology operation (see [Topology operations](../reference/neuroevo/topology-ops.md)).

## Neuron types

| Type | Hidden state | Gradient through state |
|---|---|---|
| `Neuron` | None (point neuron) | N/A |
| `GRUNeuron` | Scalar h, detached | No |
| `LSTMNeuron` | Scalar h and c, detached | No |
| `TrainableGRUNeuron` | Scalar h, not detached | Yes, up to `chunk_len` steps |
| `TrainableLSTMNeuron` | Scalar h and c, not detached | Yes, up to `chunk_len` steps |

All types implement the same protocol (`step()`, `importance()`, `can_merge_with()`, `make_relay()`, `split_copy()`) - `NeuronGraph` and `TopologyController` are type-blind.

## Trainable recurrent neurons

`TrainableGRUNeuron`/`TrainableLSTMNeuron` set `is_recurrent = True`. `RecurrentGraphAgent` detects this and switches from shuffled-minibatch PPO to sequential chunk training with truncated BPTT:

```python
from tensor_optix.neuroevo import TrainableGRUNeuron, TrainableLSTMNeuron, RecurrentGraphAgent
from tensor_optix.core.types import HyperparamSet

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

## The topology controller

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

A grow operation fires only when all three statistical signals agree:

1. **Improvement test** - the score slope is *not* significantly positive (a corrected t-test) - i.e. gradient updates have stalled.
2. **Structure test** - score residuals show significant lag-1 autocorrelation - i.e. there's unexplained structure the model isn't capturing.
3. **Capacity test** - gradient utilization across hidden neurons exceeds `grow_grad_threshold` - i.e. existing neurons are saturated, not idle.

For multi-region graphs, use `TopologyController.for_brain(brain, scheduler=...)` - each region gets independent signal buffers and cooldown timers, so a saturated encoder region can grow without triggering growth elsewhere.

## Save and load

Topology is saved alongside weights, so a checkpoint fully reconstructs the graph on load - no manual topology bookkeeping:

```python
agent.save_weights("checkpoint.pt")

agent2 = make_agent(env, neuroevo=True)
agent2.load_weights("checkpoint.pt")

# Or skip constructing an agent entirely
agent3 = GraphAgent.from_checkpoint("checkpoint.pt")
agent3.act(obs)
```

Rollback on degradation uses the same path - when the loop restores a best checkpoint, the topology *at the time of that checkpoint* is fully restored, even if the graph has grown or been pruned since.

## BrainNetwork: composing multiple regions

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

Inter-region edges are learnable parameters; when a neuron is pruned from a region, `BrainNetwork` automatically removes all inter-region edges referencing it.

## Hebbian learning

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

`HebbianHook` is a `LoopCallback` - pass it directly via `callbacks=` and it wires itself automatically. Use `HebbianHook.from_brain(brain, ...)` for `BrainNetwork` graphs.

## Neuromodulation

```python
from tensor_optix.neuroevo import NeuromodulatorSignal
from tensor_optix.core import RegimeDetector

signal = NeuromodulatorSignal(
    detector=RegimeDetector(),
    hebbian_hook=hook,           # optional
    agent=agent,                 # optional - modulates entropy_coef
    topology_controller=tc,      # optional - modulates grow/prune thresholds
)

opt = Optimizer(agent, env, callbacks=[hook, signal])
opt.run()
```

`NeuromodulatorSignal` maps a `RegimeDetector` classification (trending / ranging / volatile) to coordinated parameter changes: trending → lower Hebbian lr and entropy (consolidate); ranging → raise Hebbian lr, lower grow threshold (explore structure); volatile → lower Hebbian lr, raise entropy (cautious plasticity).

## Dale's Law

```python
# clamp mode (default): outgoing weights clamped post-step
graph = NeuronGraph(dale_mode="clamp")
graph.add_neuron(role="hidden", activation="relu", cell_type="excitatory")  # weights >= 0
graph.add_neuron(role="hidden", activation="tanh", cell_type="inhibitory")  # weights <= 0

# softplus mode: raw parameter theta, effective weight = softplus(theta) * sign
# gradient-safe, no dead zone at the clamp boundary
graph = NeuronGraph(dale_mode="softplus")
w = graph.effective_weight(edge_id)  # reads post-softplus value
```

## Compiled forward

`NeuronGraph` runs eager by default - `_raw_forward` mutates Python-side neuron state, which `torch.compile` can't safely trace without replaying side effects. If your topology is static and you manage neuron state externally, opt in:

```python
graph.compile_forward()   # one-time call; re-call after any topology mutation
```

`TopologyController` calls `graph.invalidate_compile()` automatically after every grow/prune/merge. If you mutate the graph directly outside the controller, call it yourself.

## Reference

Full class and function documentation: [Neuroevo reference](../reference/neuroevo/index.md).
