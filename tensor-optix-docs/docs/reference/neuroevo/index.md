# Neuroevo

```bash
pip install tensor-optix[neuroevo]
```

The neuroevo subsystem represents a policy as a mutable directed graph of heterogeneous scalar neurons with variable-delay edges, and evolves its topology live during training under statistical control.

- **[NeuronGraph](neuron-graph.md)** - the graph data structure: neurons, edges, Dale's Law, compiled/eager forward.
- **[Neuron types](neurons.md)** - point, GRU, LSTM, trainable-GRU, trainable-LSTM.
- **[Topology operations](topology-ops.md)** - the function-preserving primitives (insert, split, prune, merge) that all mutations are built from.
- **[TopologyController](topology-controller.md)** - the `LoopCallback` that decides when to grow, prune, and merge.
- **[GraphAgent](graph-agent.md)** - wraps a `NeuronGraph` as a `BaseAgent` with PPO-style learning; `RecurrentGraphAgent` adds truncated BPTT for trainable recurrent neurons.
- **[BrainNetwork](brain-network.md)** - composes multiple named `NeuronGraph` regions with sparse learnable inter-region pathways.
- **[HebbianHook](hebbian.md)** - local Oja-style weight updates alongside PPO gradients.
- **[NeuromodulatorSignal](neuromodulator.md)** - maps detected training regime to coordinated parameter changes across the stack.
- **[TopologyAwareAdam](topology-aware-adam.md)** - Adam variant that resets momentum state for parameters touched by a structural mutation.

See the [Build a neuroevo agent](../../guides/neuroevo.md) guide for task-oriented usage.
