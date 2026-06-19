# GraphAgent

```
GraphAgent - a BaseAgent backed by a free-form NeuronGraph.

Weight learning is PPO-style: the graph acts as both actor and critic.
The actor head reads output neurons [0:-1], the critic head reads the last
output neuron. For discrete actions the actor outputs logits; for continuous
actions it outputs means (std is a learned parameter).

The agent is intentionally minimal - it wires the NeuronGraph into the
tensor-optix contract. Topology mutations are performed externally by
TopologyController, which holds a reference to the graph.
```

## GraphAgent

```python
class GraphAgent(BaseAgent):
    """
    RL agent whose policy network is a mutable NeuronGraph.

    Parameters
    ----------
    graph:
        A NeuronGraph already configured with input/output neurons.
        The last output neuron is the value head; all others are action
        logits (discrete) or action means (continuous).
    obs_dim:
        Expected observation dimensionality. If a new obs arrives with more
        dimensions, the agent automatically grows new input neurons.
    n_actions:
        Number of discrete actions, or dimension of continuous action space.
    continuous:
        If True, actions are sampled from a Gaussian. If False, from Categorical.
    hyperparams:
        Initial HyperparamSet. Keys used: learning_rate, clip_ratio,
        entropy_coef, vf_coef, gamma, gae_lambda, n_epochs, minibatch_size,
        max_grad_norm.
    """

    is_on_policy = True

    default_hyperparams = {
        "learning_rate": 3e-4, "clip_ratio": 0.2, "entropy_coef": 0.01, "vf_coef": 0.5,
        "gamma": 0.99, "gae_lambda": 0.95, "n_epochs": 4, "minibatch_size": 64,
        "max_grad_norm": 0.5,
    }

    def act(self, observation) -> any:
        """
        Given a numpy observation, return an action (and store log_prob).
        Dynamically grows input neurons if obs_dim has expanded.
        """

    @classmethod
    def from_checkpoint(cls, path: str, device: str = "auto") -> "GraphAgent":
        """
        Load a fully-evolved GraphAgent from a checkpoint saved by save_weights().

        Unlike load_weights(), this does not require a pre-constructed agent -
        it reconstructs the topology from the checkpoint and returns a ready
        agent.

            agent = GraphAgent.from_checkpoint("checkpoints/best.pt")
            agent.act(obs)
        """
```

`_batch_forward` runs a vectorized batch forward over the graph - instead of looping over `B` observations, it keeps an activation buffer `[B, n_neurons]` and propagates in topological order, so each neuron's pre-activation is a sum of weighted columns via pure tensor ops.

## RecurrentGraphAgent

```python
class RecurrentGraphAgent(GraphAgent):
    """
    GraphAgent subclass that supports TrainableGRUNeuron / TrainableLSTMNeuron.

    When the graph contains any neuron with is_recurrent=True, learn() switches
    from the default shuffled-minibatch PPO to sequential chunk-based training
    (truncated BPTT). Inference via act() is unchanged - it still calls the
    parent step() path with detached hidden states.

    Extra hyperparams:
      chunk_len (int, default 64): BPTT truncation length in timesteps.
    """

    default_hyperparams = {**GraphAgent.default_hyperparams, "chunk_len": 64}

    def recurrent_forward(self, obs_sequence: torch.Tensor, chunk_len: int = 64) -> torch.Tensor:
        """
        Process T timesteps in order with truncated BPTT every chunk_len steps.
        Returns [T, n_outputs].
        """
```

`RecurrentGraphAgent.learn()` falls back to standard shuffled-minibatch PPO automatically when no recurrent neurons are present in the graph - it's safe to use in place of `GraphAgent` even for non-recurrent topologies.

See [Build a neuroevo agent](../../guides/neuroevo.md) for usage, including save/load and checkpoint round-tripping.
