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

`RecurrentGraphAgent.learn()` falls back to standard shuffled-minibatch PPO automatically when the graph holds no state (no GRU/LSTM neurons of any kind and no delayed edges) - it's safe to use in place of `GraphAgent` even for non-recurrent topologies.

See [Build a neuroevo agent](../../guides/neuroevo.md) for usage, including save/load and checkpoint round-tripping.

## Training forward, log-probabilities and bootstrap

- **One function for acting and learning.** `act()` resets the graph state and runs one `graph(obs)` step. `learn()` evaluates the same function in batch (`_batch_forward`, which is `NeuronGraph.forward_batch` from a zero state): it uses each edge's *effective* (post-Dale) weight, gives delayed edges no contribution (their history is zero after the per-step reset), and evaluates GRU/LSTM neurons from a zero state. The policy that is optimised is therefore exactly the policy that collected the data, for every neuron type and both `dale_mode`s.
- **Old log-probabilities.** Pipelines do not record `EpisodeData.log_probs`. When a window carries none, `learn()` evaluates the collected actions under the pre-update policy and uses that as `log pi_old` - no gradient step has happened since collection, so it is the behaviour policy. Supply `log_probs` yourself only if the data came from a different policy.
- **Bootstrap.** When a window ends mid-episode and the pipeline provides `final_obs`, the GAE target is bootstrapped from the value output on it.
- **Optimizer state survives topology changes.** The agent's optimizer is a `TopologyAwareAdam`. After a grow, prune or merge it is rebuilt and Adam's moments are carried over by edge id (see [TopologyAwareAdam](topology-aware-adam.md)). `load_weights()` starts it fresh.
- **Checkpoint restore is in place.** `load_weights()` rebuilds the existing `NeuronGraph` object (`graph.restore_from_dict()`) instead of replacing it, so a `TopologyController`, `HebbianHook` or `BrainNetwork` holding that graph keeps following it across rollbacks and best-weight restores.

`RecurrentGraphAgent` keeps memory across timesteps. `act()` carries a `GraphState` (GRU/LSTM hidden and cell states, delayed-edge history) from one call to the next and steps it with `NeuronGraph.forward_batch`. The state resets when the pipeline calls `reset_hidden()` at episode start, which `BatchPipeline` does. `learn()` replays the window from the state `act()` started it with, resetting wherever `act()` did, so the log-probabilities and values it computes before updating are exactly the behaviour policy's. Training is truncated BPTT over chunks of `chunk_len` steps, each starting from the state `act()` had at that point. A window that `act()` did not produce starts from zeros and resets after every done. When the graph holds no state at all (point neurons only, no delayed edges), `learn()` uses `GraphAgent`'s shuffled-minibatch PPO, since the two policies are then the same function.
