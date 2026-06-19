# Neuron types

```python
CELL_TYPES = {"excitatory", "inhibitory", "any"}
```

All neuron types implement the same protocol - `NeuronGraph` and `TopologyController` are type-blind:

```python
step(weighted_sum)            # full forward: bias + activation
importance(incident_w_sum)    # I(v) = Σ|w_e| * (‖h‖₁/d + ε)
can_merge_with(other)         # same type + same dim required
make_relay()                  # returns a linear point neuron (exact identity)
split_copy()                  # deep-copy of self; caller halves outgoing edges
```

| Type | Hidden state | Gradient through state |
|---|---|---|
| `Neuron` | None (point neuron) | N/A |
| `GRUNeuron` | Scalar h, detached | No |
| `LSTMNeuron` | Scalar h and c, detached | No |
| `TrainableGRUNeuron` | Scalar h, not detached | Yes, up to `chunk_len` steps |
| `TrainableLSTMNeuron` | Scalar h and c, not detached | Yes, up to `chunk_len` steps |

## Neuron

```python
class Neuron(nn.Module):
    """
    A single point neuron: bias + activation + delay history buffer.

    cell_type enforces Dale's Law:
      'excitatory' → all outgoing weights clamped ≥ 0
      'inhibitory' → all outgoing weights clamped ≤ 0
      'any'        → unconstrained (default, backward-compatible)
    """

    def __init__(self, activation="tanh", neuron_id=None, max_delay=1, cell_type="any"): ...

    def step(self, weighted_sum: torch.Tensor) -> torch.Tensor:
        """Full forward: bias addition + activation. Sets and returns self._current."""

    def importance(self, incident_weight_sum: float) -> float:
        """
        I(v) = Σ|w_e| * (‖h‖₁/d + ε)
        incident_weight_sum is pre-computed by the controller in a shared edge pass.
        """

    def make_relay(self) -> "Neuron":
        """
        Return a new neuron suitable for insertion on an edge incident to self.

        Always returns a linear point neuron regardless of self's type - this
        is the only choice that gives exact function preservation (verified:
        GRU/LSTM relays have 24–36% error in the tanh activation range).
        Gradient descent adapts the relay afterward.
        """

    def split_copy(self) -> "Neuron":
        """
        Deep copy of self with a new neuron_id. Caller halves all outgoing
        edge weights on both original and copy so combined output is unchanged.
        """
```

## GRUNeuron

```python
class GRUNeuron(Neuron):
    """
    Scalar GRU cell embedded as a graph neuron.

    Receives a scalar weighted-sum input x and maintains a scalar hidden
    state h that persists across timesteps via the GRU gating equations:

        z = σ(wz·h_prev + uz·x + bz)          # update gate
        r = σ(wr·h_prev + ur·x + br)          # reset gate
        n = tanh(wn·r·h_prev + un·x + bn)     # candidate
        h = (1-z)·h_prev + z·n                # output

    bias (inherited) is not used - gate biases (bz, br, bn) replace it.
    Dale's Law and delay history work identically to a point Neuron.
    """
```

## LSTMNeuron

```python
class LSTMNeuron(Neuron):
    """
    Scalar LSTM cell embedded as a graph neuron.

    Maintains scalar hidden state h and cell state c:
        f = σ(wf·h_prev + uf·x + bf)          # forget gate
        i = σ(wi·h_prev + ui·x + bi)          # input gate
        g = tanh(wg·h_prev + ug·x + bg)       # cell gate
        o = σ(wo·h_prev + uo·x + bo)          # output gate
        c = f·c_prev + i·g
        h = o·tanh(c)
    """
```

## TrainableGRUNeuron / TrainableLSTMNeuron

```python
class TrainableGRUNeuron(GRUNeuron):
    """
    GRU neuron whose gate parameters train via PPO gradients (truncated BPTT).

    The key difference from GRUNeuron.step(): the hidden state is NOT
    detached, so gradients flow backward through time up to chunk_len steps.

    Requirements:
      - Use with RecurrentGraphAgent (or any agent calling recurrent_forward).
      - The agent must call reset_train_state() at sequence start and detach
        _h_train every chunk_len steps (truncated BPTT).
      - Inference (act()) still uses step(), which detaches as normal.

    is_recurrent = True signals RecurrentGraphAgent to route training through
    recurrent_forward() instead of _batch_forward().
    """
    is_recurrent: bool = True

    def recurrent_step(self, x: torch.Tensor) -> torch.Tensor:
        """Single-timestep GRU forward with undetached hidden state."""

    def reset_train_state(self) -> None:
        """Reset the training hidden state (call at sequence start)."""


class TrainableLSTMNeuron(LSTMNeuron):
    """
    LSTM neuron whose gate parameters train via PPO gradients (truncated BPTT).
    Neither _h nor _c is detached during recurrent_step, so gradients flow
    backward through time up to chunk_len steps. Same requirements as
    TrainableGRUNeuron.
    """
    is_recurrent: bool = True

    def recurrent_step(self, x: torch.Tensor) -> torch.Tensor:
        """Single-timestep LSTM forward with undetached hidden/cell states."""

    def reset_train_state(self) -> None:
        """Reset the training hidden/cell states (call at sequence start)."""
```

See [GraphAgent](graph-agent.md) for `RecurrentGraphAgent`, which detects `is_recurrent` neurons and switches training mode automatically.
