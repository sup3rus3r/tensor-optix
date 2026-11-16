# NeuronGraph

## Edge

```python
@dataclass
class Edge:
    """A directed, variable-delay weighted edge."""
    edge_id: str
    src: str
    dst: str
    weight: nn.Parameter
    delay: int
```

## NeuronGraph

```python
class NeuronGraph(nn.Module):
    """
    A mutable directed graph of neurons with variable-delay edges.

    Forward pass:
        h_v^(t) = σ_v( b_v + Σ_{(u,v,w,d) ∈ E} w · h_u^(t-d) )

    DAG edges use d=0 (resolved by topological sort within a timestep).
    Recurrent edges use d>=1 (resolved from each neuron's history buffer).

    The graph is split into:
    - input neurons:  receive external observations, no incoming edges
    - hidden neurons: fully internal
    - output neurons: their activations form the action vector

    Topology can be mutated at any time via add_neuron / add_edge /
    remove_neuron / remove_edge without interrupting gradient flow on
    surviving parameters.
    """

    def __init__(self, dale_mode: str = "clamp"):
        """
        dale_mode: 'clamp' (default) or 'softplus'.

        'clamp'    - enforce_dale() post-step clamp (gradient-dead at boundary).
        'softplus' - raw parameter θ; effective_weight = softplus(θ) * sign.
                     Zero-weight init uses θ=-10 (softplus(-10)≈4.5e-5).
                     enforce_dale() is a no-op in this mode.
        """
```

### Building the graph

```python
def add_neuron(self, role="hidden", activation="tanh", neuron_id=None,
                max_delay=1, cell_type="any", neuron=None) -> str:
    """
    Add a neuron, return its id.  role: 'input' | 'hidden' | 'output'.

    Pass a pre-constructed Neuron instance (GRUNeuron, LSTMNeuron, etc.)
    via the `neuron` kwarg for heterogeneous graphs.  When `neuron` is
    given the activation/neuron_id/max_delay/cell_type kwargs are ignored.
    """

def add_edge(self, src, dst, weight=0.0, delay=0, edge_id=None) -> str:
    """
    Add a directed edge src->dst with given weight and delay.
    weight=0.0 default ensures function-preserving insertion.
    """

def remove_edge(self, edge_id: str) -> None:
    """Remove an edge and free its parameter."""

def remove_neuron(self, neuron_id: str) -> None:
    """Remove a neuron and all its incident edges."""
```

### Forward pass and compilation

```python
def compile_forward(self):
    """
    Replace the forward pass with a torch.compile'd version.

    Call this ONLY when the graph topology is static and you manage neuron
    state (reset_state, push_history) manually outside the forward call.
    Not recommended for general use - prefer the default eager forward.
    """

def invalidate_compile(self) -> None:
    """
    Reset the forward function after a topology change or device move.

    In eager mode (default): rebuilds matrix cache and resets to eager forward.
    If compile_forward() was previously called: resets dynamo state and
    re-compiles with the new topology. This reset is process-global - all
    NeuronGraph instances in the process retrace on their next forward call.
    """

def forward(self, obs: torch.Tensor) -> torch.Tensor: ...
```

`NeuronGraph` runs in eager mode by default because the forward pass mutates Python-side neuron state (`neuron._current`, `push_history`), which `torch.compile` cannot safely trace without replaying those side effects. The backend for `compile_forward()` is selected automatically: `inductor` on Linux/macOS, `aot_eager` on Windows.

Internally, a fast vectorized path (`_fast_forward`) is used automatically for "uniform" graphs (no recurrent edges, all non-input neurons share the same point-neuron type) - one matmul per topological level per activation group, instead of a per-neuron Python loop.

### Device

`NeuronGraph(device=None)` and `BrainNetwork(device=None)` take an explicit device. With `None` they use the global tensor-optix device, which is `"cpu"` unless you call `tensor_optix.set_device("cuda")` or `tensor_optix.auto_device()`. The default is not changed to CUDA automatically.

### Batched forward with explicit state

```python
def initial_state(self, batch_size: int = 1) -> GraphState: ...

def forward_batch(self, obs: Tensor, state: Optional[GraphState] = None
                  ) -> tuple[Tensor, GraphState]:
    """
    One timestep for a whole batch. obs [B, n_inputs] -> (out [B, n_outputs],
    next state). The recurrent state is passed in and returned instead of
    living inside the neurons.
    """

@dataclass
class GraphState:
    h: Tensor            # [B, n] previous activations
    c: Tensor            # [B, n] LSTM cell state
    hist: Optional[Tensor]  # [B, max_delay, n] past activations, newest first
    def detach(self) -> "GraphState": ...
```

`forward_batch` computes the same function as running `forward()` on each row from a reset graph, as a few tensor ops per topological level instead of a Python loop over neurons. It handles point, GRU and LSTM neurons (and the Trainable variants) and delayed edges. Neuron objects are only read for their parameters, so `_current`, `_h`, `_c` and the history buffers are not touched. Gradients flow through the returned state, so an unrolled sequence trains by backprop through time; call `state.detach()` every k steps to truncate. Plain `GRUNeuron` and `LSTMNeuron` keep detaching their own hidden state each step, as their `step()` does; the Trainable variants do not. A custom `Neuron` subclass raises `NotImplementedError`.

### Edge access

```python
def edge_table(self) -> EdgeTable:
    """All edges as aligned tensors: edge_ids, src, dst (neuron indices),
    weight (effective, post-Dale, detached) and grad (gradient of the stored
    parameter, 0 before any backward)."""
```

Read edge gradients from here. For packed feedforward edges `edge.weight` is a view that sits outside the forward graph, so `edge.weight.grad` is always empty; the gradient lives on the packed vector. `edge_table()` does it for the whole graph at once with no per-edge host sync.

### Activation recording

```python
def enable_activation_trace(self, capacity: int = 100) -> None: ...
def record_activations(self, h: Tensor) -> None:
    """h is [n] or [S, n], columns in neuron_index order."""
def drain_activation_trace(self) -> Optional[Tensor]: ...
def activation_snapshot(self) -> Tensor: ...
```

`forward()` and `forward_batch()` record every step; `TopologyController` turns the recorder on and drains it each episode. If you run your own forward pass, call `record_activations(h)` with its activations so the controller can see them.

### Delayed edges

A delayed edge `u -> v` with delay `d` reads `u`'s activation from `d` steps ago, so `u` is the neuron that stores history. Earlier versions set the history flag on `v` instead, which meant a delayed edge between two different neurons always delivered zeros unless `u` happened to have a delayed input of its own (a self loop worked, which is why it went unnoticed).

### Serialization

```python
def to_dict(self, include_weights: bool = True) -> dict:
    """
    Serialize the graph. JSON-serializable. With include_weights=True (the
    default) every parameter is stored under "state", so from_dict() returns a
    graph that computes the same function. With include_weights=False only the
    structure is stored; from_dict() then gives freshly initialised weights and
    you load state_dict() yourself (GraphAgent checkpoints do this).
    """

@classmethod
def from_dict(cls, d: dict, device=None) -> "NeuronGraph":
    """
    Reconstruct a NeuronGraph from a dict produced by to_dict(). Neurons are
    added with their original IDs so state_dict keys match exactly. Loads the
    weights if the dict carries them.
    """
```

### Dale's Law

```python
def cell_type_of(self, neuron_id: str) -> str:
    """Return the cell_type of a neuron ('excitatory', 'inhibitory', or 'any')."""

def enforce_dale(self) -> None:
    """
    Enforce Dale's Law after each optimizer step.

    'clamp' mode: clamp outgoing weights (excitatory >= 0, inhibitory <= 0).
    'softplus' mode: no-op - the softplus transform already enforces the
                     sign constraint at every forward pass.
    """

def effective_weight(self, edge_id: str) -> float:
    """
    Return the effective float weight of an edge (post-softplus if applicable).
    Use this instead of edge.weight.item() when dale_mode='softplus'.
    """
```

### Other properties and methods

```python
def reset_state(self) -> None:
    """Zero all neuron histories. Call at episode start."""

@property
def input_ids(self) -> List[str]: ...
@property
def output_ids(self) -> List[str]: ...
@property
def hidden_ids(self) -> List[str]: ...
def n_neurons(self) -> int: ...
def n_edges(self) -> int: ...
def get_neuron(self, neuron_id: str) -> "Neuron": ...
def get_edge(self, edge_id: str) -> Edge: ...
def edges_into(self, neuron_id: str) -> List[Edge]: ...
def edges_from(self, neuron_id: str) -> List[Edge]: ...
def all_edges(self) -> List[Edge]: ...
def all_neuron_ids(self) -> List[str]: ...
```

`_topological_order()` uses Kahn's algorithm over feedforward (`d=0`) edges only - recurrent edges always read from history, so they impose no within-timestep ordering constraint.

### In-place restore and forward hooks

```python
def restore_from_dict(self, d: dict) -> "NeuronGraph":
    """Rebuild THIS graph object in place from a to_dict() dict. Parameters are
    re-created; loaded from the dict if it carries weights, otherwise follow
    with load_state_dict()."""
```

Use this (not `from_dict`) when other objects hold a reference to the graph. `graph._post_forward_hooks` is a list of zero-argument callables run after every `forward()`; `HebbianHook` uses it to sample activations.
