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

### Serialization

```python
def to_dict(self) -> dict:
    """
    Serialize the full graph topology (structure only - weights come from
    state_dict). JSON-serializable; captures enough information for
    from_dict() to reconstruct an identical graph that accepts the original
    state_dict without key mismatches.
    """

@classmethod
def from_dict(cls, d: dict) -> "NeuronGraph":
    """
    Reconstruct a NeuronGraph from a topology dict produced by to_dict().
    Neurons are added with their original IDs so state_dict keys match
    exactly after a subsequent load_state_dict() call.
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
