# Topology operations

```
Function-preserving topology operations for NeuronGraph.

Every operation guarantees: graph_output_before ≈ graph_output_after
at the moment of application (before any subsequent gradient updates).
```

These are the low-level primitives that `TopologyController` calls when it decides to grow, prune, or merge - call them directly if you want manual control over topology evolution.

```python
def insert_neuron_on_edge(graph, edge_id, neuron_id=None) -> str:
    """
    Insert a relay neuron on an existing edge (u -> v, w, d).

    Math:
        Remove (u, v, w, d)
        Add    (u, new, 1.0, floor(d/2))
        Add    (new, v, w,   d - floor(d/2))

    The relay is obtained from src_neuron.make_relay() which always returns
    a linear point neuron (exact function preservation).

    Returns the new neuron_id.
    """

def split_neuron(graph, neuron_id) -> tuple[str, str]:
    """
    Split neuron v_k into v_k1 (kept, renamed in-place) and v_k2 (new).

    Math:
        incoming edges: both copies receive full weight
        outgoing edges: both copies send w/2  (sum = w, output preserved)
        bias:           both copies receive b/2

    Returns (original_id, new_id). Original neuron keeps its id.
    """

def add_input_neuron(graph, activation="linear", neuron_id=None) -> str:
    """
    Add a new input neuron for a new observation dimension.
    All outgoing edges are added with w=0, so existing outputs are unchanged.
    Returns the new neuron_id.
    """

def add_free_edge(graph, src, dst, delay=1, edge_id=None) -> str:
    """
    Add a free-form edge between any two existing neurons with w=0.
    Zero weight → output preserved at insertion time. delay >= 1 adds a
    recurrent connection. Returns the edge_id.
    """

def prune_edge(graph, edge_id) -> None:
    """Remove a single edge unconditionally."""

def prune_neuron(graph, neuron_id, redistribute=True) -> None:
    """
    Remove a neuron, optionally redistributing its signal to preserve output.

    Redistribution math (approximate, exact only for linear activation):
        For each (u -> v_k, w1, d1) and (v_k -> z, w2, d2):
            Add (u -> z, w1 * w2, d1 + d2)

    Input/output neurons cannot be pruned.
    """

def merge_neurons(graph, neuron_id_a, neuron_id_b) -> str:
    """
    Merge two neurons with near-identical activations into one.

    Strategy:
        - Keep neuron_a
        - For each incoming edge to b: if a already has an edge from same
          src with same delay, add weights; else create new edge to a
        - Same merge logic for outgoing edges
        - Average the biases
        - Remove neuron_b

    Returns the surviving neuron_id (always neuron_id_a).
    """

def neuron_importance(graph, neuron_id) -> float:
    """
    I(v) = Σ|w_e| * (‖h‖₁/d + ε)
    Delegates to neuron.importance() so GRU/LSTM neurons normalise by hidden dim.
    """

def edge_importance(graph, edge_id) -> float:
    """Simple |w| magnitude."""

def cosine_similarity_neurons(graph, nid_a, nid_b) -> float:
    """Cosine similarity between two neurons' current activations - used to detect merge candidates."""
```

See [TopologyController](topology-controller.md) for the statistical signals that decide *when* to call these.
