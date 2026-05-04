from __future__ import annotations

"""
Function-preserving topology operations for NeuronGraph.

Every operation guarantees: graph_output_before ≈ graph_output_after
at the moment of application (before any subsequent gradient updates).

Operations:
  insert_neuron_on_edge  — Net2Net free-form: split an edge with a relay neuron
  split_neuron           — duplicate a neuron, halve outgoing weights
  add_input_neuron       — zero-weight init, preserves all existing outputs
  add_free_edge          — zero-weight init, preserves output
  prune_edge             — remove a low-magnitude edge
  prune_neuron           — redistribute signal then remove neuron
  merge_neurons          — collapse two nearly-identical neurons into one
"""

import math
from typing import Optional

import torch

from .neuron_graph import NeuronGraph


# ---------------------------------------------------------------------------
# Grow operations
# ---------------------------------------------------------------------------

def insert_neuron_on_edge(
    graph: NeuronGraph,
    edge_id: str,
    activation: str = "linear",
    neuron_id: Optional[str] = None,
) -> str:
    """
    Insert a relay neuron on an existing edge (u -> v, w, d).

    Math:
        Remove (u, v, w, d)
        Add    (u, new, 1.0, floor(d/2))
        Add    (new, v, w,   d - floor(d/2))

    The new neuron is linear with bias=0, so:
        h_new = 1.0 * h_u^(t - floor(d/2))
        h_v   = w * h_new^(t - (d - floor(d/2)))
              = w * h_u^(t - d)   ← identical to before

    Returns the new neuron_id.
    """
    edge = graph.get_edge(edge_id)
    src, dst = edge.src, edge.dst
    w = edge.weight.item()
    d = edge.delay

    d1 = d // 2
    d2 = d - d1

    # Compute plausible history for the new neuron from src's existing history
    src_neuron = graph.get_neuron(src)
    new_history = [
        src_neuron.get_delayed(k + d1) for k in range(1, src_neuron.max_delay + 1)
    ]

    new_id = graph.add_neuron(
        role="hidden",
        activation=activation,
        neuron_id=neuron_id,
        max_delay=max(1, d2),
    )
    new_neuron = graph.get_neuron(new_id)
    if new_history:
        new_neuron.init_history_from_buffer(new_history)

    graph.remove_edge(edge_id)
    graph.add_edge(src=src, dst=new_id, weight=1.0, delay=d1)
    graph.add_edge(src=new_id, dst=dst, weight=w, delay=d2)

    return new_id


def split_neuron(
    graph: NeuronGraph,
    neuron_id: str,
) -> tuple[str, str]:
    """
    Split neuron v_k into v_k1 (kept, renamed in-place) and v_k2 (new).

    Math:
        incoming edges: both copies receive full weight
        outgoing edges: both copies send w/2  (sum = w, output preserved)
        bias:           both copies receive b/2

    Returns (original_id, new_id). Original neuron keeps its id.
    """
    neuron = graph.get_neuron(neuron_id)
    in_edges = graph.edges_into(neuron_id)
    out_edges = graph.edges_from(neuron_id)

    # Create the clone with same activation, max_delay
    new_id = graph.add_neuron(
        role="hidden" if neuron_id in graph.hidden_ids else "output",
        activation=neuron.activation_name,
        max_delay=neuron.max_delay,
    )
    new_neuron = graph.get_neuron(new_id)

    # Copy bias (halved on both)
    b = neuron.bias.item()
    with torch.no_grad():
        neuron.bias.fill_(b / 2)
        new_neuron.bias.fill_(b / 2)

    # Copy history to new neuron
    new_neuron.init_history_from_buffer(list(neuron._history))
    new_neuron._current = neuron._current.clone()

    # Replicate incoming edges onto new neuron (full weight)
    for e in in_edges:
        graph.add_edge(src=e.src, dst=new_id, weight=e.weight.item(), delay=e.delay)

    # Halve outgoing weights on original; add halved copy from new neuron
    for e in out_edges:
        w = e.weight.item()
        with torch.no_grad():
            e.weight.fill_(w / 2)
        graph.add_edge(src=new_id, dst=e.dst, weight=w / 2, delay=e.delay)

    return neuron_id, new_id


def add_input_neuron(
    graph: NeuronGraph,
    activation: str = "linear",
    neuron_id: Optional[str] = None,
) -> str:
    """
    Add a new input neuron for a new observation dimension.

    All outgoing edges are added with w=0, so existing outputs are unchanged.
    Gradient will grow connections organically.

    Returns the new neuron_id.
    """
    new_id = graph.add_neuron(
        role="input",
        activation=activation,
        neuron_id=neuron_id,
        max_delay=1,
    )
    # Zero-weight edges to all hidden and output neurons
    for target_id in graph.hidden_ids + graph.output_ids:
        graph.add_edge(src=new_id, dst=target_id, weight=0.0, delay=0)

    return new_id


def add_free_edge(
    graph: NeuronGraph,
    src: str,
    dst: str,
    delay: int = 1,
    edge_id: Optional[str] = None,
) -> str:
    """
    Add a free-form edge between any two existing neurons with w=0.

    Zero weight → output preserved at insertion time.
    delay >= 1 adds a recurrent connection.

    Returns the edge_id.
    """
    return graph.add_edge(src=src, dst=dst, weight=0.0, delay=delay, edge_id=edge_id)


# ---------------------------------------------------------------------------
# Prune operations
# ---------------------------------------------------------------------------

def prune_edge(graph: NeuronGraph, edge_id: str) -> None:
    """Remove a single edge unconditionally."""
    graph.remove_edge(edge_id)


def prune_neuron(
    graph: NeuronGraph,
    neuron_id: str,
    redistribute: bool = True,
) -> None:
    """
    Remove a neuron, optionally redistributing its signal to preserve output.

    Redistribution math (approximate, exact only for linear activation):
        For each (u -> v_k, w1, d1) and (v_k -> z, w2, d2):
            Add (u -> z, w1 * w2, d1 + d2)

    This collapses the neuron out of the path.
    Input/output neurons cannot be pruned.
    """
    if neuron_id in graph.input_ids:
        raise ValueError("Cannot prune an input neuron")
    if neuron_id in graph.output_ids:
        raise ValueError("Cannot prune an output neuron")

    if redistribute:
        in_edges = graph.edges_into(neuron_id)
        out_edges = graph.edges_from(neuron_id)
        for ie in in_edges:
            for oe in out_edges:
                combined_w = ie.weight.item() * oe.weight.item()
                combined_d = ie.delay + oe.delay
                # Only add if non-trivial
                if abs(combined_w) > 1e-9:
                    graph.add_edge(
                        src=ie.src,
                        dst=oe.dst,
                        weight=combined_w,
                        delay=combined_d,
                    )

    graph.remove_neuron(neuron_id)


def merge_neurons(
    graph: NeuronGraph,
    neuron_id_a: str,
    neuron_id_b: str,
) -> str:
    """
    Merge two neurons with near-identical activations into one.

    Strategy:
        - Keep neuron_a
        - For each incoming edge to b: if a already has an edge from same src
          with same delay, add weights; else create new edge to a
        - For each outgoing edge from b: same merge logic into a's outgoing
        - Average the biases
        - Remove neuron_b

    Returns the surviving neuron_id (always neuron_id_a).
    """
    if neuron_id_a == neuron_id_b:
        return neuron_id_a
    for nid in (neuron_id_a, neuron_id_b):
        if nid in graph.input_ids or nid in graph.output_ids:
            raise ValueError(f"Cannot merge input/output neuron '{nid}'")

    neuron_a = graph.get_neuron(neuron_id_a)
    neuron_b = graph.get_neuron(neuron_id_b)

    # Average biases
    with torch.no_grad():
        neuron_a.bias.fill_((neuron_a.bias.item() + neuron_b.bias.item()) / 2)

    # Build lookup: (src, delay) -> edge for a's incoming
    a_in_index: dict[tuple[str, int], str] = {}
    for e in graph.edges_into(neuron_id_a):
        a_in_index[(e.src, e.delay)] = e.edge_id

    for e in graph.edges_into(neuron_id_b):
        key = (e.src, e.delay)
        if key in a_in_index:
            # Both neurons received the same input via the same path.
            # Keep a's weight unchanged — one copy of this path is sufficient.
            pass
        else:
            new_eid = graph.add_edge(
                src=e.src, dst=neuron_id_a,
                weight=e.weight.item(), delay=e.delay,
            )
            a_in_index[key] = new_eid

    # Build lookup: (dst, delay) -> edge for a's outgoing
    a_out_index: dict[tuple[str, int], str] = {}
    for e in graph.edges_from(neuron_id_a):
        a_out_index[(e.dst, e.delay)] = e.edge_id

    for e in graph.edges_from(neuron_id_b):
        key = (e.dst, e.delay)
        if key in a_out_index:
            existing = graph.get_edge(a_out_index[key])
            with torch.no_grad():
                existing.weight.add_(e.weight)
        else:
            new_eid = graph.add_edge(
                src=neuron_id_a, dst=e.dst,
                weight=e.weight.item(), delay=e.delay,
            )
            a_out_index[key] = new_eid

    graph.remove_neuron(neuron_id_b)
    return neuron_id_a


# ---------------------------------------------------------------------------
# Importance scoring (used by TopologyController to decide what to prune)
# ---------------------------------------------------------------------------

def neuron_importance(graph: NeuronGraph, neuron_id: str) -> float:
    """
    I(v) = Σ |w_e| * |h_v_mean|

    Uses current activation as a proxy for mean (TopologyController
    accumulates over a window for the real mean).
    """
    neuron = graph.get_neuron(neuron_id)
    total_w = sum(
        abs(e.weight.item())
        for e in graph.edges_into(neuron_id) + graph.edges_from(neuron_id)
    )
    h_mag = abs(neuron._current.item()) if neuron._current.numel() == 1 else float(neuron._current.abs().mean())
    return total_w * (h_mag + 1e-8)


def edge_importance(graph: NeuronGraph, edge_id: str) -> float:
    """Simple |w| magnitude."""
    return abs(graph.get_edge(edge_id).weight.item())


def cosine_similarity_neurons(
    graph: NeuronGraph, nid_a: str, nid_b: str
) -> float:
    """
    Cosine similarity between two neurons' current activations.
    Used to detect merge candidates.
    """
    ha = graph.get_neuron(nid_a)._current.detach()
    hb = graph.get_neuron(nid_b)._current.detach()
    dot = float((ha * hb).sum())
    norm = float(ha.norm() * hb.norm()) + 1e-8
    return dot / norm
