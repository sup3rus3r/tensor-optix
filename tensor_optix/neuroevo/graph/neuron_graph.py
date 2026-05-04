from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .neuron import Neuron
from tensor_optix.core.device import get_device


@dataclass
class Edge:
    """A directed, variable-delay weighted edge."""
    edge_id: str
    src: str        # neuron_id
    dst: str        # neuron_id
    weight: nn.Parameter
    delay: int      # timesteps, 0 = feedforward, >=1 = recurrent

    def __repr__(self) -> str:
        return (
            f"Edge({self.src[:6]}->{self.dst[:6]}, "
            f"w={self.weight.item():.4f}, d={self.delay})"
        )


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

    def __init__(self) -> None:
        super().__init__()

        # nn.ModuleDict so PyTorch tracks all neuron parameters
        self._neurons: nn.ModuleDict = nn.ModuleDict()
        # edge weights tracked separately so they show up in parameters()
        self._edge_weights: nn.ParameterDict = nn.ParameterDict()

        # Raw edge metadata (not nn.Modules, just data)
        self._edges: Dict[str, Edge] = {}

        # Role sets
        self._input_ids: List[str] = []
        self._output_ids: List[str] = []
        self._hidden_ids: List[str] = []

        # Adjacency: dst -> list of edge_ids arriving at dst
        self._in_edges: Dict[str, List[str]] = {}

        # Cached device — initialised from the global registry, updated on .to()
        self._device: torch.device = get_device()

    def to(self, *args, **kwargs):
        result = super().to(*args, **kwargs)
        try:
            result._device = next(result.parameters()).device
        except StopIteration:
            pass
        return result

    # ------------------------------------------------------------------
    # Graph mutation API
    # ------------------------------------------------------------------

    def add_neuron(
        self,
        role: str = "hidden",
        activation: str = "tanh",
        neuron_id: Optional[str] = None,
        max_delay: int = 1,
    ) -> str:
        """Add a neuron, return its id. role: 'input' | 'hidden' | 'output'."""
        n = Neuron(activation=activation, neuron_id=neuron_id, max_delay=max_delay)
        n = n.to(self._device)
        nid = n.neuron_id
        self._neurons[nid] = n
        self._in_edges[nid] = []
        if role == "input":
            self._input_ids.append(nid)
        elif role == "output":
            self._output_ids.append(nid)
        else:
            self._hidden_ids.append(nid)
        return nid

    def add_edge(
        self,
        src: str,
        dst: str,
        weight: float = 0.0,
        delay: int = 0,
        edge_id: Optional[str] = None,
    ) -> str:
        """
        Add a directed edge src->dst with given weight and delay.
        Returns the edge_id.
        weight=0.0 default ensures function-preserving insertion.
        """
        if src not in self._neurons:
            raise ValueError(f"src neuron '{src}' not in graph")
        if dst not in self._neurons:
            raise ValueError(f"dst neuron '{dst}' not in graph")

        eid = edge_id or str(uuid.uuid4())
        param = nn.Parameter(torch.tensor(weight, dtype=torch.float32, device=self._device))
        # Sanitize key for ParameterDict (no hyphens)
        param_key = eid.replace("-", "_")
        self._edge_weights[param_key] = param

        edge = Edge(
            edge_id=eid,
            src=src,
            dst=dst,
            weight=param,
            delay=delay,
        )
        self._edges[eid] = edge
        self._in_edges[dst].append(eid)

        # Ensure destination neuron history buffer is deep enough
        dst_neuron: Neuron = self._neurons[dst]  # type: ignore
        if delay > dst_neuron.max_delay:
            dst_neuron.expand_history(delay)

        return eid

    def remove_edge(self, edge_id: str) -> None:
        """Remove an edge and free its parameter."""
        if edge_id not in self._edges:
            return
        edge = self._edges.pop(edge_id)
        param_key = edge_id.replace("-", "_")
        if param_key in self._edge_weights:
            del self._edge_weights[param_key]
        if edge.dst in self._in_edges:
            self._in_edges[edge.dst] = [
                e for e in self._in_edges[edge.dst] if e != edge_id
            ]

    def remove_neuron(self, neuron_id: str) -> None:
        """Remove a neuron and all its incident edges."""
        if neuron_id not in self._neurons:
            return
        # Remove all edges touching this neuron
        to_remove = [
            eid for eid, e in self._edges.items()
            if e.src == neuron_id or e.dst == neuron_id
        ]
        for eid in to_remove:
            self.remove_edge(eid)
        del self._neurons[neuron_id]
        del self._in_edges[neuron_id]
        for lst in (self._input_ids, self._output_ids, self._hidden_ids):
            if neuron_id in lst:
                lst.remove(neuron_id)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Run one timestep of the graph.

        obs: 1-D tensor of length len(input_ids)
        Returns: 1-D tensor of length len(output_ids)
        """
        if obs.shape[0] != len(self._input_ids):
            raise ValueError(
                f"obs dim {obs.shape[0]} != {len(self._input_ids)} input neurons"
            )

        # 1. Inject observations into input neurons (bypass edge aggregation)
        for i, nid in enumerate(self._input_ids):
            neuron: Neuron = self._neurons[nid]  # type: ignore
            neuron._current = obs[i].unsqueeze(0)

        # 2. Compute execution order for non-input neurons
        order = self._topological_order()

        # 3. Forward each neuron in order
        for nid in order:
            neuron: Neuron = self._neurons[nid]  # type: ignore
            pre = torch.zeros(1, device=self._device)
            for eid in self._in_edges.get(nid, []):
                edge = self._edges[eid]
                src_neuron: Neuron = self._neurons[edge.src]  # type: ignore
                h = src_neuron.get_delayed(edge.delay)
                pre = pre + edge.weight * h
            neuron(pre)

        # 4. Push all neurons' current activations into history
        for nid in self._neurons:
            self._neurons[nid].push_history()  # type: ignore

        # 5. Collect output
        outputs = [
            self._neurons[nid]._current  # type: ignore
            for nid in self._output_ids
        ]
        return torch.cat(outputs, dim=0)

    # ------------------------------------------------------------------
    # Topology queries
    # ------------------------------------------------------------------

    def get_neuron(self, neuron_id: str) -> Neuron:
        return self._neurons[neuron_id]  # type: ignore

    def get_edge(self, edge_id: str) -> Edge:
        return self._edges[edge_id]

    def edges_into(self, neuron_id: str) -> List[Edge]:
        return [self._edges[eid] for eid in self._in_edges.get(neuron_id, [])]

    def edges_from(self, neuron_id: str) -> List[Edge]:
        return [e for e in self._edges.values() if e.src == neuron_id]

    def all_edges(self) -> List[Edge]:
        return list(self._edges.values())

    def all_neuron_ids(self) -> List[str]:
        return list(self._neurons.keys())

    @property
    def input_ids(self) -> List[str]:
        return list(self._input_ids)

    @property
    def output_ids(self) -> List[str]:
        return list(self._output_ids)

    @property
    def hidden_ids(self) -> List[str]:
        return list(self._hidden_ids)

    def n_neurons(self) -> int:
        return len(self._neurons)

    def n_edges(self) -> int:
        return len(self._edges)

    # ------------------------------------------------------------------
    # Episode reset
    # ------------------------------------------------------------------

    def reset_state(self) -> None:
        """Zero all neuron histories. Call at episode start."""
        for nid in self._neurons:
            self._neurons[nid].reset_state()  # type: ignore

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _topological_order(self) -> List[str]:
        """
        Kahn's algorithm over feedforward (d=0) edges only.
        Recurrent edges (d>=1) are ignored for ordering — they always read
        from history, so there is no within-timestep dependency.
        Nodes with no feedforward inputs are processed first.
        Input neurons are excluded (already resolved).
        """
        non_input = set(self._hidden_ids) | set(self._output_ids)
        in_degree: Dict[str, int] = {nid: 0 for nid in non_input}

        input_set = set(self._input_ids)
        for edge in self._edges.values():
            # Only count edges from non-input sources — input nodes are
            # pre-resolved before this loop, so they never decrement in_degree.
            if edge.delay == 0 and edge.dst in in_degree and edge.src not in input_set:
                in_degree[edge.dst] += 1

        queue = [nid for nid, deg in in_degree.items() if deg == 0]
        order: List[str] = []

        while queue:
            nid = queue.pop(0)
            order.append(nid)
            for out_edge in self.edges_from(nid):
                if out_edge.delay == 0 and out_edge.dst in in_degree:
                    in_degree[out_edge.dst] -= 1
                    if in_degree[out_edge.dst] == 0:
                        queue.append(out_edge.dst)

        # Any node not reached has only recurrent inputs — append at end
        remaining = [nid for nid in non_input if nid not in order]
        return order + remaining
