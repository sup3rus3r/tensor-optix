from __future__ import annotations

"""
BrainNetwork — a container of named NeuronGraph regions with sparse,
learnable inter-region pathways.

                ┌─────────────┐      pathway      ┌─────────────┐
  observations →│   sensory   │ ─────────────────▶ │   memory    │
                └─────────────┘                    └──────┬──────┘
                                                          │ pathway
                                                   ┌──────▼──────┐      actions
                                                   │  executive  │ ──▶ (output)
                                                   └─────────────┘

Each region is an independent NeuronGraph that can be evolved by its own
TopologyController. Pathways are sparse sets of learnable edges that cross
region boundaries — they live in BrainNetwork itself, not inside either region.

Forward pass:
  1. Run each region in topological order (regions form a DAG via pathways;
     recurrent inter-region pathways use delay >= 1 from the source region's
     output history).
  2. Inject pathway signals into destination regions as additional pre-activation
     inputs, accumulated before that region's own forward pass.
  3. Return the activations of all output neurons across all regions (or a
     named subset if output_regions is specified).

Usage example::

    from tensor_optix.neuroevo.brain_network import BrainNetwork
    from tensor_optix.neuroevo.graph.neuron_graph import NeuronGraph

    sensory   = NeuronGraph()
    executive = NeuronGraph()
    # ... build each graph ...

    brain = BrainNetwork()
    brain.add_region("sensory",   sensory)
    brain.add_region("executive", executive)
    brain.add_pathway("sensory", "executive", n_connections=4, delay=1)

    out = brain(obs_tensor)
"""

import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .graph.neuron_graph import NeuronGraph, Edge
from .graph.neuron import Neuron
from tensor_optix.core.device import get_device


@dataclass
class InterRegionEdge:
    """A single learnable edge crossing two regions."""
    edge_id: str
    src_region: str
    src_neuron: str
    dst_region: str
    dst_neuron: str
    weight: nn.Parameter
    delay: int  # timesteps; 0 = same-step (requires src before dst in region order)


@dataclass
class Pathway:
    """A named bundle of InterRegionEdges from one region to another."""
    pathway_id: str
    src_region: str
    dst_region: str
    edge_ids: List[str] = field(default_factory=list)


class BrainNetwork(nn.Module):
    """
    A collection of named NeuronGraph regions connected by sparse inter-region
    pathways.

    Parameters
    ----------
    name : str
        Optional human-readable name (for logging / repr).
    output_regions : list[str] | None
        If set, forward() only collects outputs from these regions.
        If None, all regions contribute to the output tensor.
    """

    def __init__(
        self,
        name: str = "brain",
        output_regions: Optional[List[str]] = None,
    ) -> None:
        super().__init__()

        self.name = name
        self._output_regions: Optional[List[str]] = output_regions

        # Region graphs tracked as submodules so their parameters are visible
        self._regions: nn.ModuleDict = nn.ModuleDict()
        self._region_order: List[str] = []  # insertion order

        # Inter-region edge weights tracked as parameters
        self._pathway_weights: nn.ParameterDict = nn.ParameterDict()

        # Metadata (not nn.Modules)
        self._edges: Dict[str, InterRegionEdge] = {}
        self._pathways: Dict[str, Pathway] = {}

        # dst_region -> list of edge_ids arriving at that region
        self._region_in_edges: Dict[str, List[str]] = {}

        # Per-region output history for delayed inter-region edges
        # region_name -> deque of past output tensors (list for simplicity)
        self._region_output_history: Dict[str, List[torch.Tensor]] = {}

        self._device: torch.device = get_device()

    def to(self, *args, **kwargs):
        result = super().to(*args, **kwargs)
        try:
            result._device = next(result.parameters()).device
        except StopIteration:
            pass
        for name in result._region_order:
            region = result._regions[name]
            try:
                region._device = next(region.parameters()).device
            except StopIteration:
                pass
            region.reset_state()
        return result

    # ------------------------------------------------------------------
    # Region management
    # ------------------------------------------------------------------

    def add_region(self, name: str, graph: NeuronGraph) -> None:
        """Register a NeuronGraph as a named region."""
        if name in self._regions:
            raise ValueError(f"Region '{name}' already exists.")
        graph = graph.to(self._device)
        self._regions[name] = graph
        self._region_order.append(name)
        self._region_in_edges[name] = []
        self._region_output_history[name] = []

    def get_region(self, name: str) -> NeuronGraph:
        return self._regions[name]  # type: ignore

    @property
    def region_names(self) -> List[str]:
        return list(self._region_order)

    # ------------------------------------------------------------------
    # Pathway management
    # ------------------------------------------------------------------

    def add_pathway(
        self,
        src_region: str,
        dst_region: str,
        n_connections: int = 4,
        delay: int = 1,
        weight_init: float = 0.0,
        pathway_id: Optional[str] = None,
    ) -> str:
        """
        Create a sparse pathway from src_region to dst_region.

        Randomly pairs n_connections output neurons from src with input/hidden
        neurons of dst. All weights initialised to weight_init (default 0.0,
        function-preserving).

        Returns the pathway_id.
        """
        if src_region not in self._regions:
            raise ValueError(f"src_region '{src_region}' not found.")
        if dst_region not in self._regions:
            raise ValueError(f"dst_region '{dst_region}' not found.")

        src_graph: NeuronGraph = self._regions[src_region]  # type: ignore
        dst_graph: NeuronGraph = self._regions[dst_region]  # type: ignore

        src_candidates = src_graph.output_ids + src_graph.hidden_ids
        dst_candidates = dst_graph.input_ids + dst_graph.hidden_ids

        if not src_candidates:
            raise ValueError(f"Region '{src_region}' has no output or hidden neurons to project from.")
        if not dst_candidates:
            raise ValueError(f"Region '{dst_region}' has no input or hidden neurons to project to.")

        pid = pathway_id or str(uuid.uuid4())
        pathway = Pathway(pathway_id=pid, src_region=src_region, dst_region=dst_region)

        import random
        for _ in range(n_connections):
            src_nid = random.choice(src_candidates)
            dst_nid = random.choice(dst_candidates)
            eid = str(uuid.uuid4())
            param_key = eid.replace("-", "_")
            param = nn.Parameter(
                torch.tensor(weight_init, dtype=torch.float32, device=self._device)
            )
            self._pathway_weights[param_key] = param
            edge = InterRegionEdge(
                edge_id=eid,
                src_region=src_region,
                src_neuron=src_nid,
                dst_region=dst_region,
                dst_neuron=dst_nid,
                weight=param,
                delay=delay,
            )
            self._edges[eid] = edge
            self._region_in_edges[dst_region].append(eid)
            pathway.edge_ids.append(eid)

        self._pathways[pid] = pathway
        return pid

    def add_inter_region_edge(
        self,
        src_region: str,
        src_neuron: str,
        dst_region: str,
        dst_neuron: str,
        weight: float = 0.0,
        delay: int = 1,
        edge_id: Optional[str] = None,
    ) -> str:
        """Add a single hand-crafted inter-region edge. Returns edge_id."""
        for rname, nid in [(src_region, src_neuron), (dst_region, dst_neuron)]:
            if rname not in self._regions:
                raise ValueError(f"Region '{rname}' not found.")
            graph: NeuronGraph = self._regions[rname]  # type: ignore
            if nid not in graph.all_neuron_ids():
                raise ValueError(f"Neuron '{nid}' not in region '{rname}'.")

        eid = edge_id or str(uuid.uuid4())
        param_key = eid.replace("-", "_")
        param = nn.Parameter(torch.tensor(weight, dtype=torch.float32, device=self._device))
        self._pathway_weights[param_key] = param
        edge = InterRegionEdge(
            edge_id=eid,
            src_region=src_region,
            src_neuron=src_neuron,
            dst_region=dst_region,
            dst_neuron=dst_neuron,
            weight=param,
            delay=delay,
        )
        self._edges[eid] = edge
        self._region_in_edges[dst_region].append(eid)
        return eid

    def remove_inter_region_edge(self, edge_id: str) -> None:
        if edge_id not in self._edges:
            return
        edge = self._edges.pop(edge_id)
        param_key = edge_id.replace("-", "_")
        if param_key in self._pathway_weights:
            del self._pathway_weights[param_key]
        dst = edge.dst_region
        if dst in self._region_in_edges:
            self._region_in_edges[dst] = [
                e for e in self._region_in_edges[dst] if e != edge_id
            ]
        for pathway in self._pathways.values():
            if edge_id in pathway.edge_ids:
                pathway.edge_ids.remove(edge_id)

    def get_pathway(self, pathway_id: str) -> Pathway:
        return self._pathways[pathway_id]

    def all_inter_region_edges(self) -> List[InterRegionEdge]:
        return list(self._edges.values())

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, region_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Run one timestep of the entire brain network.

        Parameters
        ----------
        region_inputs : dict[str, Tensor]
            Mapping of region_name -> 1-D observation tensor.
            Only regions that receive external observations need an entry.
            Regions driven purely by inter-region pathways can be omitted
            (they will receive a zero observation vector of the right size).

        Returns
        -------
        Tensor
            Concatenated output activations from all output_regions
            (or all regions if output_regions is None).
        """
        execution_order = self._region_execution_order()

        # Accumulate incoming inter-region signals per region before running it.
        # pre_inject[region][neuron_id] = summed pre-activation contribution
        pre_inject: Dict[str, Dict[str, torch.Tensor]] = {r: {} for r in self._region_order}

        for eid in self._edges:
            edge = self._edges[eid]
            src_graph: NeuronGraph = self._regions[edge.src_region]  # type: ignore
            src_neuron: Neuron = src_graph.get_neuron(edge.src_neuron)

            if edge.delay == 0:
                h = src_neuron._current
            else:
                history = self._region_output_history.get(edge.src_region, [])
                idx = edge.delay - 1
                if idx < len(history):
                    # history stores full output tensors; find the neuron's position
                    src_out_ids = src_graph.output_ids
                    if edge.src_neuron in src_out_ids:
                        pos = src_out_ids.index(edge.src_neuron)
                        h = history[idx][pos].unsqueeze(0)
                    else:
                        h = src_neuron.get_delayed(edge.delay)
                else:
                    h = torch.zeros(1, device=self._device)

            contrib = edge.weight * h
            dst_injects = pre_inject[edge.dst_region]
            if edge.dst_neuron in dst_injects:
                dst_injects[edge.dst_neuron] = dst_injects[edge.dst_neuron] + contrib
            else:
                dst_injects[edge.dst_neuron] = contrib

        # Run each region in order, injecting pathway signals
        region_outputs: Dict[str, torch.Tensor] = {}

        for region_name in execution_order:
            graph: NeuronGraph = self._regions[region_name]  # type: ignore
            inject = pre_inject[region_name]

            # Build observation tensor for this region
            if region_name in region_inputs:
                obs = region_inputs[region_name].to(self._device)
            else:
                obs = torch.zeros(len(graph.input_ids), device=self._device)

            # Inject inter-region signals as temporary bias offsets so they
            # participate in the region's forward pass, then restore.
            if inject:
                self._apply_injections(graph, inject)
            out = graph(obs)
            if inject:
                self._unapply_injections(graph, inject)

            region_outputs[region_name] = out

            # Store output in history (newest first)
            hist = self._region_output_history[region_name]
            hist.insert(0, out.detach())
            # Trim history to the max delay needed by outgoing edges
            max_d = self._max_outgoing_delay(region_name)
            if max_d > 0:
                self._region_output_history[region_name] = hist[:max_d]

        # Collect outputs
        output_regions = self._output_regions or self._region_order
        parts = [region_outputs[r] for r in output_regions if r in region_outputs]
        if not parts:
            return torch.zeros(1, device=self._device)
        return torch.cat(parts, dim=0)

    # ------------------------------------------------------------------
    # Episode reset
    # ------------------------------------------------------------------

    def reset_state(self) -> None:
        """Zero all region histories and neuron states. Call between episodes."""
        for name in self._region_order:
            graph: NeuronGraph = self._regions[name]  # type: ignore
            graph.reset_state()
            self._region_output_history[name] = []

    # ------------------------------------------------------------------
    # Dale's Law across all regions
    # ------------------------------------------------------------------

    def enforce_dale(self) -> None:
        """Enforce Dale's Law on every region graph."""
        for name in self._region_order:
            graph: NeuronGraph = self._regions[name]  # type: ignore
            graph.enforce_dale()

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        """Return a dict of per-region neuron/edge counts plus inter-region edge count."""
        info: dict = {"inter_region_edges": len(self._edges), "regions": {}}
        for name in self._region_order:
            graph: NeuronGraph = self._regions[name]  # type: ignore
            info["regions"][name] = {
                "n_neurons": graph.n_neurons(),
                "n_edges": graph.n_edges(),
            }
        return info

    def __repr__(self) -> str:
        parts = [f"BrainNetwork(name={self.name!r}, regions=["]
        for name in self._region_order:
            graph: NeuronGraph = self._regions[name]  # type: ignore
            parts.append(f"  {name}: {graph.n_neurons()} neurons, {graph.n_edges()} edges")
        parts.append(f"], inter_region_edges={len(self._edges)})")
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _region_execution_order(self) -> List[str]:
        """
        Topological sort of regions using feedforward (delay=0) inter-region edges.
        Regions with only delayed incoming inter-region edges are treated as
        having no dependency (they read from history) and can run in any order.
        Falls back to insertion order for regions with no pathway constraints.
        """
        in_degree: Dict[str, int] = {r: 0 for r in self._region_order}
        for edge in self._edges.values():
            if edge.delay == 0:
                in_degree[edge.dst_region] += 1

        queue = [r for r in self._region_order if in_degree[r] == 0]
        order: List[str] = []
        visited = set()

        while queue:
            r = queue.pop(0)
            if r in visited:
                continue
            visited.add(r)
            order.append(r)
            for edge in self._edges.values():
                if edge.src_region == r and edge.delay == 0:
                    in_degree[edge.dst_region] -= 1
                    if in_degree[edge.dst_region] == 0:
                        queue.append(edge.dst_region)

        # Append any remaining (recurrent-only loops)
        for r in self._region_order:
            if r not in visited:
                order.append(r)

        return order

    def _apply_injections(
        self, graph: NeuronGraph, inject: Dict[str, torch.Tensor]
    ) -> None:
        """
        Add pathway contributions to the pre-activation of target neurons.
        We modify _current on input neurons (which have no edge aggregation)
        and will be summed into pre-activation for hidden/output neurons via a
        small shim: we add a temporary zero-weight virtual contribution by
        directly offsetting the neuron bias temporarily.

        Simpler approach used here: store injected pre-activation on the neuron
        as _pathway_inject so the graph's forward() can pick it up.
        We patch the neurons directly before the graph runs.
        """
        for nid, contrib in inject.items():
            if nid in graph.all_neuron_ids():
                neuron: Neuron = graph.get_neuron(nid)
                # Add pathway signal to the neuron's bias temporarily.
                # This is safe because bias is a Parameter and we use no_grad.
                with torch.no_grad():
                    neuron.bias.add_(contrib.squeeze())

    def _unapply_injections(
        self, graph: NeuronGraph, inject: Dict[str, torch.Tensor]
    ) -> None:
        """Undo the temporary bias injection after the region's forward pass."""
        for nid, contrib in inject.items():
            if nid in graph.all_neuron_ids():
                neuron: Neuron = graph.get_neuron(nid)
                with torch.no_grad():
                    neuron.bias.sub_(contrib.squeeze())

    def _max_outgoing_delay(self, region_name: str) -> int:
        max_d = 0
        for edge in self._edges.values():
            if edge.src_region == region_name:
                max_d = max(max_d, edge.delay)
        return max_d
