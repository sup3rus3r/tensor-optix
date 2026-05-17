from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .neuron import GRUNeuron, LSTMNeuron, Neuron
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

    def __init__(self, dale_mode: str = "clamp") -> None:
        """
        dale_mode: 'clamp' (default) or 'softplus'.

        'clamp'    — enforce_dale() post-step clamp (gradient-dead at boundary).
        'softplus' — raw parameter θ; effective_weight = softplus(θ) * sign.
                     Zero-weight init uses θ=-10 (softplus(-10)≈4.5e-5).
                     enforce_dale() is a no-op in this mode.
        """
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

        # Dale's law mode
        self._dale_mode: str = dale_mode

        # Cached device — initialised from the global registry, updated on .to()
        self._device: torch.device = get_device()

        # -------------------------------------------------------------------
        # Matrix cache — rebuilt lazily on topology change
        # _neuron_index: stable ordered list of all neuron IDs (defines W rows/cols)
        # _nid_to_idx:   inverse map, rebuilt when dirty
        # _ff_dst/_ff_src: pre-built index tensors for d=0 edge scatter
        # _ff_params:    ordered list of d=0 edge weight Parameters
        # _rec_edges:    all delay>0 edges (rare; kept as list)
        # _topo_order:   cached topological order of non-input neurons
        # _input_pos:    maps input neuron id -> position in obs vector
        # -------------------------------------------------------------------
        self._neuron_index: List[str] = []
        self._nid_to_idx: Dict[str, int] = {}
        self._matrix_dirty: bool = True
        self._ff_dst: Optional[torch.Tensor] = None
        self._ff_src: Optional[torch.Tensor] = None
        # Single packed Parameter for all d=0 edge weights — rebuilt on topology change.
        # Eliminates per-forward torch.stack over thousands of individual Parameters.
        # Starts as None so it doesn't register as a parameter until first rebuild.
        self._ff_weight_vec: Optional[nn.Parameter] = None
        self._ff_params: List[nn.Parameter] = []   # unused after first rebuild; kept for compat
        self._ff_cell_types: List[str] = []
        self._rec_edges: List[Edge] = []
        self._topo_order: List[str] = []
        # Per-level activation groups: List[List[(idx_tensor, nid_list, act_fn)]]
        # Outer list = topo levels; inner = neurons grouped by activation within that level.
        self._level_act_groups: list = []
        self._is_uniform: bool = False   # True → use _fast_forward (no rec edges, no GRU/LSTM)
        self._input_pos: Dict[str, int] = {}

        # Compiled forward — rebuilt after every topology change and device move.
        # Falls back to _raw_forward on PyTorch < 2.0.
        self._fwd = self._make_compiled_fwd()

    def state_dict(self, *args, **kwargs):
        # Ensure packed layout before saving so ff weights appear as _ff_weight_vec,
        # not as individual _edge_weights.* keys that were added by add_edge().
        if self._matrix_dirty and self._neurons:
            self._rebuild_matrix_structure()
        return super().state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict: bool = True):
        # Ensure packed layout so the module's parameter keys match the checkpoint.
        if self._matrix_dirty and self._neurons:
            self._rebuild_matrix_structure()
        return super().load_state_dict(state_dict, strict=strict)

    def to(self, *args, **kwargs):
        result = super().to(*args, **kwargs)
        try:
            result._device = next(result.parameters()).device
        except StopIteration:
            pass
        result._matrix_dirty = True  # index tensors must be rebuilt on new device
        # _current and _history are plain tensors — not moved by nn.Module.to().
        # Call reset_state() on each neuron so they land on the new device.
        # (reset_state() uses self.bias.device, which super().to() already moved.)
        result.reset_state()
        result.invalidate_compile()
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
        cell_type: str = "any",
        neuron: Optional[Neuron] = None,
    ) -> str:
        """
        Add a neuron, return its id.  role: 'input' | 'hidden' | 'output'.

        Pass a pre-constructed Neuron instance (GRUNeuron, LSTMNeuron, etc.)
        via the `neuron` kwarg for heterogeneous graphs.  When `neuron` is
        given the activation/neuron_id/max_delay/cell_type kwargs are ignored.
        """
        if neuron is not None:
            n = neuron
        else:
            n = Neuron(activation=activation, neuron_id=neuron_id, max_delay=max_delay, cell_type=cell_type)
        n = n.to(self._device)
        n.reset_state()   # _current/_history are plain tensors; .to() doesn't move them
        nid = n.neuron_id
        self._neurons[nid] = n
        self._in_edges[nid] = []
        if role == "input":
            self._input_ids.append(nid)
        elif role == "output":
            self._output_ids.append(nid)
        else:
            self._hidden_ids.append(nid)
        self._neuron_index.append(nid)
        self._nid_to_idx[nid] = len(self._neuron_index) - 1
        self._matrix_dirty = True
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
        init_val = self._softplus_init(src, weight)
        param = nn.Parameter(torch.tensor(init_val, dtype=torch.float32, device=self._device))
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

        self._matrix_dirty = True
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
        self._matrix_dirty = True

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
        if neuron_id in self._neuron_index:
            self._neuron_index.remove(neuron_id)
        self._matrix_dirty = True

    # ------------------------------------------------------------------
    # Compile lifecycle
    # ------------------------------------------------------------------

    def _make_compiled_fwd(self):
        # NeuronGraph forward mutates Python-side neuron state (_current, push_history).
        # torch.compile traces pure computation and does not replay Python side-effects,
        # so we run the forward pass in eager mode. Compile is available via
        # compile_forward() for users who manage state externally.
        return self._raw_forward

    def compile_forward(self):
        """
        Replace the forward pass with a torch.compile'd version.

        Call this ONLY when the graph topology is static and you manage neuron
        state (reset_state, push_history) manually outside the forward call.
        Not recommended for general use — prefer the default eager forward.
        """
        if not hasattr(torch, "compile"):
            return
        import sys
        backend = "inductor" if sys.platform != "win32" else "aot_eager"
        self._fwd = torch.compile(self._raw_forward, backend=backend, dynamic=False)

    def invalidate_compile(self) -> None:
        """
        Reset the forward function after a topology change or device move.

        In eager mode (default): rebuilds matrix cache and resets to _raw_forward.
        If compile_forward() was previously called: resets dynamo state and
        re-compiles with the new topology.
        """
        if self._neurons:
            self._rebuild_matrix_structure()
        # After rebuild, _fwd is already set to fast or raw path.
        # Only re-invoke compile if user had previously called compile_forward().
        compiled = self._fwd not in (self._raw_forward, self._fast_forward)
        if compiled:
            if hasattr(torch, "_dynamo"):
                torch._dynamo.reset()
            self.compile_forward()

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self._fwd(obs)

    def _raw_forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Run one timestep of the graph.

        obs: 1-D tensor of length len(input_ids)
        Returns: 1-D tensor of length len(output_ids)
        """
        if obs.shape[0] != len(self._input_ids):
            raise ValueError(
                f"obs dim {obs.shape[0]} != {len(self._input_ids)} input neurons"
            )

        if self._matrix_dirty:
            self._rebuild_matrix_structure()

        # h[n]: current activation vector, updated in-place as we walk topo order.
        # Input neurons are seeded from obs. Non-input neurons start from their
        # stored (previous-timestep) state but are overwritten once step() runs.
        h_list: List[torch.Tensor] = []
        for nid in self._neuron_index:
            if nid in self._input_pos:
                val = obs[self._input_pos[nid]]
                self._neurons[nid]._current = val.unsqueeze(0)  # type: ignore
                h_list.append(val)
            else:
                # Previous-timestep value: detached so gradients don't flow across steps.
                h_list.append(self._neurons[nid]._current.detach().squeeze(0))  # type: ignore
        h = torch.stack(h_list)  # [n] — will be updated in topo order below

        # Precompute weight matrix (differentiable — weights change during training).
        W = self._assemble_W()

        # Delay>0 (recurrent) edge contributions — always use previous-timestep history.
        rec_pre: Optional[torch.Tensor] = None
        if self._rec_edges:
            rec_pre = torch.zeros(len(self._neuron_index), device=self._device)
            for edge in self._rec_edges:
                src_neuron: Neuron = self._neurons[edge.src]  # type: ignore
                hist = src_neuron.get_delayed(edge.delay).squeeze(0)
                w = self._effective_rec_weight(edge)
                rec_pre[self._nid_to_idx[edge.dst]] = rec_pre[self._nid_to_idx[edge.dst]] + w * hist

        # Walk non-input neurons in topological order.
        # For each neuron, sum the feedforward inputs from the CURRENT h (which
        # already holds updated values for upstream neurons processed earlier),
        # then call step() and write the result back into h.
        # This makes feedforward propagation complete in a single forward pass.
        for nid in self._topo_order:
            neuron: Neuron = self._neurons[nid]  # type: ignore
            idx = self._nid_to_idx[nid]
            # Feedforward: dot product of W[idx] with current h (row = dst neuron)
            pre_ff = (W[idx] * h).sum()
            pre = pre_ff + (rec_pre[idx] if rec_pre is not None else 0.0)
            neuron.step(pre.unsqueeze(0))
            h = h.clone()
            h[idx] = neuron._current.squeeze(0)  # type: ignore

        # Push all neurons' current activations into history
        for nid in self._neurons:
            self._neurons[nid].push_history()  # type: ignore

        # Collect output
        return torch.cat([
            self._neurons[nid]._current  # type: ignore
            for nid in self._output_ids
        ], dim=0)

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

    # ------------------------------------------------------------------
    # Serialization — topology-aware save / load
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """
        Serialize the full graph topology (structure only — weights come from
        state_dict). The returned dict is JSON-serializable and captures enough
        information for from_dict() to reconstruct an identical graph that will
        accept the original state_dict without key mismatches.
        """
        neurons = []
        for role, ids in [
            ("input",  self._input_ids),
            ("hidden", self._hidden_ids),
            ("output", self._output_ids),
        ]:
            for nid in ids:
                n: Neuron = self._neurons[nid]  # type: ignore
                neurons.append({
                    "neuron_id":       nid,
                    "role":            role,
                    "type":            type(n).__name__,
                    "activation_name": getattr(n, "activation_name", "tanh"),
                    "max_delay":       n._max_delay,
                    "cell_type":       n.cell_type,
                })

        edges = []
        for eid, e in self._edges.items():
            edges.append({
                "edge_id": eid,
                "src":     e.src,
                "dst":     e.dst,
                "delay":   e.delay,
            })

        return {
            "dale_mode": self._dale_mode,
            "neurons":   neurons,
            "edges":     edges,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "NeuronGraph":
        """
        Reconstruct a NeuronGraph from a topology dict produced by to_dict().
        Neurons are added with their original IDs so state_dict keys match
        exactly after a subsequent load_state_dict() call.
        """
        from .neuron import GRUNeuron, LSTMNeuron, TrainableGRUNeuron, TrainableLSTMNeuron

        _type_map = {
            "Neuron":              Neuron,
            "GRUNeuron":           GRUNeuron,
            "LSTMNeuron":          LSTMNeuron,
            "TrainableGRUNeuron":  TrainableGRUNeuron,
            "TrainableLSTMNeuron": TrainableLSTMNeuron,
        }

        graph = cls(dale_mode=d.get("dale_mode", "clamp"))

        for spec in d["neurons"]:
            neuron_cls = _type_map.get(spec["type"], Neuron)
            if spec["type"] == "Neuron":
                neuron = neuron_cls(
                    activation=spec["activation_name"],
                    neuron_id=spec["neuron_id"],
                    max_delay=spec["max_delay"],
                    cell_type=spec["cell_type"],
                )
            else:
                neuron = neuron_cls(
                    neuron_id=spec["neuron_id"],
                    max_delay=spec["max_delay"],
                    cell_type=spec["cell_type"],
                )
            graph.add_neuron(role=spec["role"], neuron=neuron)

        for spec in d["edges"]:
            graph.add_edge(
                src=spec["src"],
                dst=spec["dst"],
                delay=spec["delay"],
                edge_id=spec["edge_id"],
            )

        return graph

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
    # Dale's Law
    # ------------------------------------------------------------------

    def cell_type_of(self, neuron_id: str) -> str:
        """Return the cell_type of a neuron ('excitatory', 'inhibitory', or 'any')."""
        return self.get_neuron(neuron_id).cell_type

    def enforce_dale(self) -> None:
        """
        Enforce Dale's Law after each optimizer step.

        'clamp' mode: clamp outgoing weights (excitatory >= 0, inhibitory <= 0).
        'softplus' mode: no-op — the softplus transform in _assemble_W already
                         enforces the sign constraint at every forward pass.
        """
        if self._dale_mode == "softplus":
            return
        with torch.no_grad():
            for edge in self._edges.values():
                ct = self.get_neuron(edge.src).cell_type
                if ct == "excitatory":
                    edge.weight.clamp_(min=0.0)
                elif ct == "inhibitory":
                    edge.weight.clamp_(max=0.0)

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

    # ------------------------------------------------------------------
    # Softplus Dale helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _softplus_inv(w: float) -> float:
        """θ such that softplus(θ) == w. Requires w > 0."""
        return math.log(math.exp(w) - 1.0)

    def _softplus_init(self, src: str, weight: float) -> float:
        """Convert requested weight to raw θ for softplus mode."""
        if self._dale_mode != "softplus":
            return weight
        ct = self._neurons[src].cell_type
        if ct not in ("excitatory", "inhibitory"):
            return weight
        abs_w = abs(weight)
        if abs_w < 1e-6:
            return -10.0  # softplus(-10) ≈ 4.5e-5 ≈ 0
        return self._softplus_inv(abs_w)

    def _apply_dale(self, raw: torch.Tensor, cell_types: List[str]) -> torch.Tensor:
        """Apply softplus Dale transform to a tensor of raw parameters."""
        if self._dale_mode != "softplus":
            return raw
        sp = F.softplus(raw)
        result = raw.clone()
        for i, ct in enumerate(cell_types):
            if ct == "excitatory":
                result[i] = sp[i]
            elif ct == "inhibitory":
                result[i] = -sp[i]
            # "any": use raw directly (unconstrained)
        return result

    def _effective_rec_weight(self, edge: "Edge") -> torch.Tensor:
        """Effective weight for a recurrent edge (scalar tensor, differentiable)."""
        if self._dale_mode != "softplus":
            return edge.weight
        ct = self.get_neuron(edge.src).cell_type
        if ct == "excitatory":
            return F.softplus(edge.weight)
        if ct == "inhibitory":
            return -F.softplus(edge.weight)
        return edge.weight

    def effective_weight(self, edge_id: str) -> float:
        """
        Return the effective float weight of an edge (post-softplus if applicable).
        Use this instead of edge.weight.item() when dale_mode='softplus'.
        """
        edge = self.get_edge(edge_id)
        if self._dale_mode != "softplus":
            return edge.weight.item()
        ct = self.get_neuron(edge.src).cell_type
        with torch.no_grad():
            if ct == "excitatory":
                return float(F.softplus(edge.weight).item())
            if ct == "inhibitory":
                return float(-F.softplus(edge.weight).item())
        return edge.weight.item()

    def _rebuild_matrix_structure(self) -> None:
        """
        Rebuild all topology-derived caches. Called lazily in forward()
        whenever _matrix_dirty is True (after any add/remove operation).

        All d=0 edge weights are packed into a single _ff_weight_vec Parameter
        so _assemble_W can read them as one tensor instead of stacking thousands
        of individual scalar Parameters every forward pass.
        """
        self._nid_to_idx = {nid: i for i, nid in enumerate(self._neuron_index)}
        self._input_pos = {nid: i for i, nid in enumerate(self._input_ids)}

        ff_dst: List[int] = []
        ff_src: List[int] = []
        ff_cell_types: List[str] = []
        ff_edge_ids: List[str] = []
        rec_edges: List[Edge] = []

        for edge in self._edges.values():
            if edge.delay == 0:
                ff_dst.append(self._nid_to_idx[edge.dst])
                ff_src.append(self._nid_to_idx[edge.src])
                ff_cell_types.append(self.get_neuron(edge.src).cell_type)
                ff_edge_ids.append(edge.edge_id)
            else:
                rec_edges.append(edge)

        if ff_dst:
            self._ff_dst = torch.tensor(ff_dst, dtype=torch.long, device=self._device)
            self._ff_src = torch.tensor(ff_src, dtype=torch.long, device=self._device)
            # Collect current weight values (edge.weight may be an individual
            # Parameter from add_edge() or a view from the previous rebuild).
            with torch.no_grad():
                packed = torch.tensor(
                    [self._edges[eid].weight.item() for eid in ff_edge_ids],
                    dtype=torch.float32,
                    device=self._device,
                )
            self._ff_weight_vec = nn.Parameter(packed)
            # Remove individual ff Parameters from ParameterDict — they are now
            # represented by the packed vector and must not appear twice in state_dict.
            for eid in ff_edge_ids:
                param_key = eid.replace("-", "_")
                if param_key in self._edge_weights:
                    del self._edge_weights[param_key]
            # Update each edge.weight to be an element-view of the packed vector
            # so that enforce_dale(), effective_weight(), and gradient inspection work.
            # retain_grad() ensures edge.weight.grad is populated after backward(),
            # which topology_controller uses for dead-edge detection.
            for i, eid in enumerate(ff_edge_ids):
                view = self._ff_weight_vec[i]
                view.retain_grad()
                self._edges[eid].weight = view
        else:
            self._ff_dst = None
            self._ff_src = None
            self._ff_weight_vec = nn.Parameter(torch.empty(0, device=self._device))

        self._ff_params = []          # no longer used; cleared to free references
        self._ff_cell_types = ff_cell_types
        self._rec_edges = rec_edges
        self._topo_order = self._topological_order()
        self._is_uniform = self._check_uniform()
        self._level_act_groups = self._compute_level_act_groups() if self._is_uniform else []
        self._fwd = self._fast_forward if self._is_uniform else self._raw_forward
        self._matrix_dirty = False

    def _assemble_W(self) -> torch.Tensor:
        """
        Build the [n, n] weight matrix for d=0 edges differentiably.
        Uses _ff_weight_vec (a single Parameter rebuilt on topology change)
        instead of stacking individual scalar Parameters each call.
        In softplus mode, applies softplus(θ)*sign for constrained neurons.
        Called every forward pass.
        """
        n = len(self._neuron_index)
        if self._ff_weight_vec is None or self._ff_weight_vec.numel() == 0:
            return torch.zeros(n, n, device=self._device)
        raw = self._ff_weight_vec                        # [num_ff] — single Parameter, no stack
        weights = self._apply_dale(raw, self._ff_cell_types)
        flat_idx = self._ff_dst * n + self._ff_src       # [num_ff]
        W_flat = torch.zeros(n * n, device=self._device)
        W_flat = W_flat.scatter_add(0, flat_idx, weights)
        return W_flat.view(n, n)

    def _check_uniform(self) -> bool:
        """True when the fast vectorized path is valid: no recurrent edges and
        all non-input neurons are base Neurons (not GRU/LSTM).
        Mixed activations are fine — _compute_level_act_groups handles them."""
        if self._rec_edges:
            return False
        non_input = self._hidden_ids + self._output_ids
        if not non_input:
            return False
        return all(type(self._neurons[nid]) is Neuron for nid in non_input)

    def _compute_level_act_groups(self) -> list:
        """
        BFS topo levels, then group neurons within each level by activation.
        Returns List[List[(idx_tensor, nid_list, act_fn)]].
        One matmul per activation group per level — handles mixed-activation graphs.
        """
        from .neuron import ACTIVATIONS
        non_input = set(self._hidden_ids) | set(self._output_ids)
        in_degree: Dict[str, int] = {nid: 0 for nid in non_input}
        input_set = set(self._input_ids)
        for edge in self._edges.values():
            if edge.delay == 0 and edge.dst in in_degree and edge.src not in input_set:
                in_degree[edge.dst] += 1

        queue = [nid for nid in non_input if in_degree[nid] == 0]
        result = []
        while queue:
            # Group neurons in this level by activation function name
            by_act: Dict[str, List[str]] = {}
            for nid in queue:
                act_name = self._neurons[nid].activation_name  # type: ignore
                by_act.setdefault(act_name, []).append(nid)
            level_groups = []
            for act_name, nids in by_act.items():
                idx_t = torch.tensor(
                    [self._nid_to_idx[nid] for nid in nids],
                    dtype=torch.long, device=self._device,
                )
                level_groups.append((idx_t, nids, ACTIVATIONS[act_name]))
            result.append(level_groups)

            next_q: List[str] = []
            for nid in queue:
                for e in self.edges_from(nid):
                    if e.delay == 0 and e.dst in in_degree:
                        in_degree[e.dst] -= 1
                        if in_degree[e.dst] == 0:
                            next_q.append(e.dst)
            queue = next_q
        return result

    def _fast_forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Vectorized forward for uniform graphs (same activation, no recurrent edges).
        Replaces the per-neuron clone loop with one matmul slice per topo level.
        """
        if obs.shape[0] != len(self._input_ids):
            raise ValueError(
                f"obs dim {obs.shape[0]} != {len(self._input_ids)} input neurons"
            )
        if self._matrix_dirty:
            self._rebuild_matrix_structure()
            if not self._is_uniform:   # topology change made graph irregular — fall back
                return self._raw_forward(obs)

        h_list: List[torch.Tensor] = []
        for nid in self._neuron_index:
            if nid in self._input_pos:
                val = obs[self._input_pos[nid]]
                self._neurons[nid]._current = val.unsqueeze(0)  # type: ignore
                h_list.append(val)
            else:
                h_list.append(self._neurons[nid]._current.detach().squeeze(0))  # type: ignore
        h = torch.stack(h_list)

        W = self._assemble_W()

        for act_groups in self._level_act_groups:
            for level_idx, level_nids, act_fn in act_groups:
                pre = (W[level_idx] * h.unsqueeze(0)).sum(dim=1)   # [k]
                bias = torch.stack([self._neurons[nid].bias.squeeze(0) for nid in level_nids])  # type: ignore
                vals = act_fn(pre + bias)
                h = h.index_put((level_idx,), vals)
                for j, nid in enumerate(level_nids):
                    self._neurons[nid]._current = vals[j].unsqueeze(0)  # type: ignore

        for nid in self._neurons:
            self._neurons[nid].push_history()  # type: ignore

        return torch.cat([
            self._neurons[nid]._current  # type: ignore
            for nid in self._output_ids
        ], dim=0)

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
