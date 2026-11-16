from __future__ import annotations

import math
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .neuron import (
    ACTIVATIONS, GRUNeuron, LSTMNeuron, Neuron,
    TrainableGRUNeuron, TrainableLSTMNeuron,
)
from tensor_optix.core.device import get_device

_GRU_ATTRS = ("wz", "uz", "bz", "wr", "ur", "br", "wn", "un", "bn")
_LSTM_ATTRS = ("wf", "uf", "bf", "wi", "ui", "bi", "wg", "ug", "bg", "wo", "uo", "bo")


@dataclass
class EdgeTable:
    """Every edge of a graph as aligned tensors (see NeuronGraph.edge_table)."""
    edge_ids: List[str]
    src: torch.Tensor       # [E] source neuron index
    dst: torch.Tensor       # [E] destination neuron index
    weight: torch.Tensor    # [E] effective (post-Dale) weight, detached
    grad: torch.Tensor      # [E] gradient of the stored parameter, 0 if none yet


@dataclass
class GraphState:
    """Explicit recurrent state for NeuronGraph.forward_batch().

    Columns follow the graph's neuron index order (graph.all_neuron_ids()).
    """
    h: torch.Tensor                 # [B, n] activations of the previous step
    c: torch.Tensor                 # [B, n] LSTM cell state (zero for other neurons)
    hist: Optional[torch.Tensor]    # [B, max_delay, n] past activations, newest first

    def detach(self) -> "GraphState":
        """Cut the gradient path into earlier steps (truncated BPTT)."""
        return GraphState(
            h=self.h.detach(),
            c=self.c.detach(),
            hist=None if self.hist is None else self.hist.detach(),
        )


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

    def __init__(self, dale_mode: str = "clamp", device=None) -> None:
        """
        dale_mode: 'clamp' (default) or 'softplus'.

        'clamp'    — enforce_dale() post-step clamp (gradient-dead at boundary).
        'softplus' — raw parameter θ; effective_weight = softplus(θ) * sign.
                     Zero-weight init uses θ=-10 (softplus(-10)≈4.5e-5).
                     enforce_dale() is a no-op in this mode.

        device: where the graph lives. None uses the global tensor-optix
        device (tensor_optix.set_device / auto_device), which is "cpu" unless
        you change it. Pass "cuda" (or call tx.auto_device()) to run on GPU.
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
        self._device: torch.device = (
            torch.device(device) if device is not None else get_device()
        )

        # Per-step activation recorder, filled by forward() and forward_batch()
        # and drained by TopologyController. None = recording off.
        self._trace: Optional[deque] = None
        self._trace_struct: int = -1

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
        # Edge id of each entry of _ff_weight_vec, in packed order. Replaced
        # (never mutated) on rebuild, so holders can keep a reference to the
        # order they last saw and map old packed positions to new ones by id.
        self._ff_edge_ids: List[str] = []
        # Dale sign per packed entry: +1 excitatory source, -1 inhibitory, 0 free.
        self._ff_sign: Optional[torch.Tensor] = None
        self._rec_sign: Optional[torch.Tensor] = None
        self._rec_src: Optional[torch.Tensor] = None
        self._rec_dst: Optional[torch.Tensor] = None
        self._rec_delay: Optional[torch.Tensor] = None
        self._max_rec_delay: int = 0
        # Vectorized execution plan for _raw_forward / forward_batch.
        self._plan: list = []
        self._input_idx: Optional[torch.Tensor] = None
        self._output_idx: Optional[torch.Tensor] = None
        self._lstm_idx: Optional[torch.Tensor] = None
        self._lstm_neurons: List[Neuron] = []
        self._state_src: List[Tuple[Neuron, bool]] = []
        self._custom_types: List[str] = []
        # Activations of the most recent forward(), [n] in neuron index order.
        self._last_h: Optional[torch.Tensor] = None
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

        # Bumped every time _rebuild_matrix_structure() replaces _ff_weight_vec
        # with a new Parameter object. Callers that hold parameter references
        # elsewhere (e.g. an optimizer) can compare this to detect that their
        # captured parameters are now stale and must be re-fetched.
        self._struct_version: int = 0
        # Zero-argument callables run after every forward() — lets observers
        # such as HebbianHook sample per-step activations without the caller
        # having to invoke them manually after each act().
        self._post_forward_hooks: list = []

    def ensure_built(self) -> None:
        """Force a matrix-structure rebuild now if one is pending.

        Call this before capturing self.parameters() externally (e.g. to
        build an optimizer) so the captured objects are the live ones the
        forward pass actually uses, not soon-to-be-orphaned per-edge Parameters.
        """
        if self._matrix_dirty and self._neurons:
            self._rebuild_matrix_structure()

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

        # A delayed edge reads the SOURCE neuron's history, so the source is
        # the one that must keep a deep enough buffer and push into it.
        if delay >= 1:
            src_neuron: Neuron = self._neurons[src]  # type: ignore
            if delay > src_neuron.max_delay:
                src_neuron.expand_history(delay)
            src_neuron._has_delayed_readers = True  # type: ignore

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
        # _has_delayed_readers is recomputed for every neuron on the next rebuild.
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
        out = self._fwd(obs)
        for hook in self._post_forward_hooks:
            hook()
        return out

    # ------------------------------------------------------------------
    # Activation recording (feeds TopologyController's merge / importance
    # statistics, from this graph's own forwards or from an external one)
    # ------------------------------------------------------------------

    @property
    def neuron_index(self) -> List[str]:
        """Neuron ids in the column order used by every activation tensor."""
        return list(self._neuron_index)

    def enable_activation_trace(self, capacity: int = 100) -> None:
        """Start keeping the last `capacity` forward calls' activations."""
        if self._trace is None or self._trace.maxlen != capacity:
            self._trace = deque(self._trace or (), maxlen=capacity)

    def record_activations(self, h: torch.Tensor) -> None:
        """
        Report activations from a forward pass, shape [n] or [S, n], columns
        in neuron_index order. forward() and forward_batch() call this
        themselves; call it from your own forward pass if you bypass both and
        still want the controller to see what the network is doing.
        """
        self.ensure_built()   # tag rows with the structure they were produced under
        h = h.detach()
        self._last_h = h[-1] if h.dim() == 2 else h
        self._last_h_struct = self._struct_version
        if self._trace is not None:
            self._trace.append((self._struct_version, h.reshape(-1, h.shape[-1])))

    def drain_activation_trace(self) -> Optional[torch.Tensor]:
        """
        Return every activation row recorded since the last drain as one
        [S, n] tensor (neuron_index order) and clear the recorder. Rows
        recorded under an older topology are dropped. None if nothing usable.
        """
        if not self._trace:
            return None
        rows = [t for ver, t in self._trace if ver == self._struct_version]
        self._trace.clear()
        if not rows:
            return None
        return torch.cat(rows, dim=0)

    def activation_snapshot(self) -> torch.Tensor:
        """Latest activation of every neuron as an [n] tensor (neuron_index order)."""
        if self._matrix_dirty and self._neurons:
            self._rebuild_matrix_structure()
        last = getattr(self, "_last_h", None)
        if last is not None and getattr(self, "_last_h_struct", -1) == self._struct_version:
            return last
        return torch.cat([
            self._neurons[nid]._current.detach().reshape(-1)[:1]  # type: ignore
            for nid in self._neuron_index
        ]).to(self._device)

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

        self.record_activations(h)

        # Push history only for neurons that have delayed incoming edges
        for nid in self._neurons:
            if self._neurons[nid]._has_delayed_readers:  # type: ignore
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

    def to_dict(self, include_weights: bool = True) -> dict:
        """
        Serialize the graph. The returned dict is JSON-serializable.

        include_weights=True (default) stores every parameter under "state", so
        from_dict() returns a graph that computes the same function. With
        include_weights=False only the structure is stored and from_dict()
        gives back freshly initialised weights: use that only when you save
        state_dict() next to it yourself (GraphAgent checkpoints do).
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

        out = {
            "dale_mode": self._dale_mode,
            "neurons":   neurons,
            "edges":     edges,
        }
        if include_weights:
            out["state"] = {
                k: v.detach().cpu().tolist() for k, v in self.state_dict().items()
            }
        return out

    def _load_state_lists(self, state: dict) -> None:
        self.load_state_dict({
            k: torch.tensor(v, dtype=torch.float32, device=self._device)
            for k, v in state.items()
        })

    @classmethod
    def from_dict(cls, d: dict, device=None) -> "NeuronGraph":
        """
        Reconstruct a NeuronGraph from a dict produced by to_dict().
        Neurons are added with their original IDs so state_dict keys match
        exactly after a subsequent load_state_dict() call. If the dict carries
        weights ("state") they are loaded; if it was made with
        include_weights=False the weights are freshly initialised and you must
        call load_state_dict() yourself.
        """
        from .neuron import GRUNeuron, LSTMNeuron, TrainableGRUNeuron, TrainableLSTMNeuron

        _type_map = {
            "Neuron":              Neuron,
            "GRUNeuron":           GRUNeuron,
            "LSTMNeuron":          LSTMNeuron,
            "TrainableGRUNeuron":  TrainableGRUNeuron,
            "TrainableLSTMNeuron": TrainableLSTMNeuron,
        }

        graph = cls(dale_mode=d.get("dale_mode", "clamp"), device=device)
        graph._populate_from_dict(d)
        if "state" in d:
            graph._load_state_lists(d["state"])
        return graph

    def restore_from_dict(self, d: dict) -> "NeuronGraph":
        """
        Rebuild THIS graph object in place from a topology dict.

        Checkpoint restore must not swap in a new NeuronGraph instance:
        TopologyController, HebbianHook and BrainNetwork all hold references
        to the graph object, and would silently keep operating on the
        discarded one. Parameters are re-created; call load_state_dict()
        afterwards to restore their values.
        """
        for nid in list(self._neurons.keys()):
            del self._neurons[nid]
        for key in list(self._edge_weights.keys()):
            del self._edge_weights[key]
        self._edges.clear()
        self._input_ids.clear()
        self._output_ids.clear()
        self._hidden_ids.clear()
        self._in_edges.clear()
        self._neuron_index.clear()
        self._nid_to_idx.clear()
        self._ff_dst = None
        self._ff_src = None
        self._ff_weight_vec = None
        self._ff_cell_types = []
        self._rec_edges = []
        self._topo_order = []
        self._level_act_groups = []
        self._is_uniform = False
        self._input_pos = {}
        self._dale_mode = d.get("dale_mode", "clamp")
        self._matrix_dirty = True
        self._ff_edge_ids = []
        self._last_h = None
        if self._trace is not None:
            self._trace.clear()
        self._populate_from_dict(d)
        self.invalidate_compile()
        if "state" in d:
            self._load_state_lists(d["state"])
        return self

    def _populate_from_dict(self, d: dict) -> None:
        from .neuron import GRUNeuron, LSTMNeuron, TrainableGRUNeuron, TrainableLSTMNeuron

        _type_map = {
            "Neuron":              Neuron,
            "GRUNeuron":           GRUNeuron,
            "LSTMNeuron":          LSTMNeuron,
            "TrainableGRUNeuron":  TrainableGRUNeuron,
            "TrainableLSTMNeuron": TrainableLSTMNeuron,
        }
        graph = self

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
            if self._matrix_dirty or self._ff_sign is None:
                # Edges not packed yet: each one is still its own Parameter.
                for edge in self._edges.values():
                    ct = self.get_neuron(edge.src).cell_type
                    if ct == "excitatory":
                        edge.weight.clamp_(min=0.0)
                    elif ct == "inhibitory":
                        edge.weight.clamp_(max=0.0)
                return
            vec = self._ff_weight_vec
            if vec is not None and vec.numel():
                zero = torch.zeros_like(vec)
                inf = torch.full_like(vec, float("inf"))
                lo = torch.where(self._ff_sign > 0, zero, -inf)
                hi = torch.where(self._ff_sign < 0, zero, inf)
                vec.copy_(torch.minimum(torch.maximum(vec, lo), hi))
            for edge in self._rec_edges:
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

    def _apply_dale(
        self, raw: torch.Tensor, cell_types: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """Apply the softplus Dale transform to a vector of raw edge parameters.

        cell_types=None means the packed feedforward edges (uses the cached
        sign vector); pass a list to transform some other edge set."""
        if self._dale_mode != "softplus":
            return raw
        sign = self._ff_sign if cell_types is None else self._sign_tensor(cell_types)
        return torch.where(sign == 0, raw, sign * F.softplus(raw))

    def _rec_effective_weights(self) -> torch.Tensor:
        """Effective weights of all delayed edges as one differentiable [E_rec] tensor."""
        raw = torch.stack([e.weight.reshape(()) for e in self._rec_edges])
        if self._dale_mode != "softplus":
            return raw
        return torch.where(self._rec_sign == 0, raw, self._rec_sign * F.softplus(raw))

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
            # One stack, no per-edge .item(): a host sync per edge here would
            # cost a GPU round trip for every edge on every topology change.
            with torch.no_grad():
                packed = torch.stack(
                    [self._edges[eid].weight.detach().reshape(()) for eid in ff_edge_ids]
                ).to(device=self._device, dtype=torch.float32)
            self._ff_weight_vec = nn.Parameter(packed)
            # Remove individual ff Parameters from ParameterDict — they are now
            # represented by the packed vector and must not appear twice in state_dict.
            for eid in ff_edge_ids:
                param_key = eid.replace("-", "_")
                if param_key in self._edge_weights:
                    del self._edge_weights[param_key]
            # Each edge.weight becomes an element-view of the packed vector so
            # that enforce_dale(), effective_weight() and in-place edits work.
            # Gradients are NOT read from these views: they sit outside the
            # forward graph, so their .grad is never populated. Read them from
            # the packed vector via edge_table() instead.
            with torch.enable_grad():   # views must stay attached even if rebuilt under no_grad
                for i, eid in enumerate(ff_edge_ids):
                    self._edges[eid].weight = self._ff_weight_vec[i]
        else:
            self._ff_dst = None
            self._ff_src = None
            self._ff_weight_vec = nn.Parameter(torch.empty(0, device=self._device))

        self._ff_params = []          # no longer used; cleared to free references
        self._ff_cell_types = ff_cell_types
        self._ff_edge_ids = ff_edge_ids   # new list object each rebuild
        self._ff_sign = self._sign_tensor(ff_cell_types)
        self._rec_edges = rec_edges
        rec_cells = [self.get_neuron(e.src).cell_type for e in rec_edges]
        self._rec_sign = self._sign_tensor(rec_cells)
        self._rec_src = torch.tensor(
            [self._nid_to_idx[e.src] for e in rec_edges], dtype=torch.long, device=self._device)
        self._rec_dst = torch.tensor(
            [self._nid_to_idx[e.dst] for e in rec_edges], dtype=torch.long, device=self._device)
        self._rec_delay = torch.tensor(
            [e.delay - 1 for e in rec_edges], dtype=torch.long, device=self._device)
        self._max_rec_delay = max((e.delay for e in rec_edges), default=0)
        # A delayed edge reads its SOURCE's history, so the source is the
        # neuron that has to push into it every step.
        readers = {e.src for e in rec_edges}
        for nid, neuron in self._neurons.items():
            neuron._has_delayed_readers = nid in readers  # type: ignore
        self._topo_order = self._topological_order()
        self._is_uniform = self._check_uniform()
        self._level_act_groups = self._compute_level_act_groups() if self._is_uniform else []
        self._plan = self._build_plan()
        self._input_idx = torch.tensor(
            [self._nid_to_idx[nid] for nid in self._input_ids],
            dtype=torch.long, device=self._device)
        self._output_idx = torch.tensor(
            [self._nid_to_idx[nid] for nid in self._output_ids],
            dtype=torch.long, device=self._device)
        self._fwd = self._fast_forward if self._is_uniform else self._raw_forward
        self._matrix_dirty = False
        self._struct_version += 1

    def _sign_tensor(self, cell_types: List[str]) -> torch.Tensor:
        """+1 excitatory, -1 inhibitory, 0 unconstrained, one entry per edge."""
        table = {"excitatory": 1.0, "inhibitory": -1.0}
        return torch.tensor(
            [table.get(ct, 0.0) for ct in cell_types],
            dtype=torch.float32, device=self._device,
        )

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
        weights = self._apply_dale(raw)
        flat_idx = self._ff_dst * n + self._ff_src       # [num_ff]
        W_flat = torch.zeros(n * n, device=self._device)
        W_flat = W_flat.scatter_add(0, flat_idx, weights)
        return W_flat.view(n, n)

    # ------------------------------------------------------------------
    # Whole-graph edge access (no per-edge host syncs)
    # ------------------------------------------------------------------

    def edge_table(self) -> "EdgeTable":
        """
        Every edge as aligned tensors: packed feedforward edges first (in
        _ff_edge_ids order), then delayed edges. Weights are effective
        (post-Dale) and detached; grad is the gradient of the stored
        parameter, zero where no backward has run yet.

        This is how to read edge gradients. edge.weight.grad is always empty
        for packed edges because edge.weight is a view that sits outside the
        forward graph; the gradient lives on the packed vector.
        """
        self.ensure_built()
        dev = self._device
        ids = list(self._ff_edge_ids)
        src, dst, w, g = [], [], [], []
        with torch.no_grad():
            vec = self._ff_weight_vec
            if vec is not None and vec.numel():
                src.append(self._ff_src)
                dst.append(self._ff_dst)
                w.append(self._apply_dale(vec.detach()))
                g.append(vec.grad.detach() if vec.grad is not None else torch.zeros_like(vec))
            if self._rec_edges:
                ids.extend(e.edge_id for e in self._rec_edges)
                src.append(self._rec_src)
                dst.append(self._rec_dst)
                w.append(self._rec_effective_weights().detach())
                g.append(torch.stack([
                    e.weight.grad.reshape(()) if e.weight.grad is not None
                    else torch.zeros((), device=dev)
                    for e in self._rec_edges
                ]))
        if not ids:
            empty_l = torch.zeros(0, dtype=torch.long, device=dev)
            empty_f = torch.zeros(0, device=dev)
            return EdgeTable([], empty_l, empty_l, empty_f, empty_f)
        return EdgeTable(
            edge_ids=ids,
            src=torch.cat(src), dst=torch.cat(dst),
            weight=torch.cat(w), grad=torch.cat(g),
        )

    # ------------------------------------------------------------------
    # Batched forward with explicit recurrent state
    # ------------------------------------------------------------------

    def initial_state(self, batch_size: int = 1) -> GraphState:
        """All-zero GraphState for `batch_size` independent sequences."""
        self.ensure_built()
        n = len(self._neuron_index)
        z = torch.zeros(batch_size, n, device=self._device)
        hist = (
            torch.zeros(batch_size, self._max_rec_delay, n, device=self._device)
            if self._max_rec_delay > 0 else None
        )
        return GraphState(h=z, c=z.clone(), hist=hist)

    def forward_batch(
        self, obs: torch.Tensor, state: Optional[GraphState] = None,
    ) -> Tuple[torch.Tensor, GraphState]:
        """
        One timestep for a whole batch, with the recurrent state passed in and
        returned instead of living inside the neurons.

            obs   [B, n_inputs]
            state GraphState from the previous step (None = all zeros)
            ->    (out [B, n_outputs], next GraphState)

        Computes the same function as running forward() on each row from a
        reset graph, but as a few tensor ops per topological level instead of
        a Python loop over neurons, and it handles point, GRU and LSTM neurons
        and delayed edges. Neuron objects are only read for their parameters:
        _current, _h, _c and the history buffers are not touched, so any number
        of independent batches can run through one graph.

        Gradients flow through the returned state, so a sequence unrolled with
        this method is trained by backprop through time; call state.detach()
        every k steps to truncate. The one exception is plain GRUNeuron and
        LSTMNeuron, whose own hidden state is detached each step exactly as
        their step() does; TrainableGRUNeuron / TrainableLSTMNeuron are not.
        """
        if self._matrix_dirty:
            self._rebuild_matrix_structure()
        n_in = len(self._input_ids)
        if obs.dim() != 2 or obs.shape[1] != n_in:
            raise ValueError(
                f"forward_batch expects obs of shape [B, {n_in}], got {tuple(obs.shape)}"
            )
        obs = obs.to(device=self._device, dtype=torch.float32)
        B = obs.shape[0]
        n = len(self._neuron_index)
        if state is None:
            state = self.initial_state(B)
        elif state.h.shape != (B, n):
            raise ValueError(
                f"state.h has shape {tuple(state.h.shape)} but this graph needs {(B, n)}; "
                f"the topology changed since the state was created, or the batch size differs."
            )
        for step in self._plan:
            for key, _, _ in step:
                if key[0] == "custom":
                    raise NotImplementedError(
                        f"forward_batch does not support neuron type {key[1]}; "
                        f"supported: Neuron, GRUNeuron, LSTMNeuron and their Trainable variants."
                    )

        D = self._max_rec_delay
        hist = state.hist
        rec_pre = None
        if D > 0:
            if hist is None:
                hist = torch.zeros(B, D, n, device=self._device)
            elif hist.shape[1] < D:
                pad = torch.zeros(B, D - hist.shape[1], n, device=self._device)
                hist = torch.cat([hist, pad], dim=1)
            past = hist[:, self._rec_delay, self._rec_src]            # [B, E_rec]
            contrib = past * self._rec_effective_weights()
            rec_pre = torch.zeros(B, n, device=self._device).index_add(
                1, self._rec_dst, contrib)

        W = self._assemble_W()
        h = state.h.index_copy(1, self._input_idx, obs)
        c = state.c
        sig, tanh = torch.sigmoid, torch.tanh

        for step in self._plan:
            for key, idx, neurons in step:
                kind = key[0]
                pre = h @ W[idx].t()                                  # [B, k]
                if rec_pre is not None:
                    pre = pre + rec_pre[:, idx]
                if kind == "point":
                    bias = torch.cat([nr.bias for nr in neurons])
                    vals = ACTIVATIONS[key[1]](pre + bias)
                elif kind == "gru":
                    hp = state.h[:, idx]
                    if key[1]:
                        hp = hp.detach()
                    p = {a: torch.cat([getattr(nr, a) for nr in neurons]) for a in _GRU_ATTRS}
                    z = sig(p["wz"] * hp + p["uz"] * pre + p["bz"])
                    r = sig(p["wr"] * hp + p["ur"] * pre + p["br"])
                    cand = tanh(p["wn"] * r * hp + p["un"] * pre + p["bn"])
                    vals = (1 - z) * hp + z * cand
                else:  # lstm
                    hp = state.h[:, idx]
                    cp = state.c[:, idx]
                    if key[1]:
                        hp, cp = hp.detach(), cp.detach()
                    p = {a: torch.cat([getattr(nr, a) for nr in neurons]) for a in _LSTM_ATTRS}
                    f = sig(p["wf"] * hp + p["uf"] * pre + p["bf"])
                    i = sig(p["wi"] * hp + p["ui"] * pre + p["bi"])
                    g = tanh(p["wg"] * hp + p["ug"] * pre + p["bg"])
                    o = sig(p["wo"] * hp + p["uo"] * pre + p["bo"])
                    c_new = f * cp + i * g
                    vals = o * tanh(c_new)
                    c = c.index_copy(1, idx, c_new)
                h = h.index_copy(1, idx, vals)

        new_hist = None
        if D > 0:
            new_hist = torch.cat([h.unsqueeze(1), hist[:, :D - 1]], dim=1)
        self.record_activations(h)
        return h.index_select(1, self._output_idx), GraphState(h=h, c=c, hist=new_hist)

    def _build_plan(self) -> list:
        """Topological levels, each split into groups that can be computed
        together: point neurons by activation, GRUs, LSTMs."""
        non_input = set(self._hidden_ids) | set(self._output_ids)
        input_set = set(self._input_ids)
        in_degree: Dict[str, int] = {nid: 0 for nid in non_input}
        succ: Dict[str, List[str]] = {}
        for edge in self._edges.values():
            if edge.delay == 0 and edge.dst in in_degree and edge.src not in input_set:
                in_degree[edge.dst] += 1
                succ.setdefault(edge.src, []).append(edge.dst)

        level = [nid for nid in self._neuron_index if nid in in_degree and in_degree[nid] == 0]
        levels: List[List[str]] = []
        seen: set = set()
        while level:
            levels.append(level)
            seen.update(level)
            nxt: List[str] = []
            for nid in level:
                for dst in succ.get(nid, ()):
                    in_degree[dst] -= 1
                    if in_degree[dst] == 0:
                        nxt.append(dst)
            level = nxt
        # Neurons on a delay-0 cycle never reach in-degree 0; run them last,
        # one at a time, as _raw_forward does.
        levels.extend([nid] for nid in self._neuron_index if nid in in_degree and nid not in seen)

        plan = []
        for level in levels:
            groups: Dict[tuple, List[str]] = {}
            for nid in level:
                t = type(self._neurons[nid])
                if t is Neuron:
                    key = ("point", self._neurons[nid].activation_name)  # type: ignore
                elif t in (GRUNeuron, TrainableGRUNeuron):
                    key = ("gru", t is GRUNeuron)
                elif t in (LSTMNeuron, TrainableLSTMNeuron):
                    key = ("lstm", t is LSTMNeuron)
                else:
                    key = ("custom", t.__name__)
                groups.setdefault(key, []).append(nid)
            plan.append([
                (
                    key,
                    torch.tensor([self._nid_to_idx[nid] for nid in nids],
                                 dtype=torch.long, device=self._device),
                    [self._neurons[nid] for nid in nids],
                )
                for key, nids in groups.items()
            ])
        return plan

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

        self.record_activations(h)

        for nid in self._neurons:
            if self._neurons[nid]._has_delayed_readers:  # type: ignore
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
