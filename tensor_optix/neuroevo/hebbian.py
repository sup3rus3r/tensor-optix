from __future__ import annotations

"""
HebbianHook — local Hebbian learning running alongside PPO gradient updates.

Rule (Oja-style with decay to prevent unbounded growth):
    Δw(u→v) = η · mean_t(h_u^t · h_v^t) - λ · w(u→v)

    η  : hebbian_lr   — how fast co-activation strengthens connections
    λ  : weight_decay — prevents weights from growing without bound
    t  : timestep index within the current episode

Usage with GraphAgent::

    from tensor_optix.neuroevo.hebbian import HebbianHook

    hook = HebbianHook(graph, hebbian_lr=1e-3, weight_decay=1e-4)

    # In your training loop:
    for step in episode:
        action = agent.act(obs)          # forward pass populates _current on neurons
        hook.record()                    # call immediately after each act()
        obs, reward, done, _ = env.step(action)

    agent.learn(episode_data)            # PPO update
    hook.apply()                         # Hebbian update (after PPO)
    hook.reset()                         # clear accumulators for next episode

Usage with BrainNetwork::

    hook = HebbianHook.from_brain(brain, hebbian_lr=1e-3, weight_decay=1e-4)
    # same record() / apply() / reset() interface
"""

from collections import defaultdict
from typing import Dict, List, Optional, Union

import torch

from .graph.neuron_graph import NeuronGraph
from .brain_network import BrainNetwork
from tensor_optix.core.loop_controller import LoopCallback


class HebbianHook(LoopCallback):
    """
    Accumulates co-activation statistics across a full episode, then applies
    a local Hebbian weight update to every edge in one or more NeuronGraphs.

    Parameters
    ----------
    graphs : NeuronGraph | list[NeuronGraph]
        The graph(s) to apply Hebbian updates to. Pass a list to cover all
        regions of a BrainNetwork, or use HebbianHook.from_brain().
    hebbian_lr : float
        Hebbian learning rate η. Typical range: 1e-4 – 1e-2.
    weight_decay : float
        Decay coefficient λ. Prevents unbounded growth. Typical: 1e-4 – 1e-3.
    clip_weight : float | None
        If set, clamps all weights to [-clip_weight, +clip_weight] after update.
    respect_dale : bool
        If True, calls graph.enforce_dale() after each Hebbian update so Dale's
        Law constraints are maintained.
    """

    def __init__(
        self,
        graphs: Union[NeuronGraph, List[NeuronGraph]],
        hebbian_lr: float = 1e-3,
        weight_decay: float = 1e-4,
        clip_weight: Optional[float] = None,
        respect_dale: bool = True,
    ) -> None:
        self.graphs: List[NeuronGraph] = (
            [graphs] if isinstance(graphs, NeuronGraph) else list(graphs)
        )
        self.hebbian_lr = hebbian_lr
        self.weight_decay = weight_decay
        self.clip_weight = clip_weight
        self.respect_dale = respect_dale

        # Per graph: running sum of h_pre * h_post per edge, kept on the device
        # as one vector aligned with the graph's edge_table() order, plus the
        # number of steps summed. Moved to the host (one transfer) only when
        # the topology changes or the update is applied, into _accum:
        # edge_id -> [sum, count].
        self._run_sum: List[Optional[torch.Tensor]] = [None for _ in self.graphs]
        self._run_count: List[int] = [0 for _ in self.graphs]
        self._run_ids: List[List[str]] = [[] for _ in self.graphs]
        self._run_version: List[int] = [-1 for _ in self.graphs]
        self._accum: List[Dict[str, List[float]]] = [
            defaultdict(lambda: [0.0, 0]) for _ in self.graphs
        ]
        self._steps: int = 0

    @classmethod
    def from_brain(
        cls,
        brain: BrainNetwork,
        hebbian_lr: float = 1e-3,
        weight_decay: float = 1e-4,
        clip_weight: Optional[float] = None,
        respect_dale: bool = True,
    ) -> "HebbianHook":
        """Create a HebbianHook covering all regions of a BrainNetwork."""
        graphs = [brain.get_region(name) for name in brain.region_names]
        return cls(
            graphs=graphs,
            hebbian_lr=hebbian_lr,
            weight_decay=weight_decay,
            clip_weight=clip_weight,
            respect_dale=respect_dale,
        )

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def record(self) -> None:
        """
        Snapshot the current co-activation product (h_pre · h_post) for every
        edge across all tracked graphs.

        Call this immediately after each forward pass / agent.act() while
        neurons still hold their activations for this timestep. All edges are
        handled as one tensor op, with no host sync.
        """
        for g_idx, graph in enumerate(self.graphs):
            table = graph.edge_table()
            if graph._struct_version != self._run_version[g_idx]:
                self._flush(g_idx)
                self._run_ids[g_idx] = table.edge_ids
                self._run_version[g_idx] = graph._struct_version
            # Each neuron's current activation, gathered without a host sync.
            h = torch.cat([
                graph.get_neuron(nid)._current.detach().reshape(-1)[:1]
                for nid in graph.neuron_index
            ]).to(table.weight.device)
            coact = h.index_select(0, table.src) * h.index_select(0, table.dst)
            run = self._run_sum[g_idx]
            self._run_sum[g_idx] = coact if run is None else run + coact
            self._run_count[g_idx] += 1
        self._steps += 1

    def _flush(self, g_idx: int) -> None:
        """Move the on-device running sum for one graph into the per-edge dict."""
        run = self._run_sum[g_idx]
        if run is not None and self._run_count[g_idx]:
            accum = self._accum[g_idx]
            count = self._run_count[g_idx]
            for eid, s in zip(self._run_ids[g_idx], run.tolist()):
                entry = accum[eid]
                entry[0] += s
                entry[1] += count
        self._run_sum[g_idx] = None
        self._run_count[g_idx] = 0

    def apply(self) -> None:
        """
        Apply the Hebbian update using accumulated co-activation statistics.

        Δw = η · mean(h_pre · h_post) - λ · w

        Call once per episode, after agent.learn() so PPO and Hebbian updates
        are both applied before the next episode starts. Computed for all edges
        at once on the device.
        """
        if self._steps == 0:
            return

        for g_idx, graph in enumerate(self.graphs):
            self._flush(g_idx)
            accum = self._accum[g_idx]
            table = graph.edge_table()
            if not table.edge_ids:
                continue
            dev = table.weight.device
            means, has = [], []
            for eid in table.edge_ids:
                entry = accum.get(eid)
                if entry and entry[1]:
                    means.append(entry[0] / entry[1])
                    has.append(True)
                else:
                    means.append(0.0)
                    has.append(False)
            mean_coact = torch.tensor(means, device=dev)
            mask = torch.tensor(has, device=dev)

            # The rule is defined on the EFFECTIVE weight. In dale_mode=
            # "softplus" the stored parameter is a raw theta (effective =
            # +/-softplus(theta)); decaying or clipping theta directly would,
            # e.g., GROW a near-zero excitatory edge (theta << 0) instead of
            # shrinking it.
            w = table.weight
            new_w = w + self.hebbian_lr * mean_coact - self.weight_decay * w
            if self.clip_weight is not None:
                new_w = new_w.clamp(-self.clip_weight, self.clip_weight)
            new_raw = self._to_raw(graph, new_w)

            n_ff = len(graph._ff_edge_ids)
            with torch.no_grad():
                vec = graph._ff_weight_vec
                if n_ff:
                    vec.copy_(torch.where(mask[:n_ff], new_raw[:n_ff], vec))
                for i, edge in enumerate(graph._rec_edges):
                    j = n_ff + i
                    if has[j]:
                        edge.weight.copy_(new_raw[j].reshape(edge.weight.shape))

            if self.respect_dale:
                graph.enforce_dale()

    @staticmethod
    def _to_raw(graph: NeuronGraph, new_w: torch.Tensor) -> torch.Tensor:
        """Effective weights -> stored parameters, for every edge in edge_table() order."""
        if graph._dale_mode != "softplus":
            return new_w
        sign = torch.cat([graph._ff_sign, graph._rec_sign])
        # Dale: an excitatory edge cannot go negative, an inhibitory one positive.
        clamped = torch.where(sign > 0, new_w.clamp(min=0.0),
                              torch.where(sign < 0, new_w.clamp(max=0.0), new_w))
        mag = clamped.abs()
        # Inverse of softplus; near-zero magnitudes map to theta = -10 as in
        # NeuronGraph._softplus_init.
        theta = torch.where(
            mag < 1e-6, torch.full_like(mag, -10.0),
            torch.log(torch.expm1(mag.clamp(min=1e-6))),
        )
        return torch.where(sign == 0, new_w, theta)

    def reset(self) -> None:
        """Clear all accumulated co-activation data. Call at episode start or end."""
        for g_idx, accum in enumerate(self._accum):
            accum.clear()
            self._run_sum[g_idx] = None
            self._run_count[g_idx] = 0
            self._run_version[g_idx] = -1
        self._steps = 0

    # ------------------------------------------------------------------
    # Convenience: apply + reset in one call
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # LoopCallback wiring — makes callbacks=[hook] fully automatic
    # ------------------------------------------------------------------

    def on_loop_start(self) -> None:
        """Record co-activations after every graph forward for the run."""
        for graph in self.graphs:
            if self.record_graph_step not in graph._post_forward_hooks:
                graph._post_forward_hooks.append(self.record_graph_step)

    def on_loop_stop(self) -> None:
        for graph in self.graphs:
            if self.record_graph_step in graph._post_forward_hooks:
                graph._post_forward_hooks.remove(self.record_graph_step)

    def on_episode_end(self, episode_id, eval_metrics) -> None:
        """Runs after agent.learn() for the window: apply the Hebbian update
        on top of the gradient step, then clear the accumulators."""
        self.apply_and_reset()

    def record_graph_step(self) -> None:
        """Post-forward hook target. With several graphs each forward fires
        once per graph; count one step per full sweep."""
        self._hook_calls = getattr(self, "_hook_calls", 0) + 1
        if self._hook_calls >= len(self.graphs):
            self._hook_calls = 0
            self.record()

    def apply_and_reset(self) -> None:
        """Apply the Hebbian update then immediately clear accumulators."""
        self.apply()
        self.reset()

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def n_steps_recorded(self) -> int:
        """Number of timesteps recorded since last reset."""
        return self._steps

    def mean_coactivation(self) -> Dict[str, float]:
        """
        Return a dict of edge_id -> mean co-activation value across all graphs.
        Useful for logging how strongly pairs of neurons are correlating.
        """
        result: Dict[str, float] = {}
        for g_idx, accum in enumerate(self._accum):
            self._flush(g_idx)
            for eid, (total, count) in accum.items():
                if count:
                    result[eid] = total / count
        return result
