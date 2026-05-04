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


class HebbianHook:
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

        # accum[graph_idx][edge_id] = list of (h_pre * h_post) scalar values
        self._accum: List[Dict[str, List[float]]] = [
            defaultdict(list) for _ in self.graphs
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
        neurons still hold their _current activations for this timestep.
        """
        for g_idx, graph in enumerate(self.graphs):
            accum = self._accum[g_idx]
            for edge in graph.all_edges():
                src_neuron = graph.get_neuron(edge.src)
                dst_neuron = graph.get_neuron(edge.dst)
                h_pre = src_neuron._current.detach().squeeze().item()
                h_post = dst_neuron._current.detach().squeeze().item()
                accum[edge.edge_id].append(h_pre * h_post)
        self._steps += 1

    def apply(self) -> None:
        """
        Apply the Hebbian update using accumulated co-activation statistics.

        Δw = η · mean(h_pre · h_post) - λ · w

        Call once per episode, after agent.learn() so PPO and Hebbian updates
        are both applied before the next episode starts.
        """
        if self._steps == 0:
            return

        for g_idx, graph in enumerate(self.graphs):
            accum = self._accum[g_idx]
            with torch.no_grad():
                for edge in graph.all_edges():
                    samples = accum.get(edge.edge_id)
                    if not samples:
                        continue
                    mean_coact = sum(samples) / len(samples)
                    delta = self.hebbian_lr * mean_coact - self.weight_decay * edge.weight.item()
                    edge.weight.add_(delta)

                    if self.clip_weight is not None:
                        edge.weight.clamp_(-self.clip_weight, self.clip_weight)

            if self.respect_dale:
                graph.enforce_dale()

    def reset(self) -> None:
        """Clear all accumulated co-activation data. Call at episode start or end."""
        for accum in self._accum:
            accum.clear()
        self._steps = 0

    # ------------------------------------------------------------------
    # Convenience: apply + reset in one call
    # ------------------------------------------------------------------

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
        for accum in self._accum:
            for eid, samples in accum.items():
                if samples:
                    result[eid] = sum(samples) / len(samples)
        return result
