from __future__ import annotations

"""
TopologyAwareAdam — Adam optimizer that resets momentum state for parameters
affected by topology changes (grow / prune / merge).

Without state reset, Adam's stale (m, v) estimates from before the structural
change distort the first several updates on the modified parameter, causing
transient instability. Calling notify_topology_change() after any grow/prune
operation discards only the affected parameters' state so the rest of the
network continues uninterrupted.

Usage::

    opt = TopologyAwareAdam(graph.parameters(), lr=3e-4)

    # After growing a new edge or neuron:
    new_params = [graph.get_edge(eid).weight for eid in new_edge_ids]
    opt.notify_topology_change(new_params)      # clean start for new params

    # After pruning / merging (surviving params that absorbed weight):
    opt.notify_topology_change(modified_params) # reset stale momentum
"""

from typing import Iterable, List

import torch
import torch.nn as nn


class TopologyAwareAdam:
    """
    Thin wrapper around torch.optim.Adam with topology-aware state management.

    All standard Adam methods (step, zero_grad, state_dict, load_state_dict,
    add_param_group) delegate to the inner optimizer so this is a drop-in
    replacement.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        amsgrad: bool = False,
    ) -> None:
        self._opt = torch.optim.Adam(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
        )

    # ------------------------------------------------------------------
    # Topology lifecycle
    # ------------------------------------------------------------------

    def notify_topology_change(self, params: Iterable[nn.Parameter]) -> None:
        """
        Reset Adam momentum state for the given parameters.

        Call after any grow, prune, or merge operation — pass the nn.Parameter
        objects whose weights have structurally changed (new edges/neurons, or
        surviving params that absorbed the weight of a pruned/merged neighbour).

        Parameters not in the optimizer's tracked groups are silently ignored.
        """
        for p in params:
            if p in self._opt.state:
                del self._opt.state[p]

    # ------------------------------------------------------------------
    # Standard optimizer delegation
    # ------------------------------------------------------------------

    def step(self, closure=None):
        return self._opt.step(closure)

    def zero_grad(self, set_to_none: bool = True) -> None:
        self._opt.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return self._opt.state_dict()

    def load_state_dict(self, state_dict) -> None:
        self._opt.load_state_dict(state_dict)

    def add_param_group(self, param_group) -> None:
        self._opt.add_param_group(param_group)

    @property
    def param_groups(self):
        return self._opt.param_groups

    @property
    def state(self):
        return self._opt.state

    def __repr__(self) -> str:
        return f"TopologyAwareAdam({self._opt})"
