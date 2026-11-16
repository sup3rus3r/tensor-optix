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

from typing import Iterable, List, Optional, Sequence

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

    def carry_over(
        self,
        old: "TopologyAwareAdam",
        old_packed: Optional[nn.Parameter] = None,
        old_ids: Optional[Sequence[str]] = None,
        new_packed: Optional[nn.Parameter] = None,
        new_ids: Optional[Sequence[str]] = None,
    ) -> None:
        """
        Copy Adam state from `old` after a topology rebuild replaced some
        parameters with new objects.

        * A parameter that is the same object in both optimizers keeps its
          state untouched.
        * The packed edge-weight vector is a new Parameter after every
          rebuild. Pass the old and new vector together with the edge id of
          each position (NeuronGraph._ff_edge_ids at the time of each
          rebuild): positions are matched by edge id, so a surviving edge keeps
          its momentum wherever it moved, and a new edge starts from zero.

        Adam keeps one step counter per Parameter, so a new edge in a long-
        running vector gets a mature step count with empty moments; its first
        updates are therefore a few times larger than lr. The transient is
        short, and notify_topology_change() cannot reset single entries of a
        packed vector since it works per Parameter.
        """
        packed_state = None
        for group in self._opt.param_groups:
            for p in group["params"]:
                if p is new_packed:
                    continue
                st = old._opt.state.get(p)
                if st:
                    self._opt.state[p] = st
        if (
            new_packed is None or old_packed is None
            or old_ids is None or new_ids is None
        ):
            return
        packed_state = old._opt.state.get(old_packed)
        if not packed_state:
            return
        old_index = {eid: i for i, eid in enumerate(old_ids)}
        src, dst = [], []
        for j, eid in enumerate(new_ids):
            i = old_index.get(eid)
            if i is not None:
                src.append(i)
                dst.append(j)
        n_old, n_new = len(old_ids), len(new_ids)
        new_state = {}
        for key, val in packed_state.items():
            if torch.is_tensor(val) and val.dim() == 1 and val.numel() == n_old:
                nv = torch.zeros(n_new, dtype=val.dtype, device=val.device)
                if src:
                    nv[torch.tensor(dst, device=val.device)] = val[torch.tensor(src, device=val.device)]
                new_state[key] = nv
            else:
                new_state[key] = val.clone() if torch.is_tensor(val) else val
        self._opt.state[new_packed] = new_state

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
