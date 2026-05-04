from __future__ import annotations
import uuid
from collections import deque
from typing import Callable, Deque, Optional

import torch
import torch.nn as nn


def _linear(x: torch.Tensor) -> torch.Tensor: return x

ACTIVATIONS: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "linear": _linear,
    "relu": torch.relu,
    "tanh": torch.tanh,
    "sigmoid": torch.sigmoid,
    "elu": torch.nn.functional.elu,
}

CELL_TYPES = {"excitatory", "inhibitory", "any"}


class Neuron(nn.Module):
    """
    A single neuron in the free-form graph.

    Owns:
    - a scalar bias (learnable)
    - an activation function (by name, not learnable)
    - a fixed-depth circular history buffer for variable-delay recurrence
    - a cell_type enforcing Dale's Law: excitatory neurons may only send
      positive-weight signals; inhibitory neurons may only send negative-weight
      signals. "any" (default) is unconstrained and backward-compatible.

    History buffer stores past activations so edges with delay d can read
    h_v^(t-d) without any special handling in the graph forward pass.
    """

    def __init__(
        self,
        activation: str = "tanh",
        neuron_id: Optional[str] = None,
        max_delay: int = 1,
        cell_type: str = "any",
    ) -> None:
        super().__init__()
        if activation not in ACTIVATIONS:
            raise ValueError(f"Unknown activation '{activation}'. Choose from {list(ACTIVATIONS)}")
        if cell_type not in CELL_TYPES:
            raise ValueError(f"Unknown cell_type '{cell_type}'. Choose from {CELL_TYPES}")

        self.neuron_id: str = neuron_id or str(uuid.uuid4())
        self.activation_name: str = activation
        self._activation_fn: Callable = ACTIVATIONS[activation]
        self.cell_type: str = cell_type

        self.bias = nn.Parameter(torch.zeros(1))

        self._max_delay: int = max(1, max_delay)
        # deque[0] = h^(t-1), deque[-1] = h^(t-max_delay)
        # We prepend on each step so index i = h^(t-i-1) ... managed via get_delayed()
        self._history: Deque[torch.Tensor] = deque(
            [torch.zeros(1)] * self._max_delay,
            maxlen=self._max_delay,
        )
        # Current activation (h^t), set during forward, before pushing to history
        self._current: torch.Tensor = torch.zeros(1)

    # ------------------------------------------------------------------
    # History management
    # ------------------------------------------------------------------

    def get_delayed(self, delay: int) -> torch.Tensor:
        """
        Return h^(t-delay).

        delay=0  → current activation (h^t, set during this timestep's forward)
        delay=1  → previous timestep
        delay=d  → d timesteps ago
        """
        if delay == 0:
            return self._current
        idx = delay - 1  # history[0] = h^(t-1)
        if idx >= len(self._history):
            return torch.zeros_like(self._current)
        return self._history[idx]

    def push_history(self) -> None:
        """Commit current activation to history. Called after full graph forward."""
        self._history.appendleft(self._current.detach().clone())

    def expand_history(self, new_max_delay: int) -> None:
        """Grow the history buffer when a longer-delay edge is added."""
        if new_max_delay <= self._max_delay:
            return
        pad = new_max_delay - self._max_delay
        # Pad the tail with zeros (oldest slots)
        for _ in range(pad):
            self._history.append(torch.zeros_like(self._current))
        self._history = deque(self._history, maxlen=new_max_delay)
        self._max_delay = new_max_delay

    def init_history_from_buffer(
        self, history_tensors: list[torch.Tensor]
    ) -> None:
        """
        Seed history from an external list (newest first).
        Used when inserting a new neuron mid-run to avoid cold-start discontinuity.
        """
        self._history = deque(
            (h.detach().clone() for h in history_tensors[: self._max_delay]),
            maxlen=self._max_delay,
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, pre_activation: torch.Tensor) -> torch.Tensor:
        """
        Apply bias + activation to the aggregated pre-activation sum.

        pre_activation: scalar tensor — sum of weighted inputs computed by the graph.
        Returns h^t and stores it in self._current.
        """
        h = self._activation_fn(pre_activation + self.bias)
        self._current = h
        return h

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @property
    def max_delay(self) -> int:
        return self._max_delay

    def reset_state(self) -> None:
        """Zero out all history — call between episodes."""
        zero = torch.zeros(1, device=self.bias.device)
        self._history = deque([zero] * self._max_delay, maxlen=self._max_delay)
        self._current = zero

    def extra_repr(self) -> str:
        return f"id={self.neuron_id[:8]}, act={self.activation_name}, max_delay={self._max_delay}, cell_type={self.cell_type}"
