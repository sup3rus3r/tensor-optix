from __future__ import annotations

import copy
import uuid
from collections import deque
from typing import TYPE_CHECKING, Callable, Deque, List, Optional

import torch
import torch.nn as nn

if TYPE_CHECKING:
    pass


def _linear(x: torch.Tensor) -> torch.Tensor: return x


ACTIVATIONS: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "linear": _linear,
    "relu":    torch.relu,
    "tanh":    torch.tanh,
    "sigmoid": torch.sigmoid,
    "elu":     torch.nn.functional.elu,
}

CELL_TYPES = {"excitatory", "inhibitory", "any"}


# ──────────────────────────────────────────────────────────────────────────────
# Base / Point Neuron
# ──────────────────────────────────────────────────────────────────────────────

class Neuron(nn.Module):
    """
    A single point neuron: bias + activation + delay history buffer.

    Protocol methods (shared interface for all neuron types):
      step(weighted_sum)           — full forward: bias + activation
      importance(incident_w_sum)  — I(v) = Σ|w_e| * (‖h‖₁/d + ε)
      can_merge_with(other)        — same type + same dim required
      make_relay()                 — returns a linear point neuron (exact identity)
      split_copy()                 — deep-copy of self; caller halves outgoing edges

    cell_type enforces Dale's Law:
      'excitatory' → all outgoing weights clamped ≥ 0
      'inhibitory' → all outgoing weights clamped ≤ 0
      'any'        → unconstrained (default, backward-compatible)
    """

    # Subclasses set this to their hidden-state dimension for importance normalisation.
    # Point neurons are always scalar → dim = 1.
    _hidden_dim: int = 1

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
        self._history: Deque[torch.Tensor] = deque(
            [torch.zeros(1)] * self._max_delay,
            maxlen=self._max_delay,
        )
        self._current: torch.Tensor = torch.zeros(1)

    # ------------------------------------------------------------------
    # Protocol — step
    # ------------------------------------------------------------------

    def step(self, weighted_sum: torch.Tensor) -> torch.Tensor:
        """
        Full forward: bias addition + activation.
        weighted_sum is the pre-activation from all incoming edges (W @ h_prev)[idx].
        Sets and returns self._current.
        """
        h = self._activation_fn(weighted_sum + self.bias)
        self._current = h
        return h

    # ------------------------------------------------------------------
    # Protocol — importance
    # ------------------------------------------------------------------

    def importance(self, incident_weight_sum: float) -> float:
        """
        I(v) = Σ|w_e| * (‖h‖₁/d + ε)

        incident_weight_sum: Σ|w_e| for all incident edges, pre-computed by
        the controller in one shared edge pass.
        """
        h_mag = float(self._current.abs().mean().item()) / self._hidden_dim
        return incident_weight_sum * (h_mag + 1e-8)

    # ------------------------------------------------------------------
    # Protocol — can_merge_with
    # ------------------------------------------------------------------

    def can_merge_with(self, other: "Neuron") -> bool:
        """
        Same concrete type required. Caller additionally checks activation
        history correlation before merging.
        """
        return type(self) is type(other)

    # ------------------------------------------------------------------
    # Protocol — make_relay
    # ------------------------------------------------------------------

    def make_relay(self) -> "Neuron":
        """
        Return a new neuron suitable for insertion on an edge incident to self.

        Always returns a linear point neuron regardless of self's type — this
        is the only choice that gives exact function preservation (verified by
        test_neuron_protocol_math.py: GRU/LSTM relays have 24–36% error in the
        tanh activation range). Gradient descent adapts the relay afterward.
        """
        return Neuron(
            activation="linear",
            max_delay=self._max_delay,
            cell_type=self.cell_type,
        )

    # ------------------------------------------------------------------
    # Protocol — split_copy
    # ------------------------------------------------------------------

    def split_copy(self) -> "Neuron":
        """
        Return a deep copy of self with a new neuron_id.
        Caller is responsible for halving all outgoing edge weights on both
        original and copy so the combined output equals the original.
        """
        new = Neuron(
            activation=self.activation_name,
            max_delay=self._max_delay,
            cell_type=self.cell_type,
        )
        with torch.no_grad():
            new.bias.copy_(self.bias)
        return new

    # ------------------------------------------------------------------
    # History management
    # ------------------------------------------------------------------

    def get_delayed(self, delay: int) -> torch.Tensor:
        if delay == 0:
            return self._current
        idx = delay - 1
        if idx >= len(self._history):
            return torch.zeros_like(self._current)
        return self._history[idx]

    def push_history(self) -> None:
        self._history.appendleft(self._current.detach().clone())

    def expand_history(self, new_max_delay: int) -> None:
        if new_max_delay <= self._max_delay:
            return
        pad = new_max_delay - self._max_delay
        for _ in range(pad):
            self._history.append(torch.zeros_like(self._current))
        self._history = deque(self._history, maxlen=new_max_delay)
        self._max_delay = new_max_delay

    def init_history_from_buffer(self, history_tensors: list[torch.Tensor]) -> None:
        self._history = deque(
            (h.detach().clone() for h in history_tensors[: self._max_delay]),
            maxlen=self._max_delay,
        )

    # ------------------------------------------------------------------
    # Legacy forward (kept for external callers; graph uses step() directly)
    # ------------------------------------------------------------------

    def forward(self, pre_activation: torch.Tensor) -> torch.Tensor:
        return self.step(pre_activation)

    # ------------------------------------------------------------------
    # State reset
    # ------------------------------------------------------------------

    def reset_state(self) -> None:
        zero = torch.zeros(1, device=self.bias.device)
        self._history = deque([zero] * self._max_delay, maxlen=self._max_delay)
        self._current = zero

    # ------------------------------------------------------------------
    # Properties / repr
    # ------------------------------------------------------------------

    @property
    def max_delay(self) -> int:
        return self._max_delay

    def extra_repr(self) -> str:
        return (
            f"id={self.neuron_id[:8]}, act={self.activation_name}, "
            f"max_delay={self._max_delay}, cell_type={self.cell_type}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# GRU Neuron
# ──────────────────────────────────────────────────────────────────────────────

class GRUNeuron(Neuron):
    """
    Scalar GRU cell embedded as a graph neuron.

    Receives a scalar weighted-sum input x (from incoming edges, same as a
    point neuron) and maintains a scalar hidden state h that persists across
    timesteps via the GRU gating equations:

        z = σ(wz·h_prev + uz·x + bz)          # update gate
        r = σ(wr·h_prev + ur·x + br)          # reset gate
        n = tanh(wn·r·h_prev + un·x + bn)     # candidate
        h = (1-z)·h_prev + z·n                # output

    bias (inherited) is not used — gate biases (bz, br, bn) replace it.
    Dale's Law and delay history work identically to PointNeuron.
    """

    _hidden_dim: int = 1  # scalar GRU; importance formula unchanged

    def __init__(
        self,
        neuron_id: Optional[str] = None,
        max_delay: int = 1,
        cell_type: str = "any",
    ) -> None:
        # activation_name="linear" is a placeholder — step() overrides forward
        super().__init__(
            activation="linear",
            neuron_id=neuron_id,
            max_delay=max_delay,
            cell_type=cell_type,
        )
        # Update gate
        self.wz = nn.Parameter(torch.zeros(1))
        self.uz = nn.Parameter(torch.zeros(1))
        self.bz = nn.Parameter(torch.zeros(1))
        # Reset gate
        self.wr = nn.Parameter(torch.zeros(1))
        self.ur = nn.Parameter(torch.zeros(1))
        self.br = nn.Parameter(torch.zeros(1))
        # Candidate
        self.wn = nn.Parameter(torch.zeros(1))
        self.un = nn.Parameter(torch.randn(1) * 0.1)   # small non-zero init
        self.bn = nn.Parameter(torch.zeros(1))

        # Internal hidden state (separate from _current which is the output)
        self._h: torch.Tensor = torch.zeros(1)

    # ------------------------------------------------------------------
    # Protocol — step
    # ------------------------------------------------------------------

    def step(self, weighted_sum: torch.Tensor) -> torch.Tensor:
        """
        GRU forward step.  weighted_sum is the aggregated input from edges.
        Maintains self._h (internal state) and sets self._current = self._h.
        """
        x = weighted_sum
        z = torch.sigmoid(self.wz * self._h.detach() + self.uz * x + self.bz)
        r = torch.sigmoid(self.wr * self._h.detach() + self.ur * x + self.br)
        n = torch.tanh(self.wn * r * self._h.detach() + self.un * x + self.bn)
        self._h = (1 - z) * self._h.detach() + z * n
        self._current = self._h
        return self._current

    # ------------------------------------------------------------------
    # Protocol — can_merge_with
    # ------------------------------------------------------------------

    def can_merge_with(self, other: "Neuron") -> bool:
        return type(other) is GRUNeuron

    # ------------------------------------------------------------------
    # Protocol — make_relay (inherited — always returns linear PointNeuron)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Protocol — split_copy
    # ------------------------------------------------------------------

    def split_copy(self) -> "GRUNeuron":
        new = GRUNeuron(max_delay=self._max_delay, cell_type=self.cell_type)
        with torch.no_grad():
            for attr in ("wz", "uz", "bz", "wr", "ur", "br", "wn", "un", "bn", "bias"):
                getattr(new, attr).copy_(getattr(self, attr))
        return new

    # ------------------------------------------------------------------
    # State reset
    # ------------------------------------------------------------------

    def reset_state(self) -> None:
        super().reset_state()
        self._h = torch.zeros(1, device=self.bias.device)

    def extra_repr(self) -> str:
        return (
            f"id={self.neuron_id[:8]}, type=GRU, "
            f"max_delay={self._max_delay}, cell_type={self.cell_type}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# LSTM Neuron
# ──────────────────────────────────────────────────────────────────────────────

class LSTMNeuron(Neuron):
    """
    Scalar LSTM cell embedded as a graph neuron.

    Receives a scalar weighted-sum input x and maintains scalar hidden state h
    and cell state c:

        f = σ(wf·h_prev + uf·x + bf)          # forget gate
        i = σ(wi·h_prev + ui·x + bi)          # input gate
        g = tanh(wg·h_prev + ug·x + bg)       # cell gate
        o = σ(wo·h_prev + uo·x + bo)          # output gate
        c = f·c_prev + i·g
        h = o·tanh(c)
    """

    _hidden_dim: int = 1

    def __init__(
        self,
        neuron_id: Optional[str] = None,
        max_delay: int = 1,
        cell_type: str = "any",
    ) -> None:
        super().__init__(
            activation="linear",
            neuron_id=neuron_id,
            max_delay=max_delay,
            cell_type=cell_type,
        )
        # Forget gate
        self.wf = nn.Parameter(torch.zeros(1))
        self.uf = nn.Parameter(torch.zeros(1))
        self.bf = nn.Parameter(torch.ones(1))    # positive init → forget gate open
        # Input gate
        self.wi = nn.Parameter(torch.zeros(1))
        self.ui = nn.Parameter(torch.zeros(1))
        self.bi = nn.Parameter(torch.zeros(1))
        # Cell gate
        self.wg = nn.Parameter(torch.zeros(1))
        self.ug = nn.Parameter(torch.randn(1) * 0.1)
        self.bg = nn.Parameter(torch.zeros(1))
        # Output gate
        self.wo = nn.Parameter(torch.zeros(1))
        self.uo = nn.Parameter(torch.zeros(1))
        self.bo = nn.Parameter(torch.ones(1))    # positive init → output gate open

        self._h: torch.Tensor = torch.zeros(1)
        self._c: torch.Tensor = torch.zeros(1)

    # ------------------------------------------------------------------
    # Protocol — step
    # ------------------------------------------------------------------

    def step(self, weighted_sum: torch.Tensor) -> torch.Tensor:
        x = weighted_sum
        h = self._h.detach()
        c = self._c.detach()
        f = torch.sigmoid(self.wf * h + self.uf * x + self.bf)
        i = torch.sigmoid(self.wi * h + self.ui * x + self.bi)
        g = torch.tanh(self.wg * h + self.ug * x + self.bg)
        o = torch.sigmoid(self.wo * h + self.uo * x + self.bo)
        self._c = f * c + i * g
        self._h = o * torch.tanh(self._c)
        self._current = self._h
        return self._current

    # ------------------------------------------------------------------
    # Protocol — can_merge_with
    # ------------------------------------------------------------------

    def can_merge_with(self, other: "Neuron") -> bool:
        return type(other) is LSTMNeuron

    # ------------------------------------------------------------------
    # Protocol — split_copy
    # ------------------------------------------------------------------

    def split_copy(self) -> "LSTMNeuron":
        new = LSTMNeuron(max_delay=self._max_delay, cell_type=self.cell_type)
        with torch.no_grad():
            for attr in ("wf","uf","bf","wi","ui","bi","wg","ug","bg","wo","uo","bo","bias"):
                getattr(new, attr).copy_(getattr(self, attr))
        return new

    # ------------------------------------------------------------------
    # State reset
    # ------------------------------------------------------------------

    def reset_state(self) -> None:
        super().reset_state()
        dev = self.bias.device
        self._h = torch.zeros(1, device=dev)
        self._c = torch.zeros(1, device=dev)

    def extra_repr(self) -> str:
        return (
            f"id={self.neuron_id[:8]}, type=LSTM, "
            f"max_delay={self._max_delay}, cell_type={self.cell_type}"
        )
