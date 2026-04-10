import random
from collections import deque
from typing import Dict, List, Optional, Tuple
from tensor_optix.core.base_optimizer import BaseOptimizer
from tensor_optix.core.types import EvalMetrics, HyperparamSet


class BackoffOptimizer(BaseOptimizer):
    """
    Staggered two-phase finite difference optimizer.

    All probing is done in normalized [0, 1] param space:
        x = (θ - lo) / (hi - lo)

    This makes delta scale-invariant across params of wildly different
    magnitudes (e.g. lr=3e-4 and clip_ratio=0.2 both probe with the same
    fractional step in their respective ranges). Without normalization,
    perturbation_scale * |θ| collapses to min_delta for small params.

    Each bounded param gets its own independent probe/commit cycle:

    Phase 1 — PROBE:
        Normalize current value to x ∈ [0,1].
        Apply x + δ (δ = perturbation_scale, a fraction of the unit range).
        Denormalize back to raw space and apply to agent.
        Record base_score.

    Phase 2 — COMMIT:
        gradient = (probe_score - base_score) / δ   (in normalized space)
        If gradient > 0: keep probe value
        If gradient < 0: apply x - δ (reflected step)
        If gradient ≈ 0: keep current value
        Advance to next param.

    Params are cycled round-robin. With N bounded params, a full cycle
    takes 2N episodes (probe + commit per param).

    On improvement: increase perturbation_scale (explore more aggressively).
    On plateau: increase perturbation_scale further, reset cycle.
    """

    def __init__(
        self,
        param_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        perturbation_scale: float = 0.05,
        min_delta: float = 1e-4,        # minimum step in normalized [0,1] space
        gradient_threshold: float = 1e-10,
    ):
        self._param_bounds = param_bounds or {}
        self._perturbation_scale = perturbation_scale
        self._min_delta = min_delta
        self._gradient_threshold = gradient_threshold

        self._param_names: List[str] = []
        self._current_param_idx: int = 0

        # Per-param probe state
        self._phase: Dict[str, str] = {}
        self._probe_delta: Dict[str, float] = {}    # δ in normalized space
        self._base_score: Dict[str, float] = {}
        self._probe_value_raw: Dict[str, float] = {}  # denormalized probe value
        self._current_x: Dict[str, float] = {}       # normalized current value
        self._currently_probing: bool = False

    # ------------------------------------------------------------------
    # Normalization helpers
    # ------------------------------------------------------------------

    def _normalize(self, value: float, lo: float, hi: float) -> float:
        """Map raw param value to [0, 1]."""
        span = hi - lo
        if span == 0:
            return 0.0
        return (value - lo) / span

    def _denormalize(self, x: float, lo: float, hi: float) -> float:
        """Map normalized [0, 1] value back to raw param space."""
        return lo + x * (hi - lo)

    # ------------------------------------------------------------------

    def suggest(
        self,
        current_hyperparams: HyperparamSet,
        metrics_history: List[EvalMetrics],
    ) -> HyperparamSet:
        if not metrics_history:
            return current_hyperparams.copy()

        if not self._param_names:
            self._param_names = [
                k for k in current_hyperparams.params
                if k in self._param_bounds
                and isinstance(current_hyperparams.params[k], (int, float))
            ]
            if not self._param_names:
                return current_hyperparams.copy()
            for name in self._param_names:
                self._phase[name] = "probe"

        new_params = dict(current_hyperparams.params)
        latest_score = metrics_history[-1].primary_score

        param_name = self._param_names[self._current_param_idx]
        lo, hi = self._param_bounds[param_name]
        current_value = float(current_hyperparams.params[param_name])
        x_current = self._normalize(current_value, lo, hi)

        if self._phase[param_name] == "probe":
            # Probe in normalized space — delta is a fixed fraction of [0,1]
            delta = max(self._perturbation_scale, self._min_delta)
            x_probe = min(1.0, x_current + delta)
            # If already at upper bound, probe downward instead
            if x_probe == x_current:
                x_probe = max(0.0, x_current - delta)
            probe_value_raw = self._denormalize(x_probe, lo, hi)

            self._phase[param_name] = "commit"
            self._probe_delta[param_name] = delta
            self._base_score[param_name] = latest_score
            self._probe_value_raw[param_name] = probe_value_raw
            self._current_x[param_name] = x_current
            self._currently_probing = True

            new_params[param_name] = probe_value_raw

        else:
            # Commit: measure gradient in normalized space, decide direction
            probe_score = latest_score
            base_score = self._base_score[param_name]
            delta = self._probe_delta[param_name]
            x_cur = self._current_x[param_name]
            probe_raw = self._probe_value_raw[param_name]

            gradient = (probe_score - base_score) / delta

            if abs(gradient) <= self._gradient_threshold:
                new_params[param_name] = current_value
            elif gradient > 0:
                new_params[param_name] = probe_raw
            else:
                x_opposite = max(0.0, x_cur - delta)
                new_params[param_name] = self._denormalize(x_opposite, lo, hi)

            self._phase[param_name] = "probe"
            self._current_param_idx = (self._current_param_idx + 1) % len(self._param_names)
            self._currently_probing = False

        return HyperparamSet(
            params=new_params,
            episode_id=current_hyperparams.episode_id,
        )

    @property
    def is_probing(self) -> bool:
        return self._currently_probing

    def on_improvement(self, metrics: EvalMetrics) -> None:
        # Widen exploration slightly after improvement
        self._perturbation_scale = min(self._perturbation_scale * 1.2, 0.3)

    def on_plateau(self, metrics_history: List[EvalMetrics]) -> None:
        # Plateau: push harder, reset cycle so all params get re-probed
        self._perturbation_scale = min(self._perturbation_scale * 2.0, 0.5)
        self._current_param_idx = 0
        for name in self._param_names:
            self._phase[name] = "probe"
