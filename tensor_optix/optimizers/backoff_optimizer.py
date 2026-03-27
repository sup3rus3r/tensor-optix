import random
from collections import deque
from typing import Dict, List, Optional, Tuple
from tensor_optix.core.base_optimizer import BaseOptimizer
from tensor_optix.core.types import EvalMetrics, HyperparamSet


class BackoffOptimizer(BaseOptimizer):
    """
    Staggered two-phase finite difference optimizer.

    Each bounded param gets its own independent probe/commit cycle:

    Phase 1 — PROBE:
        Apply θᵢ + δᵢ to the agent (δᵢ = perturbation_scale * |θᵢ|).
        Record base_score (score before the probe).
        Run one episode.

    Phase 2 — COMMIT:
        Measure probe_score from the completed episode.
        gradient = (probe_score - base_score) / δᵢ
        If gradient > 0: keep θᵢ + δᵢ  (moving up was good)
        If gradient < 0: apply θᵢ - δᵢ  (moving up was bad, go the other way)
        If gradient ≈ 0: keep θᵢ unchanged (no signal)
        Move to next param.

    Params are cycled round-robin. With N bounded params, a full cycle
    takes N episodes. Each param is updated independently.

    Adaptive step size:
        δᵢ = perturbation_scale * |θᵢ|   (multiplicative, scale-invariant)
        δᵢ = max(δᵢ, min_delta)           (floor to avoid zero)

    On improvement: increase perturbation scale (explore more aggressively).
    On plateau: increase perturbation scale even more, reset cycle.
    """

    def __init__(
        self,
        param_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        perturbation_scale: float = 0.05,
        min_delta: float = 1e-7,
        gradient_threshold: float = 1e-10,  # treat gradient below this as zero
    ):
        self._param_bounds = param_bounds or {}
        self._perturbation_scale = perturbation_scale
        self._min_delta = min_delta
        self._gradient_threshold = gradient_threshold

        # Ordered list of params being tuned (set on first suggest call)
        self._param_names: List[str] = []
        self._current_param_idx: int = 0

        # Per-param probe state
        # phase: "probe" = about to probe, "commit" = probe applied, awaiting result
        self._phase: Dict[str, str] = {}
        self._probe_delta: Dict[str, float] = {}    # δᵢ used for this probe
        self._base_score: Dict[str, float] = {}     # score before probe
        self._probe_value: Dict[str, float] = {}    # θᵢ + δᵢ applied

    def suggest(
        self,
        current_hyperparams: HyperparamSet,
        metrics_history: List[EvalMetrics],
    ) -> HyperparamSet:
        if not metrics_history:
            return current_hyperparams.copy()

        # Initialize param list on first call
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

        # Get current param being worked on
        param_name = self._param_names[self._current_param_idx]
        low, high = self._param_bounds[param_name]
        current_value = float(current_hyperparams.params[param_name])

        if self._phase[param_name] == "probe":
            # Phase 1: apply probe, record base score
            delta = max(self._perturbation_scale * abs(current_value), self._min_delta)
            probe_value = current_value + delta
            probe_value = max(low, min(high, probe_value))

            self._phase[param_name] = "commit"
            self._probe_delta[param_name] = delta
            self._base_score[param_name] = latest_score
            self._probe_value[param_name] = probe_value

            new_params[param_name] = probe_value

        else:
            # Phase 2: commit — measure gradient, decide direction
            probe_score = latest_score
            base_score = self._base_score[param_name]
            delta = self._probe_delta[param_name]
            probe_value = self._probe_value[param_name]

            gradient = (probe_score - base_score) / delta

            if abs(gradient) <= self._gradient_threshold:
                # No signal — keep current value unchanged
                new_params[param_name] = current_value
            elif gradient > 0:
                # Probe direction was good — keep probe value
                new_params[param_name] = probe_value
            else:
                # Probe direction was bad — go the other way
                opposite = current_value - delta
                new_params[param_name] = max(low, min(high, opposite))

            # Reset phase for this param, advance to next
            self._phase[param_name] = "probe"
            self._current_param_idx = (self._current_param_idx + 1) % len(self._param_names)

        return HyperparamSet(
            params=new_params,
            episode_id=current_hyperparams.episode_id,
        )

    def on_improvement(self, metrics: EvalMetrics) -> None:
        # Widen exploration slightly after improvement
        self._perturbation_scale = min(self._perturbation_scale * 1.2, 0.3)

    def on_plateau(self, metrics_history: List[EvalMetrics]) -> None:
        # Plateau: push harder, reset cycle so all params get re-probed
        self._perturbation_scale = min(self._perturbation_scale * 2.0, 0.5)
        self._current_param_idx = 0
        for name in self._param_names:
            self._phase[name] = "probe"
