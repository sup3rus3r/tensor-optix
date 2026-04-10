import numpy as np
from typing import Dict, List, Optional, Tuple

from tensor_optix.core.base_optimizer import BaseOptimizer
from tensor_optix.core.types import EvalMetrics, HyperparamSet


class MomentumOptimizer(BaseOptimizer):
    """
    Adam-momentum finite-difference optimizer for hyperparameters.

    Treats each bounded hyperparam as a coordinate in a continuous
    search space and applies Adam to ascend the performance surface.

    Gradient estimation — finite difference (probe/commit cycle):

        δᵢ = probe_scale × |θᵢ|
        gᵢ = (score_probe − score_base) / δᵢ   ← raw gradient

    Adam update in normalized [0,1] parameter space:

        mᵢ ← β₁·mᵢ + (1−β₁)·gᵢ               (1st moment — momentum)
        vᵢ ← β₂·vᵢ + (1−β₂)·gᵢ²              (2nd moment — RMSProp)
        m̂ᵢ = mᵢ / (1 − β₁ᵗ)                   (bias correction)
        v̂ᵢ = vᵢ / (1 − β₂ᵗ)                   (bias correction)
        xᵢ ← clip(xᵢ + α · m̂ᵢ/√(v̂ᵢ+ε), 0, 1) (normalize to bounds)

    All params are normalized to [0,1] before updates so no param
    dominates because of its raw scale. α is a fraction of the param range.

    Why Adam over plain finite-difference (BackoffOptimizer):
      - Momentum (β₁): smooths noisy stochastic gradient estimates
        across episodes — critical when RL rewards are high-variance
      - Adaptive rates (β₂): params with inconsistent gradients get
        smaller steps; stable-gradient params get larger steps
      - Exponential decay: old gradient info decays naturally so the
        optimizer tracks a non-stationary landscape as the policy evolves
      - Bias correction: unbiased estimates from the very first episode

    Usage:
        optimizer = MomentumOptimizer(param_bounds={
            "learning_rate": (1e-5, 1e-2),
            "clip_ratio":    (0.05, 0.4),
            "entropy_coef":  (0.0,  0.1),
        })
        rl_opt = RLOptimizer(..., optimizer=optimizer)
    """

    def __init__(
        self,
        param_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        alpha: float = 0.05,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        probe_scale: float = 0.05,
        min_delta: float = 1e-7,
    ):
        """
        Args:
            param_bounds: {param_name: (low, high)} — defines the search space.
                Only params listed here are tuned. Others pass through unchanged.
            alpha: Step size in normalized [0,1] parameter space.
                0.05 = move at most 5% of the param range per update.
                Conservative by design — RL policies are sensitive to LR jumps.
            beta1: 1st moment (momentum) decay. 0.9 means the gradient EMA
                carries 90% of the previous estimate each step.
            beta2: 2nd moment (RMSProp) decay. 0.999 gives a very long
                history for the variance estimate — stable adaptive rates.
            eps: Numerical stability floor in the Adam denominator.
            probe_scale: Finite-difference step: δ = probe_scale × |θ|.
                0.05 = probe ±5% of current value to estimate the gradient.
            min_delta: Absolute minimum δ (prevents divide-by-zero when θ≈0).
        """
        self._param_bounds = param_bounds or {}
        self._alpha  = alpha
        self._beta1  = beta1
        self._beta2  = beta2
        self._eps    = eps
        self._probe_scale = probe_scale
        self._min_delta   = min_delta

        self._param_names: List[str] = []
        self._current_param_idx: int = 0

        # Adam state — one entry per bounded param
        self._m: Dict[str, float] = {}   # 1st moment (mean)
        self._v: Dict[str, float] = {}   # 2nd moment (uncentered variance)
        self._t: Dict[str, int]   = {}   # per-param step counter

        # Probe/commit state
        self._phase:       Dict[str, str]   = {}  # "probe" | "commit"
        self._base_score:  Dict[str, float] = {}
        self._probe_delta: Dict[str, float] = {}
        self._probe_value: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Normalization helpers
    # ------------------------------------------------------------------

    def _normalize(self, name: str, val: float) -> float:
        lo, hi = self._param_bounds[name]
        return (val - lo) / (hi - lo + 1e-12)

    def _denormalize(self, name: str, x: float) -> float:
        lo, hi = self._param_bounds[name]
        return float(np.clip(lo + x * (hi - lo), lo, hi))

    # ------------------------------------------------------------------
    # BaseOptimizer interface
    # ------------------------------------------------------------------

    def suggest(
        self,
        current_hyperparams: HyperparamSet,
        metrics_history: List[EvalMetrics],
    ) -> HyperparamSet:
        if not metrics_history:
            return current_hyperparams.copy()

        # Build tunable param list on first call
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
                self._m[name] = 0.0
                self._v[name] = 0.0
                self._t[name] = 0

        new_params  = dict(current_hyperparams.params)
        latest_score = float(metrics_history[-1].primary_score)

        param_name  = self._param_names[self._current_param_idx]
        lo, hi      = self._param_bounds[param_name]
        current_val = float(current_hyperparams.params[param_name])

        if self._phase[param_name] == "probe":
            # ── Phase 1: apply +δ probe, record base score ──────────────
            delta     = max(self._probe_scale * abs(current_val), self._min_delta)
            probe_val = float(np.clip(current_val + delta, lo, hi))

            self._phase[param_name]       = "commit"
            self._base_score[param_name]  = latest_score
            self._probe_delta[param_name] = delta
            self._probe_value[param_name] = probe_val

            new_params[param_name] = probe_val

        else:
            # ── Phase 2: commit — estimate gradient, apply Adam ──────────
            probe_score = latest_score
            base_score  = self._base_score[param_name]
            delta       = self._probe_delta[param_name]

            # Raw finite-difference gradient: score / param_unit
            g_raw = (probe_score - base_score) / (delta + 1e-12)

            # Convert to gradient in normalized [0,1] space:
            #   dx/dθ = 1/(hi-lo)  ⟹  d_score/dx = g_raw × (hi-lo)
            g = g_raw * (hi - lo + 1e-12)

            # Adam update
            self._t[param_name] += 1
            t = self._t[param_name]

            self._m[param_name] = (
                self._beta1 * self._m[param_name] + (1.0 - self._beta1) * g
            )
            self._v[param_name] = (
                self._beta2 * self._v[param_name] + (1.0 - self._beta2) * g * g
            )

            m_hat = self._m[param_name] / (1.0 - self._beta1 ** t)
            v_hat = self._v[param_name] / (1.0 - self._beta2 ** t)

            # Step in normalized space → convert back to raw
            x_cur = self._normalize(param_name, current_val)
            x_new = float(np.clip(
                x_cur + self._alpha * m_hat / (np.sqrt(v_hat) + self._eps),
                0.0, 1.0,
            ))
            new_params[param_name] = self._denormalize(param_name, x_new)

            # Advance to next param
            self._phase[param_name]    = "probe"
            self._current_param_idx    = (
                (self._current_param_idx + 1) % len(self._param_names)
            )

        return HyperparamSet(
            params=new_params,
            episode_id=current_hyperparams.episode_id,
        )

    def on_improvement(self, metrics: EvalMetrics) -> None:
        # Adam's adaptive rates already handle this — no manual adjustment.
        pass

    def on_plateau(self, metrics_history: List[EvalMetrics]) -> None:
        # Reset probe cycle so every param gets re-evaluated from the new
        # weights after a PolicyManager spawn.
        self._current_param_idx = 0
        for name in self._param_names:
            self._phase[name] = "probe"
