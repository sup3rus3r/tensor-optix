import numpy as np
from typing import Dict, List, Optional, Tuple
from tensor_optix.core.base_optimizer import BaseOptimizer
from tensor_optix.core.types import EvalMetrics, HyperparamSet


class SPSAOptimizer(BaseOptimizer):
    """
    SPSA (Simultaneous Perturbation Stochastic Approximation) optimizer.

    Reference: Spall, J.C. (1992). "Multivariate Stochastic Approximation
    Using a Simultaneous Perturbation Gradient Approximation."
    IEEE Transactions on Automatic Control, 37(3), 332-341.

    All N bounded params are updated in exactly 2 episodes regardless of N,
    compared to BackoffOptimizer's 2N episodes (round-robin). For 3 params
    this is 3x faster; for 5 params it is 5x faster.

    Algorithm (all operations in normalized [0,1] param space):

    Episode 1 — PLUS probe:
        Sample Δ ∈ {-1, +1}^N  (Rademacher vector, each entry iid uniform)
        Apply x⁺ = clip(x + c·Δ, 0, 1)  where c = perturbation_scale
        Record f⁺ (score after episode)

    Episode 2 — MINUS probe:
        Apply x⁻ = clip(x - c·Δ, 0, 1)
        Record f⁻ (score after episode)

    COMMIT (after episode 2):
        ĝᵢ = (f⁺ - f⁻) / (2c·Δᵢ)   for each param i
             (unbiased gradient estimate in normalized space)
        x_new_i = clip(x_i + α·ĝᵢ, 0, 1)
        θ_new_i = denormalize(x_new_i, lo_i, hi_i)

    where:
        c = perturbation_scale   (step size for probing, default 0.05)
        α = learning_rate        (step size for update, default 0.1)

    The Rademacher distribution (equal prob ±1 per component) is used
    instead of Gaussian because it minimises the variance of the gradient
    estimator for a given c (Spall 1992, Theorem 1).

    is_probing returns True during the plus and minus probe episodes so
    LoopController suppresses degradation detection (the score drop is
    self-inflicted, not a policy collapse).
    """

    # Internal phase constants
    _PHASE_PLUS  = "plus"   # about to run the +c·Δ episode
    _PHASE_MINUS = "minus"  # about to run the -c·Δ episode
    _PHASE_IDLE  = "idle"   # waiting for next suggest() call to start a new cycle

    def __init__(
        self,
        param_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        perturbation_scale: float = 0.05,   # c: probe step in normalized space
        learning_rate: float = 0.1,          # α: update step in normalized space
        min_perturbation: float = 1e-4,      # floor on c to avoid zero
        gradient_clip: float = 1.0,          # clip |ĝᵢ| to this in normalized space
                                             # max normalized step = alpha * clip = 0.1 * 1.0 = 10%
        log_params: Optional[List[str]] = None,  # params to normalize in log space
        warmup_episodes: int = 0,            # episodes before SPSA starts updating
    ):
        self._param_bounds = param_bounds or {}
        self._c = perturbation_scale
        self._alpha = learning_rate
        self._min_c = min_perturbation
        self._grad_clip = gradient_clip
        self._log_params: set = set(log_params or [])
        self._warmup_episodes = warmup_episodes
        self._episodes_seen: int = 0

        self._param_names: List[str] = []
        self._phase: str = self._PHASE_IDLE
        self._currently_probing: bool = False

        # State for the current 2-episode cycle
        self._delta: Optional[np.ndarray] = None     # Rademacher vector
        self._x_base: Optional[np.ndarray] = None    # normalized current values
        self._x_plus: Optional[np.ndarray] = None    # normalized plus-probe values
        self._f_plus: Optional[float] = None         # score after plus probe
        self._step: int = 0                           # Adam-style iteration counter

    # ------------------------------------------------------------------
    # Normalization helpers
    # ------------------------------------------------------------------

    def _normalize(self, values: np.ndarray) -> np.ndarray:
        """Map raw param vector to [0,1]^N. Log params normalized in log space."""
        result = np.empty_like(values)
        for i, name in enumerate(self._param_names):
            lo, hi = self._param_bounds[name]
            if name in self._log_params and lo > 0 and hi > 0:
                log_lo, log_hi = np.log(lo), np.log(hi)
                span = log_hi - log_lo
                result[i] = (np.log(max(values[i], lo)) - log_lo) / span if span > 0 else 0.0
            else:
                span = hi - lo
                result[i] = (values[i] - lo) / span if span > 0 else 0.0
        return result

    def _denormalize(self, x: np.ndarray) -> np.ndarray:
        """Map normalized [0,1]^N vector back to raw param space."""
        result = np.empty_like(x)
        for i, name in enumerate(self._param_names):
            lo, hi = self._param_bounds[name]
            if name in self._log_params and lo > 0 and hi > 0:
                log_lo, log_hi = np.log(lo), np.log(hi)
                result[i] = np.exp(log_lo + x[i] * (log_hi - log_lo))
            else:
                result[i] = lo + x[i] * (hi - lo)
        return result

    def _current_raw(self, hyperparams: HyperparamSet) -> np.ndarray:
        return np.array(
            [float(hyperparams.params[n]) for n in self._param_names],
            dtype=np.float64,
        )

    # ------------------------------------------------------------------

    def suggest(
        self,
        current_hyperparams: HyperparamSet,
        metrics_history: List[EvalMetrics],
    ) -> HyperparamSet:
        if not metrics_history:
            return current_hyperparams.copy()

        self._episodes_seen += 1
        if self._episodes_seen <= self._warmup_episodes:
            # Warmup blackout: let the policy stabilize before SPSA touches anything.
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

        latest_score = metrics_history[-1].primary_score
        new_params = dict(current_hyperparams.params)

        if self._phase == self._PHASE_IDLE:
            # Start new cycle: sample Δ, apply +c·Δ probe
            raw = self._current_raw(current_hyperparams)
            self._x_base = self._normalize(raw)
            c = max(self._c, self._min_c)

            # Rademacher vector: each entry independently ±1 with equal prob
            self._delta = np.where(
                np.random.randint(0, 2, size=len(self._param_names)),
                1.0, -1.0
            )
            self._x_plus = np.clip(self._x_base + c * self._delta, 0.0, 1.0)
            raw_plus = self._denormalize(self._x_plus)
            for i, name in enumerate(self._param_names):
                new_params[name] = float(raw_plus[i])

            self._phase = self._PHASE_PLUS
            self._currently_probing = True

        elif self._phase == self._PHASE_PLUS:
            # Record f⁺, apply -c·Δ probe
            self._f_plus = latest_score
            c = max(self._c, self._min_c)
            x_minus = np.clip(self._x_base - c * self._delta, 0.0, 1.0)
            raw_minus = self._denormalize(x_minus)
            for i, name in enumerate(self._param_names):
                new_params[name] = float(raw_minus[i])

            self._phase = self._PHASE_MINUS
            self._currently_probing = True

        elif self._phase == self._PHASE_MINUS:
            # Record f⁻, compute gradient, apply update
            f_minus = latest_score
            f_plus = self._f_plus
            c = max(self._c, self._min_c)
            self._step += 1

            # SPSA gradient estimate: ĝᵢ = (f⁺ - f⁻) / (2c·Δᵢ)
            grad = (f_plus - f_minus) / (2.0 * c * self._delta)

            # Clip gradient to prevent runaway updates on noisy episodes
            grad = np.clip(grad, -self._grad_clip, self._grad_clip)

            # Gradient ascent step in normalized space (higher score = better)
            x_new = np.clip(self._x_base + self._alpha * grad, 0.0, 1.0)
            raw_new = self._denormalize(x_new)
            for i, name in enumerate(self._param_names):
                new_params[name] = float(raw_new[i])

            self._phase = self._PHASE_IDLE
            self._currently_probing = False

        return HyperparamSet(
            params=new_params,
            episode_id=current_hyperparams.episode_id,
        )

    @property
    def is_probing(self) -> bool:
        return self._currently_probing

    def on_improvement(self, metrics: EvalMetrics) -> None:
        # Agent is converging — shrink probes to avoid disrupting an active learning phase.
        # Probing with large perturbations while the policy is improving causes collapses.
        self._c = max(self._c * 0.8, self._min_c)

    def on_plateau(self, metrics_history: List[EvalMetrics]) -> None:
        # Stuck — widen probes to explore the param space more aggressively.
        self._c = min(self._c * 2.0, 0.2)
        # Reset to idle so next suggest() starts a fresh cycle
        self._phase = self._PHASE_IDLE
        self._currently_probing = False
