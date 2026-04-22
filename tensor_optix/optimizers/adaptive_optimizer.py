import logging
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np

from tensor_optix.core.base_optimizer import BaseOptimizer
from tensor_optix.core.types import EvalMetrics, HyperparamSet
from tensor_optix.optimizers.backoff_optimizer import BackoffOptimizer
from tensor_optix.optimizers.momentum_optimizer import MomentumOptimizer
from tensor_optix.optimizers.pbt_optimizer import PBTOptimizer
from tensor_optix.optimizers.spsa_optimizer import SPSAOptimizer

logger = logging.getLogger(__name__)


class AdaptiveOptimizer(BaseOptimizer):
    """
    Meta-optimizer that routes between SPSA, Momentum, Backoff, and PBT
    based on two mathematically grounded signals computed from the score history.

    ── Signal 1: Lag-1 autocorrelation (ρ) ─────────────────────────────────

        ρ = Pearson_Corr(scores[t-1], scores[t])   over the recent window
        ρ ∈ [-1, +1]  always, by definition.

        Interpretation:
            ρ > +autocorr_threshold
                Scores have positive serial correlation — each score predicts
                the next. The landscape has smooth momentum: the gradient
                direction is consistent across episodes.
                → MomentumOptimizer (Adam accumulates the consistent direction,
                  amplifying progress on smooth surfaces).

            ρ < -autocorr_threshold
                Scores oscillate — high then low then high. The gradient
                direction flips every episode. Gradient-magnitude methods
                (SPSA, Momentum) amplify noise rather than signal.
                → BackoffOptimizer (uses gradient sign only, not magnitude;
                  immune to direction flipping).

            |ρ| ≤ autocorr_threshold
                Scores are near i.i.d. — no serial structure. Standard
                unbiased gradient estimation is efficient.
                → SPSAOptimizer (2 episodes for N params, unbiased estimator).

    ── Signal 2: Relative performance gap (Δ) ───────────────────────────────

        Δ = (current_score − historical_best) / |historical_best|
        Δ ≤ 0  always (historical_best is the maximum ever seen).

        Interpretation:
            Δ < -gap_threshold
                The current hyperparameter configuration is producing scores
                well below the agent's own historical peak. The current region
                of hyperparameter space is bad. Gradient-based local search
                cannot escape a bad basin — a jump is needed.
                → PBTOptimizer (exploits the history of top-performing configs,
                  teleporting to a better basin then perturbing from there).

    ── Routing priority ─────────────────────────────────────────────────────

        Evaluated in order; first match wins:
            1. Δ < -gap_threshold         → PBT      (escape bad region first)
            2. ρ > +autocorr_threshold    → Momentum (amplify smooth trend)
            3. ρ < -autocorr_threshold    → Backoff  (tame oscillation)
            4. otherwise                  → SPSA     (default, balanced)

    ── Hysteresis ───────────────────────────────────────────────────────────

        The active optimizer is held for at least switch_patience consecutive
        evals before any switch. A single unlucky episode cannot cause a
        regime change. Once patience is satisfied, the signals are re-evaluated
        every eval.

    ── Sub-optimizer state ──────────────────────────────────────────────────

        All four optimizers receive on_improvement / on_plateau callbacks
        regardless of which is active. Their internal state (Adam moments,
        SPSA phase/delta, PBT history, Backoff probe cycle) stays warm at
        all times so switching is seamless with no cold-start penalty.
    """

    def __init__(
        self,
        param_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        log_params: Optional[List[str]] = None,
        # ── Routing thresholds ───────────────────────────────────────────
        autocorr_threshold: float = 0.3,
        gap_threshold: float = 0.20,
        # ── Hysteresis ───────────────────────────────────────────────────
        switch_patience: int = 3,
        switch_confirmation: int = 3,
        min_history: int = 10,
        # ── SPSA sub-optimizer ───────────────────────────────────────────
        spsa_perturbation_scale: float = 0.05,
        spsa_learning_rate: float = 0.1,
        spsa_warmup_episodes: int = 0,
        # ── Momentum sub-optimizer ───────────────────────────────────────
        momentum_alpha: float = 0.05,
        momentum_probe_scale: float = 0.05,
        # ── Backoff sub-optimizer ────────────────────────────────────────
        backoff_perturbation_scale: float = 0.05,
        # ── PBT sub-optimizer ────────────────────────────────────────────
        pbt_history_size: int = 50,
        pbt_explore_scale: float = 0.10,
        pbt_exploit_scale: float = 0.05,
    ):
        self._param_bounds = param_bounds or {}
        self._log_params = set(log_params or [])
        self._autocorr_threshold = autocorr_threshold
        self._gap_threshold = gap_threshold
        self._switch_patience = switch_patience
        self._min_history = min_history

        self._spsa = SPSAOptimizer(
            param_bounds=param_bounds,
            perturbation_scale=spsa_perturbation_scale,
            learning_rate=spsa_learning_rate,
            log_params=log_params,
            warmup_episodes=spsa_warmup_episodes,
        )
        self._momentum = MomentumOptimizer(
            param_bounds=param_bounds,
            alpha=momentum_alpha,
            probe_scale=momentum_probe_scale,
        )
        self._backoff = BackoffOptimizer(
            param_bounds=param_bounds,
            perturbation_scale=backoff_perturbation_scale,
        )
        self._pbt = PBTOptimizer(
            param_bounds=param_bounds,
            history_size=pbt_history_size,
            explore_scale=pbt_explore_scale,
            exploit_scale=pbt_exploit_scale,
            log_scale_params=self._log_params if self._log_params else None,
        )

        self._all: Dict[str, BaseOptimizer] = {
            "spsa":     self._spsa,
            "momentum": self._momentum,
            "backoff":  self._backoff,
            "pbt":      self._pbt,
        }

        self._switch_patience = switch_patience
        self._switch_confirmation = switch_confirmation

        self._active_name: str = "spsa"
        self._active: BaseOptimizer = self._spsa
        self._evals_on_active: int = 0
        self._pending_target: Optional[str] = None  # candidate for switch
        self._pending_count: int = 0               # consecutive evals confirming it
        self._score_window: deque = deque(maxlen=20)
        self._historical_best: Optional[float] = None

    # ------------------------------------------------------------------
    # Signal computation
    # ------------------------------------------------------------------

    def _lag1_autocorr(self) -> Optional[float]:
        """
        Pearson lag-1 autocorrelation of the score window.

        Returns None when fewer than min_history scores exist.
        Returns 0.0 when variance is degenerate (all scores identical).
        Result is always in [-1, 1].
        """
        scores = list(self._score_window)
        if len(scores) < self._min_history:
            return None
        x = np.array(scores[:-1], dtype=np.float64)
        y = np.array(scores[1:],  dtype=np.float64)
        if x.std() < 1e-10 or y.std() < 1e-10:
            return 0.0
        rho = float(np.corrcoef(x, y)[0, 1])
        if not np.isfinite(rho):
            return 0.0
        return float(np.clip(rho, -1.0, 1.0))

    def _relative_gap(self, current_score: float) -> Optional[float]:
        """
        (current_score - historical_best) / denom

        Always ≤ 0 since historical_best is the running maximum.
        Returns None before the first score is recorded.

        Denominator floor: max(|historical_best|, score_std, 1.0)
        — score_std auto-scales to the env's reward range so the gap
          stays meaningful when absolute scores are tiny (normalised envs,
          early training where historical_best ≈ 0).
        — The 1.0 absolute floor prevents blow-up in degenerate cases
          (all scores identical, score_std == 0).
        """
        if self._historical_best is None:
            return None
        scores = list(self._score_window)
        score_std = float(np.std(scores)) if len(scores) > 1 else 0.0
        denom = max(abs(self._historical_best), score_std, 1.0)
        return (current_score - self._historical_best) / denom

    def _select(self, current_score: float) -> str:
        """
        Apply the routing priority rules.
        Returns the active optimizer name unchanged when history is too short.
        """
        rho = self._lag1_autocorr()
        if rho is None:
            return self._active_name  # not enough history — hold current

        gap = self._relative_gap(current_score)

        # Priority 1: bad region — escape via PBT
        if gap is not None and gap < -self._gap_threshold:
            return "pbt"

        # Priority 2: smooth momentum — amplify with Adam
        if rho > self._autocorr_threshold:
            return "momentum"

        # Priority 3: oscillating — gradient sign only
        if rho < -self._autocorr_threshold:
            return "backoff"

        # Default: balanced SPSA
        return "spsa"

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

        latest_score = metrics_history[-1].primary_score
        self._score_window.append(latest_score)

        if self._historical_best is None or latest_score > self._historical_best:
            self._historical_best = latest_score

        # Re-evaluate routing after patience window
        if self._evals_on_active >= self._switch_patience:
            target = self._select(latest_score)
            if target != self._active_name:
                if target == self._pending_target:
                    self._pending_count += 1
                else:
                    self._pending_target = target
                    self._pending_count = 1

                if self._pending_count >= self._switch_confirmation:
                    rho = self._lag1_autocorr() or 0.0
                    gap = self._relative_gap(latest_score) or 0.0
                    logger.info(
                        "AdaptiveOptimizer: %s → %s  ρ=%.3f  Δ=%.3f",
                        self._active_name, target, rho, gap,
                    )
                    self._active_name = target
                    self._active = self._all[target]
                    self._evals_on_active = 0
                    self._pending_target = None
                    self._pending_count = 0
            else:
                self._pending_target = None
                self._pending_count = 0

        self._evals_on_active += 1
        return self._active.suggest(current_hyperparams, metrics_history)

    @property
    def is_probing(self) -> bool:
        """Delegates to the active optimizer — suppresses degradation detection during probes."""
        return getattr(self._active, "is_probing", False)

    @property
    def active_optimizer(self) -> str:
        """Name of the currently active sub-optimizer. Useful for logging and callbacks."""
        return self._active_name

    def on_improvement(self, metrics: EvalMetrics) -> None:
        """Broadcast to all sub-optimizers so their state stays warm."""
        for opt in self._all.values():
            opt.on_improvement(metrics)

    def on_plateau(self, metrics_history: List[EvalMetrics]) -> None:
        """Broadcast to all sub-optimizers so their state stays warm."""
        for opt in self._all.values():
            opt.on_plateau(metrics_history)
