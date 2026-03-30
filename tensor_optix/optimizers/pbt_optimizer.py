import math
import random
from collections import deque
from typing import Dict, List, Optional, Set, Tuple
from tensor_optix.core.base_optimizer import BaseOptimizer
from tensor_optix.core.types import EvalMetrics, HyperparamSet


class PBTOptimizer(BaseOptimizer):
    """
    Pseudo Population-Based Training for single-agent use.

    Maintains a history of (HyperparamSet, primary_score) pairs as a virtual
    population. Uses PBT-style exploit/explore logic without parallel workers.

    Math:

    Exploit condition:
        if current_score < percentile(history_scores, 20):
            best_params = params from top 20% of history (by score)
            new_params = perturb(best_params, scale=small)

    Explore condition:
        else:
            new_params = perturb(current_params, scale=medium)

    Perturbation — two modes per parameter:

    Linear (default):
        δ = scale * (high - low)
        new_val = clip(θ + uniform(-δ, +δ), low, high)

    Log-scale (for params in log_scale_params, e.g. learning_rate):
        Multiplicative perturbation in log space is correct for parameters
        where the meaningful range spans orders of magnitude. Additive noise
        on [1e-4, 1e-1] would undersample the lower end and oversample the top.

        Following Jaderberg et al. (2017) PBT paper:
            new_val = clip(θ * exp(uniform(-δ_log, +δ_log)), low, high)
        where δ_log = scale * log(high / low)

        This ensures equal probability mass per decade regardless of position.

    History: FIFO, keeps last history_size (default: 50) entries.
    Percentiles computed over this window only.
    """

    # Default params that should always use log-scale perturbation.
    # Users can override via log_scale_params constructor arg.
    _DEFAULT_LOG_SCALE_PARAMS: Set[str] = {
        "learning_rate", "lr", "alpha", "epsilon", "weight_decay",
    }

    def __init__(
        self,
        param_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        history_size: int = 50,
        exploit_percentile: float = 0.20,    # bottom fraction triggers exploit
        top_percentile: float = 0.20,         # top fraction to copy from
        explore_scale: float = 0.10,          # perturbation magnitude as fraction of range
        exploit_scale: float = 0.05,          # smaller perturbation when exploiting
        log_scale_params: Optional[Set[str]] = None,  # override default log-scale set
    ):
        self._param_bounds = param_bounds or {}
        self._history: deque = deque(maxlen=history_size)
        self._exploit_percentile = exploit_percentile
        self._top_percentile = top_percentile
        self._explore_scale = explore_scale
        self._exploit_scale = exploit_scale
        self._log_scale_params: Set[str] = (
            log_scale_params if log_scale_params is not None
            else self._DEFAULT_LOG_SCALE_PARAMS
        )

    def suggest(
        self,
        current_hyperparams: HyperparamSet,
        metrics_history: List[EvalMetrics],
    ) -> HyperparamSet:
        if not metrics_history:
            return current_hyperparams.copy()

        latest_score = metrics_history[-1].primary_score

        # Record current configuration in history
        self._history.append((current_hyperparams.copy(), latest_score))

        if len(self._history) < 3:
            # Not enough history — small random perturbation
            return self._perturb(current_hyperparams, self._explore_scale)

        scores = [entry[1] for entry in self._history]
        scores_sorted = sorted(scores)
        n = len(scores_sorted)

        exploit_threshold = scores_sorted[max(0, int(n * self._exploit_percentile) - 1)]
        top_threshold = scores_sorted[max(0, int(n * (1.0 - self._top_percentile)))]

        if latest_score <= exploit_threshold:
            # Bottom percentile — exploit: copy from top performers
            top_entries = [
                entry for entry in self._history if entry[1] >= top_threshold
            ]
            if top_entries:
                best_entry = max(top_entries, key=lambda e: e[1])
                source_params = best_entry[0]
                return self._perturb(source_params, self._exploit_scale)

        # Not bottom percentile — explore around current params
        return self._perturb(current_hyperparams, self._explore_scale)

    def _perturb(self, hyperparams: HyperparamSet, scale: float) -> HyperparamSet:
        """
        Perturb each bounded param. Mode is selected per parameter:

        Log-scale params (e.g. learning_rate): multiplicative perturbation.
            δ_log = scale * log(high / low)
            new_val = clip(θ * exp(uniform(-δ_log, +δ_log)), low, high)
            Preserves equal probability mass per decade — correct for params
            that span orders of magnitude (Jaderberg et al. 2017, PBT).

        Linear params: additive uniform perturbation.
            δ = scale * (high - low)
            new_val = clip(θ + uniform(-δ, +δ), low, high)

        Unbounded params are left unchanged.
        """
        new_params = dict(hyperparams.params)

        for param_name, (low, high) in self._param_bounds.items():
            if param_name not in new_params:
                continue
            current_value = new_params[param_name]
            if not isinstance(current_value, (int, float)):
                continue

            theta = float(current_value)

            if param_name in self._log_scale_params and low > 0.0 and high > low:
                # Log-scale perturbation: uniform noise in log space
                log_range = math.log(high / low)
                delta_log = scale * log_range
                noise = random.uniform(-delta_log, delta_log)
                new_value = theta * math.exp(noise)
            else:
                # Linear perturbation
                delta = scale * (high - low)
                noise = random.uniform(-delta, delta)
                new_value = theta + noise

            new_params[param_name] = max(low, min(high, new_value))

        return HyperparamSet(
            params=new_params,
            episode_id=hyperparams.episode_id,
        )

    def on_improvement(self, metrics: EvalMetrics) -> None:
        pass

    def on_plateau(self, metrics_history: List[EvalMetrics]) -> None:
        # On plateau, clear history to force fresh exploration
        self._history.clear()
