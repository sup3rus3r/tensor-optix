import random
from collections import deque
from typing import Dict, List, Optional, Tuple
from tensor_optix.core.base_optimizer import BaseOptimizer
from tensor_optix.core.types import EvalMetrics, HyperparamSet


class PBTOptimizer(BaseOptimizer):
    """
    Pseudo Population-Based Training for single-agent use.

    Maintains a history of (HyperparamSet, primary_score) pairs as a virtual
    population. Uses PBT-style exploit/explore logic without parallel workers.

    Math (from PLAN.md):

    Exploit condition:
        if current_score < percentile(history_scores, 20):
            best_params = params from top 20% of history (by score)
            new_params = perturb(best_params, scale=small)

    Explore condition:
        else:
            new_params = perturb(current_params, scale=medium)

    Perturbation:
        for each param in param_bounds:
            δ = scale * (high - low)
            new_val = θ + uniform(-δ, +δ)
            new_val = clip(new_val, low, high)

    History: FIFO, keeps last history_size (default: 50) entries.
    Percentiles computed over this window only.
    """

    def __init__(
        self,
        param_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        history_size: int = 50,
        exploit_percentile: float = 0.20,    # bottom fraction triggers exploit
        top_percentile: float = 0.20,         # top fraction to copy from
        explore_scale: float = 0.10,          # perturbation as fraction of param range
        exploit_scale: float = 0.05,          # smaller perturbation when exploiting
    ):
        self._param_bounds = param_bounds or {}
        self._history: deque = deque(maxlen=history_size)
        self._exploit_percentile = exploit_percentile
        self._top_percentile = top_percentile
        self._explore_scale = explore_scale
        self._exploit_scale = exploit_scale

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
        Perturb each bounded param by uniform(−δ, +δ) where δ = scale * (high − low).
        Unbounded params are left unchanged.
        """
        new_params = dict(hyperparams.params)

        for param_name, (low, high) in self._param_bounds.items():
            if param_name not in new_params:
                continue
            current_value = new_params[param_name]
            if not isinstance(current_value, (int, float)):
                continue
            delta = scale * (high - low)
            noise = random.uniform(-delta, delta)
            new_value = float(current_value) + noise
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
