"""
tensor_optix.adapters.jax.flax_evaluator — Evaluator for Flax-based agents.
"""

from __future__ import annotations

from tensor_optix.core.base_evaluator import BaseEvaluator
from tensor_optix.core.types import EpisodeData, EvalMetrics


class FlaxEvaluator(BaseEvaluator):
    """
    Evaluator for Flax agents.

    Returns total episode reward as the primary optimisation score.
    Training diagnostics (e.g. loss, entropy) are forwarded as secondary metrics.
    """

    def score(self, episode_data: EpisodeData, train_diagnostics: dict) -> EvalMetrics:
        total = float(sum(episode_data.rewards))
        metrics = {"total_reward": total}
        metrics.update({k: float(v) for k, v in train_diagnostics.items()})
        return EvalMetrics(
            primary_score=total,
            metrics=metrics,
            episode_id=episode_data.episode_id,
        )
