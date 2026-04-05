import numpy as np
from tensor_optix.core.base_evaluator import BaseEvaluator
from tensor_optix.core.types import EpisodeData, EvalMetrics


class TorchEvaluator(BaseEvaluator):
    """
    Default evaluator for PyTorch-based RL agents.

    Identical scoring logic to TFEvaluator — no TensorFlow dependency.
    primary_score is a weighted combination of:
    - mean_reward:          average per-step reward over the window
    - reward_stability:     negative std of rewards (lower variance = higher score)
    - episode_length_score: normalized episode length (0 weight by default)

    Usage:
        evaluator = TorchEvaluator()
        # Or with a fully custom scorer:
        evaluator = TorchEvaluator(primary_score_fn=lambda ep, diag: sum(ep.rewards))
    """

    def __init__(
        self,
        reward_weight: float = 1.0,
        stability_weight: float = 1.0,
        length_weight: float = 0.0,
        primary_score_fn=None,
        max_episode_length: int = 1000,
    ):
        total = reward_weight + stability_weight + length_weight
        if total <= 0:
            raise ValueError("At least one weight must be positive")
        self._reward_w   = reward_weight   / total
        self._stability_w = stability_weight / total
        self._length_w   = length_weight   / total
        self._primary_score_fn = primary_score_fn
        self._max_episode_length = max_episode_length

    def score(self, episode_data: EpisodeData, train_diagnostics: dict) -> EvalMetrics:
        rewards = np.array(episode_data.rewards, dtype=np.float32)

        if self._primary_score_fn is not None:
            primary = float(self._primary_score_fn(episode_data, train_diagnostics))
            metrics = {
                "primary_score": primary,
                "total_reward":  float(rewards.sum()),
                "mean_reward":   float(rewards.mean()),
                "episode_length": episode_data.length,
            }
            metrics.update({k: float(v) for k, v in train_diagnostics.items()
                             if isinstance(v, (int, float))})
            return EvalMetrics(primary_score=primary, metrics=metrics,
                               episode_id=episode_data.episode_id)

        mean_reward      = float(rewards.mean())
        reward_std       = float(rewards.std()) if len(rewards) > 1 else 0.0
        reward_stability = -reward_std
        length_score     = float(episode_data.length) / self._max_episode_length

        primary_score = (
            self._reward_w    * mean_reward
            + self._stability_w * reward_stability
            + self._length_w    * length_score
        )
        metrics = {
            "primary_score":       primary_score,
            "mean_reward":         mean_reward,
            "total_reward":        float(rewards.sum()),
            "reward_std":          reward_std,
            "reward_stability":    reward_stability,
            "episode_length":      episode_data.length,
            "episode_length_score": length_score,
        }
        metrics.update({k: float(v) for k, v in train_diagnostics.items()
                         if isinstance(v, (int, float))})
        return EvalMetrics(primary_score=primary_score, metrics=metrics,
                           episode_id=episode_data.episode_id)
