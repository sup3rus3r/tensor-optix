import numpy as np
from tensor_optix.core.base_evaluator import BaseEvaluator
from tensor_optix.core.types import EpisodeData, EvalMetrics


class TFEvaluator(BaseEvaluator):
    """
    Default evaluator for TensorFlow-based RL agents.

    primary_score = mean episode return across completed episodes in the window.

    A window typically contains multiple episode fragments (the env resets
    mid-window). Using mean episode return gives the loop a stable, meaningful
    signal that improves monotonically with policy quality — unlike per-step
    mean reward which is confounded by episode length and always noisy, and
    unlike mean_reward - std_reward which actively penalises high-reward
    terminal events (e.g. +100 landing bonus in LunarLander).

    Falls back to mean per-step reward when no episode completes in the window
    (e.g. very early training or very long episodes).

    Usage:
        evaluator = TFEvaluator()
        # Custom scorer:
        evaluator = TFEvaluator(primary_score_fn=lambda ep, diag: sum(ep.rewards))
    """

    def __init__(self, primary_score_fn=None):
        self._primary_score_fn = primary_score_fn

    def score(self, episode_data: EpisodeData, train_diagnostics: dict) -> EvalMetrics:
        rewards = np.array(episode_data.rewards, dtype=np.float32)
        # Use terminated OR truncated as episode boundary (same rationale as TorchEvaluator).
        dones = episode_data.dones

        if self._primary_score_fn is not None:
            primary = float(self._primary_score_fn(episode_data, train_diagnostics))
        else:
            primary = self._mean_episode_return(rewards, dones)

        metrics = {
            "primary_score":  primary,
            "total_reward":   float(rewards.sum()),
            "mean_reward":    float(rewards.mean()),
            "reward_std":     float(rewards.std()) if len(rewards) > 1 else 0.0,
            "episode_length": episode_data.length,
        }
        metrics.update({k: float(v) for k, v in train_diagnostics.items()
                        if isinstance(v, (int, float))})
        return EvalMetrics(primary_score=primary, metrics=metrics,
                           episode_id=episode_data.episode_id)

    @staticmethod
    def _mean_episode_return(rewards: np.ndarray, dones: list) -> float:
        """
        Compute mean return of episodes that completed in this window.
        dones should be terminated OR truncated — both signal an episode
        boundary. Falls back to mean per-step reward only if the window
        contains a single incomplete episode (very long envs, early training).
        """
        episode_returns = []
        current = 0.0
        for r, done in zip(rewards, dones):
            current += float(r)
            if done:
                episode_returns.append(current)
                current = 0.0
        if episode_returns:
            return float(np.mean(episode_returns))
        return float(rewards.mean())
