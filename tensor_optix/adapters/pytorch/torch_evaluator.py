import numpy as np
from typing import Callable, Optional
from tensor_optix.core.base_evaluator import BaseEvaluator
from tensor_optix.core.types import EpisodeData, EvalMetrics


class TorchEvaluator(BaseEvaluator):
    """
    Default evaluator for PyTorch-based RL agents.

    primary_score = mean episode return (MER) across completed episodes in
    the training window.

    Noise reduction is handled automatically by LoopController's
    score_smoothing: the rolling mean of the last N primary_scores is used
    for checkpoint selection and convergence detection — not a single noisy
    sample. This is algorithm-agnostic and requires no per-env tuning.

    Falls back to mean per-step reward when no episode completes in the window
    (e.g. very early training or very long episodes).

    Usage:
        evaluator = TorchEvaluator()
        # Custom scorer:
        evaluator = TorchEvaluator(primary_score_fn=lambda ep, diag: sum(ep.rewards))
    """

    def __init__(self, primary_score_fn: Optional[Callable] = None):
        self._primary_score_fn = primary_score_fn

    def score(self, episode_data: EpisodeData, train_diagnostics: dict) -> EvalMetrics:
        rewards = np.array(episode_data.rewards, dtype=np.float32)
        # Use terminated OR truncated as episode boundary.
        # terminated-only misses time-limit endings: a window with only
        # truncated episodes returns rewards.mean() ≈ 0 (LunarLander ~0.1),
        # which looks like catastrophic degradation and fires false positives.
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
