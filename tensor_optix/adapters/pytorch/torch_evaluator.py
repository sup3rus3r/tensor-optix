import numpy as np
from typing import Callable, Optional
from tensor_optix.core.base_evaluator import BaseEvaluator
from tensor_optix.core.reward_shape import completed_episode_returns, split_episode_returns
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
            n_envs = int((episode_data.metadata or {}).get("n_envs", 1))
            # Full returns of the episodes that finished in this window
            # (tracked across windows by the pipeline when it supports it).
            episode_returns = completed_episode_returns(episode_data, n_envs)
            primary = (
                float(np.mean(episode_returns)) if episode_returns
                else float(rewards.mean()) if len(rewards) else 0.0
            )

        metrics = {
            "primary_score":  primary,
            "total_reward":   float(rewards.sum()),
            "mean_reward":    float(rewards.mean()),
            "reward_std":     float(rewards.std()) if len(rewards) > 1 else 0.0,
            "episode_length": episode_data.length,
        }
        if self._primary_score_fn is None:
            # How many finished episodes the score is based on. 0 means the
            # score above is only the per-step fallback: the training loops
            # use this to avoid treating such a window as an evaluation.
            metrics["episodes_completed"] = len(episode_returns)
        metrics.update({k: float(v) for k, v in train_diagnostics.items()
                        if isinstance(v, (int, float))})
        return EvalMetrics(primary_score=primary, metrics=metrics,
                           episode_id=episode_data.episode_id)

    @staticmethod
    def _mean_episode_return(rewards: np.ndarray, dones: list, n_envs: int = 1) -> float:
        """
        Compute mean return of episodes that completed in this window.
        dones should be terminated OR truncated — both signal an episode
        boundary. Falls back to mean per-step reward only if the window
        contains a single incomplete episode (very long envs, early training).
        """
        # Per-env accumulation: a VectorBatchPipeline window interleaves
        # n_envs envs, and each episode's return must only sum its own env.
        episode_returns = [
            float(np.sum(seg)) for seg in split_episode_returns(rewards, dones, n_envs)
        ]
        if episode_returns:
            return float(np.mean(episode_returns))
        return float(rewards.mean())
