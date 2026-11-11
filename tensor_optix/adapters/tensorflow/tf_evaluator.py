import numpy as np
from tensor_optix.core.base_evaluator import BaseEvaluator
from tensor_optix.core.reward_shape import completed_episode_returns, split_episode_returns
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
