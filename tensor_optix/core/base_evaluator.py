from abc import ABC, abstractmethod
from .types import EpisodeData, EvalMetrics


class BaseEvaluator(ABC):
    """
    Scores a completed episode.

    Core only cares about one thing: EvalMetrics.primary_score — a scalar
    where higher is always better. What that score represents is up to the user:
        - CartPole: mean episode reward
        - Robotics: task success rate
        - Trading: Sharpe ratio or risk-adjusted return
        - Custom domain: whatever "good" means there

    TFEvaluator ships as a sensible default for standard RL setups.
    Users should subclass BaseEvaluator for any non-trivial domain.
    """

    @abstractmethod
    def score(self, episode_data: EpisodeData, train_diagnostics: dict) -> EvalMetrics:
        """
        Compute evaluation metrics for a completed episode.

        Args:
            episode_data: Raw interaction data from the episode.
            train_diagnostics: Output from agent.learn() — loss, entropy, etc.
                               May be an empty dict.

        Returns:
            EvalMetrics with primary_score (higher = better) and full metrics dict.
        """

    def compare(self, candidate: EvalMetrics, baseline: EvalMetrics) -> bool:
        """
        Returns True if candidate is better than baseline.
        Default uses EvalMetrics.beats() with zero margin.
        Override for custom comparison logic (multi-objective, margin threshold, etc.).
        """
        return candidate.beats(baseline)
