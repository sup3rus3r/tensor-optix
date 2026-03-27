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

    When a val_pipeline is configured, the loop calls score() for training
    and score_validation() for validation, then combine() to merge them into
    a single EvalMetrics whose primary_score drives all adaptation decisions.

    The adaptation signal is the correlation between train and val — not
    just training performance alone:
        - High val + low gap + high corr → genuinely learning → back off
        - High val + high gap + low corr → overfitting → explore more
        - Low val + low gap + high corr → genuine plateau → spawn

    TFEvaluator ships as a sensible default for standard RL setups.
    Users should subclass BaseEvaluator for any non-trivial domain.
    """

    @abstractmethod
    def score(self, episode_data: EpisodeData, train_diagnostics: dict) -> EvalMetrics:
        """
        Compute evaluation metrics for a completed training episode.

        Args:
            episode_data: Raw interaction data from the episode.
            train_diagnostics: Output from agent.learn() — loss, entropy, etc.
                               May be an empty dict.

        Returns:
            EvalMetrics with primary_score (higher = better) and full metrics dict.
        """

    def score_validation(self, episode_data: EpisodeData) -> EvalMetrics:
        """
        Score a validation episode. The agent acts but does NOT learn.

        Default: delegates to score() with empty diagnostics.
        Override for val-specific logic (e.g. different reward shaping,
        stricter termination conditions, held-out environment seeds).
        """
        return self.score(episode_data, {})

    def combine(self, train: EvalMetrics, val: EvalMetrics) -> EvalMetrics:
        """
        Merge train and val metrics into a single EvalMetrics.

        primary_score = val_score — out-of-sample performance drives all
        checkpoint, rollback, and spawn decisions.

        metrics includes both raw scores and the generalization_gap so that
        adaptive_noise_scale() and status() can surface the overfitting signal.

        Override to use a different combination formula:
            - min(train, val): conservative — both must be good
            - harmonic_mean: penalises imbalance
            - val - λ * gap: explicit overfitting penalty
        """
        gap = train.primary_score - val.primary_score
        combined_metrics = {
            **{f"train_{k}": v for k, v in train.metrics.items()},
            **{f"val_{k}": v for k, v in val.metrics.items()},
            "train_score": train.primary_score,
            "val_score": val.primary_score,
            "generalization_gap": gap,
        }
        return EvalMetrics(
            primary_score=val.primary_score,
            metrics=combined_metrics,
            episode_id=train.episode_id,
        )

    def compare(self, candidate: EvalMetrics, baseline: EvalMetrics) -> bool:
        """
        Returns True if candidate is better than baseline.
        Default uses EvalMetrics.beats() with zero margin.
        Override for custom comparison logic (multi-objective, margin threshold, etc.).
        """
        return candidate.beats(baseline)
