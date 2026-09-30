from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Deque, Dict, Optional
import numpy as np
from .types import EpisodeData, EvalMetrics
from .reward_shape import episode_breadth, breadth_typicality

if TYPE_CHECKING:
    from .checkpoint_confirmation import CheckpointConfirmationTracker


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

    def criteria(self, episode_data: EpisodeData, train_diagnostics: dict) -> Dict[str, float]:
        """
        Optional. Return named criteria in [0, 1] (or bool) describing
        distinct ways this episode could be "good" — e.g.
        {"reached_goal": 1.0, "avoided_collision": 1.0, "energy_budget": 0.3}.

        Used by composite_score() when criteria_mode is "manual" or "both"
        (see LoopController). Default: {} — no manual criteria; a policy
        that satisfies only one of several things that matter for a task
        should not out-rank one that satisfies all of them just because a
        single-axis reward spike happened to be larger.
        """
        return {}

    def composite_score(
        self,
        metrics: EvalMetrics,
        episode_data: EpisodeData,
        train_diagnostics: dict,
        breadth_history: Deque[float],
        criteria_mode: str = "none",
        criteria_k: float = 2.0,
    ) -> float:
        """
        Adjusts primary_score by how narrowly or broadly it was earned.

        A single scalar score can't distinguish a policy that got lucky on
        one axis from one that is robustly good across many — this widens
        the measurement rather than just smoothing it.

        satisfied_fraction, depending on criteria_mode:
          "none"   (default): 1.0 always — composite_score == primary_score,
                   i.e. no behavior change unless a mode is opted into.
          "manual": mean of criteria() — {} (default, unoverridden) → 1.0.
          "auto":   breadth_typicality() of this episode's reward-magnitude
                    concentration relative to this run's OWN history (never
                    an absolute threshold — see reward_shape.py for why
                    that matters for sparse/terminal-reward domains).
          "both":   geometric mean of the two — both must hold.

        composite = primary_score - (1 - satisfied_fraction**criteria_k) * |primary_score|
        Conjunctive: narrow wins are crushed toward/away-from zero (whichever
        direction is WORSE), broad wins are unchanged. Subtractive rather
        than multiplicative on purpose: primary_score's sign is entirely
        domain-defined (higher is always better, per this class's contract,
        but that says nothing about sign) — a plain multiplicative penalty
        (primary * fraction) makes a negative score LESS negative — i.e.
        better — under penalty, which is backwards. Penalizing by
        subtracting a magnitude-relative amount is worse regardless of sign:
        composite <= primary_score always, with equality iff fully satisfied.
        """
        primary = metrics.primary_score
        if criteria_mode == "none":
            return primary

        manual_fraction = None
        if criteria_mode in ("manual", "both"):
            crit = self.criteria(episode_data, train_diagnostics)
            manual_fraction = float(sum(crit.values()) / len(crit)) if crit else 1.0

        auto_fraction = None
        if criteria_mode in ("auto", "both"):
            breadths = episode_breadth(episode_data.rewards, episode_data.dones)
            current_breadth = breadths[-1] if breadths else 1.0
            auto_fraction = breadth_typicality(current_breadth, breadth_history)

        if criteria_mode == "manual":
            satisfied_fraction = manual_fraction
        elif criteria_mode == "auto":
            satisfied_fraction = auto_fraction
        else:  # "both"
            satisfied_fraction = float(np.sqrt(max(manual_fraction, 0.0) * max(auto_fraction, 0.0)))

        penalty_factor = 1.0 - satisfied_fraction ** criteria_k
        return primary - penalty_factor * abs(primary)

    def compare(
        self,
        candidate: EvalMetrics,
        baseline: EvalMetrics,
        confirmation: Optional["CheckpointConfirmationTracker"] = None,
    ) -> bool:
        """
        Returns True if candidate should replace baseline as the best
        checkpoint.

        Default: when `confirmation` is supplied (LoopController always
        supplies one), gates the comparison through
        CheckpointConfirmationTracker.evaluate_candidate() — requiring the
        improvement to be corroborated rather than trusting a single sample.
        Falls back to plain candidate.beats(baseline) when called without a
        tracker (e.g. direct/standalone use, existing tests) — unchanged
        from before.

        Override for custom comparison logic (multi-objective, margin
        threshold, etc.) — the loop only requires a bool return.
        """
        if confirmation is None:
            return candidate.beats(baseline)
        accepted, _reason = confirmation.evaluate_candidate(
            candidate.primary_score, baseline.primary_score
        )
        return accepted
