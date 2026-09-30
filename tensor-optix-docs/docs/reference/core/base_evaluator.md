# tensor_optix.core.base_evaluator

## BaseEvaluator

```python
class BaseEvaluator(ABC):
    """
    Scores a completed episode.

    Core only cares about one thing: EvalMetrics.primary_score - a scalar
    where higher is always better. What that score represents is up to the user:
        - CartPole: mean episode reward
        - Robotics: task success rate
        - Trading: Sharpe ratio or risk-adjusted return
        - Custom domain: whatever "good" means there

    When a val_pipeline is configured, the loop calls score() for training
    and score_validation() for validation, then combine() to merge them into
    a single EvalMetrics whose primary_score drives all adaptation decisions.

    The adaptation signal is the correlation between train and val - not
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
            train_diagnostics: Output from agent.learn() - loss, entropy, etc.
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

        primary_score = val_score - out-of-sample performance drives all
        checkpoint, rollback, and spawn decisions.

        metrics includes both raw scores and the generalization_gap so that
        adaptive_noise_scale() and status() can surface the overfitting signal.

        Override to use a different combination formula:
            - min(train, val): conservative - both must be good
            - harmonic_mean: penalises imbalance
            - val - λ * gap: explicit overfitting penalty
        """

    def criteria(self, episode_data: EpisodeData, train_diagnostics: dict) -> Dict[str, float]:
        """
        Optional. Return named criteria in [0, 1] (or bool) describing
        distinct ways this episode could be "good" - e.g.
        {"reached_goal": 1.0, "avoided_collision": 1.0, "energy_budget": 0.3}.

        Used by composite_score() when criteria_mode is "manual" or "both"
        (see LoopController). Default: {} - no manual criteria; a policy
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
        one axis from one that is robustly good across many - this widens
        the measurement rather than just smoothing it.

        satisfied_fraction, depending on criteria_mode:
          "none"   (default): 1.0 always - composite_score == primary_score,
                   i.e. no behavior change unless a mode is opted into.
          "manual": mean of criteria() - {} (default, unoverridden) → 1.0.
          "auto":   breadth_typicality() of this episode's reward-magnitude
                    concentration relative to this run's OWN history (never
                    an absolute threshold - see reward_shape.py for why
                    that matters for sparse/terminal-reward domains).
          "both":   geometric mean of the two - both must hold.

        composite = primary_score - (1 - satisfied_fraction**k) * |primary_score|
        Subtractive rather than multiplicative on purpose: primary_score's
        sign is entirely domain-defined (higher is always better, but that
        says nothing about sign - cost-style domains have negative-is-good
        conventions). A plain multiplicative penalty makes a negative score
        LESS negative under penalty, which is backwards. This penalizes by a
        magnitude-relative amount instead, so composite <= primary_score
        always, regardless of sign, with equality iff fully satisfied.
        """

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
        CheckpointConfirmationTracker.evaluate_candidate() - requiring the
        improvement to be corroborated rather than trusting a single sample.
        Falls back to plain candidate.beats(baseline) when called without a
        tracker (e.g. direct/standalone use) - unchanged from before.

        Override for custom comparison logic (multi-objective, margin
        threshold, etc.) - the loop only requires a bool return.
        """
```

`combine()`'s default implementation sets `primary_score = val.primary_score` and stores both raw scores plus `generalization_gap = train.primary_score - val.primary_score` in `metrics`.

See also: `TFEvaluator` and `TorchEvaluator` in [Algorithms](../algorithms.md) for the shipped default (mean episode return, with a per-step-reward fallback when no episode completes within the window).

## Measuring "good" as more than one number

`primary_score` is a single scalar, and a single scalar can't distinguish a
policy that got lucky on one narrow axis from one that is robustly good
across many. Left unaddressed, a lucky early-exploration episode (a reward
spike concentrated on one step) can become a permanent, unbeatable "best"
checkpoint even once training produces a genuinely, durably better policy
whose raw score is lower.

`LoopController` fixes this in two complementary layers, both **off by
default** (`criteria_mode="none"` reproduces the exact legacy behavior):

1. **Composite scoring** (`composite_score()` above) widens the
   measurement itself. Two ways to feed it, controlled by
   `LoopController(criteria_mode=...)`:
   - `"manual"` - override `criteria()` with a domain-specific rubric
     (e.g. `{"reached_goal": 1.0, "avoided_hazard": 1.0}`).
   - `"auto"` - zero code required. tensor-optix measures how concentrated
     vs. distributed each episode's reward was (via the Gini coefficient of
     per-step `|reward|`, sign-agnostic so it works for both bonus-style and
     cost-style reward conventions), and compares that shape against *this
     run's own historical distribution* - never a fixed threshold, so it
     never penalizes domains where every legitimate win is naturally
     concentrated (e.g. a sparse terminal bonus).
   - `"both"` - geometric mean of the two.
2. **Confirmation** (`compare()` above, via `CheckpointConfirmationTracker`)
   is a statistical evidence gate, not a magnitude gate: it doesn't care how
   *big* an improvement is, only whether there's been enough evidence -
   either the run's own measured noise is small enough that one sample is
   already convincing, or the improvement has been corroborated across
   `checkpoint_confirm_window` consecutive evals - before trusting it.

See [LoopController](loop_controller.md) for the full config surface
(`criteria_mode`, `criteria_k`, `checkpoint_confirm_window`,
`checkpoint_noise_k`) and the [training guide](../../guides/train-rl-agent.md#avoiding-unicorn-checkpoints)
for a worked example.
