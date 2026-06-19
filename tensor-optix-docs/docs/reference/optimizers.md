# tensor_optix.optimizers

Five concrete `BaseOptimizer` implementations. All operate in normalized `[0, 1]` parameter space.

## SPSAOptimizer

```python
class SPSAOptimizer:
    """
    SPSA (Simultaneous Perturbation Stochastic Approximation) optimizer.

    Reference: Spall, J.C. (1992). "Multivariate Stochastic Approximation
    Using a Simultaneous Perturbation Gradient Approximation."

    All N bounded params are updated in exactly 2 episodes regardless of N,
    compared to BackoffOptimizer's 2N episodes (round-robin).

    Algorithm (all operations in normalized [0,1] param space):

    Episode 1 - PLUS probe:
        Sample Δ ∈ {-1, +1}^N  (Rademacher vector)
        Apply x⁺ = clip(x + c·Δ, 0, 1)  where c = perturbation_scale
        Record f⁺ (score after episode)

    Episode 2 - MINUS probe:
        Apply x⁻ = clip(x - c·Δ, 0, 1)
        Record f⁻ (score after episode)

    COMMIT (after episode 2):
        ĝᵢ = (f⁺ - f⁻) / (2c·Δᵢ)   for each param i
        x_new_i = clip(x_i + α·ĝᵢ, 0, 1)
        θ_new_i = denormalize(x_new_i, lo_i, hi_i)

    where c = perturbation_scale (default 0.05), α = learning_rate (default 0.1).

    The Rademacher distribution minimises the variance of the gradient
    estimator for a given c (Spall 1992, Theorem 1).

    is_probing returns True during the plus and minus probe episodes so
    LoopController suppresses degradation detection.
    """
```

Constructor accepts `param_bounds: dict[str, (lo, hi)]` and `log_params: set[str]` (params normalized in log space - use for learning rates and anything spanning orders of magnitude). On improvement, probes shrink (avoid disrupting active learning); on plateau, probes widen (explore more aggressively).

## MomentumOptimizer

```python
class MomentumOptimizer:
    """
    Adam-momentum finite-difference optimizer for hyperparameters.

    Gradient estimation - finite difference (probe/commit cycle):
        δᵢ = probe_scale × |θᵢ|
        gᵢ = (score_probe − score_base) / δᵢ

    Adam update in normalized [0,1] parameter space:
        mᵢ ← β₁·mᵢ + (1−β₁)·gᵢ               (momentum)
        vᵢ ← β₂·vᵢ + (1−β₂)·gᵢ²              (RMSProp)
        m̂ᵢ = mᵢ / (1 − β₁ᵗ), v̂ᵢ = vᵢ / (1 − β₂ᵗ)  (bias correction)
        xᵢ ← clip(xᵢ + α · m̂ᵢ/√(v̂ᵢ+ε), 0, 1)

    Why Adam over plain finite-difference (BackoffOptimizer):
      - Momentum smooths noisy stochastic gradient estimates across episodes
      - Adaptive rates give consistent-gradient params larger steps
      - Exponential decay tracks a non-stationary landscape
      - Bias correction gives unbiased estimates from episode 1
    """

    def __init__(
        self, param_bounds, alpha=0.05, beta1=0.9, beta2=0.999,
        eps=1e-8, probe_scale=0.05, min_delta=1e-8,
    ):
        """
        Args:
            alpha: Step size in normalized [0,1] space (0.05 = max 5% of range per update).
            beta1: Momentum decay (0.9 = gradient EMA carries 90% of previous estimate).
            beta2: RMSProp decay (0.999 = long variance history, stable adaptive rates).
            probe_scale: Finite-difference step δ = probe_scale × |θ|.
            min_delta: Absolute minimum δ (prevents divide-by-zero when θ≈0).
        """
```

## BackoffOptimizer

```python
class BackoffOptimizer:
    """
    Staggered two-phase finite difference optimizer.

    Each bounded param gets its own independent probe/commit cycle, cycled
    round-robin. With N bounded params, a full cycle takes 2N episodes
    (probe + commit per param).

    Phase 1 - PROBE: x + δ, record base_score.
    Phase 2 - COMMIT: gradient = (probe_score - base_score) / δ
        gradient > 0 → keep probe value
        gradient < 0 → apply x - δ (reflected step)
        gradient ≈ 0 → keep current value

    On improvement: increase perturbation_scale. On plateau: increase further,
    reset cycle.
    """
```

All probing happens in normalized `[0,1]` space - without normalization, `perturbation_scale * |θ|` collapses to `min_delta` for small-magnitude params.

## PBTOptimizer

```python
class PBTOptimizer:
    """
    Pseudo Population-Based Training for single-agent use.

    Maintains a history of (HyperparamSet, primary_score) pairs as a virtual
    population (FIFO, last history_size entries, default 50).

    Exploit condition:
        if current_score < percentile(history_scores, 20):
            best_params = params from top 20% of history (by score)
            new_params = perturb(best_params, scale=small)
    Explore condition:
        else: new_params = perturb(current_params, scale=medium)

    Perturbation - two modes per parameter:

    Linear (default):
        new_val = clip(θ + uniform(-δ, +δ), low, high),  δ = scale*(high-low)

    Log-scale (for log_scale_params, e.g. learning_rate):
        new_val = clip(θ * exp(uniform(-δ_log, +δ_log)), low, high)
        δ_log = scale * log(high / low)
        Equal probability mass per decade - correct for params spanning
        orders of magnitude (Jaderberg et al. 2017, PBT).
    """
```

## AdaptiveOptimizer

```python
class AdaptiveOptimizer:
    """
    Meta-optimizer that routes between SPSA, Momentum, Backoff, and PBT
    based on two mathematically grounded signals.

    Signal 1 - Lag-1 autocorrelation (ρ):
        ρ = Pearson_Corr(scores[t-1], scores[t]) over the recent window.
        ρ > +autocorr_threshold  → smooth landscape  → Momentum
        ρ < -autocorr_threshold  → oscillating       → Backoff (sign-only)
        |ρ| ≤ threshold           → i.i.d.            → SPSA

    Signal 2 - Relative performance gap (Δ):
        Δ = (current_score − historical_best) / |historical_best|  (always ≤ 0)
        Δ < -gap_threshold → bad region of hyperparameter space → PBT

    Routing priority (first match wins):
        1. Δ < -gap_threshold         → PBT      (escape bad region first)
        2. ρ > +autocorr_threshold    → Momentum (amplify smooth trend)
        3. ρ < -autocorr_threshold    → Backoff  (tame oscillation)
        4. otherwise                  → SPSA     (default, balanced)

    Hysteresis: the active optimizer is held for at least switch_patience
    consecutive evals before any switch - a single unlucky episode cannot
    cause a regime change.

    All four sub-optimizers receive on_improvement/on_plateau callbacks
    regardless of which is active, so their internal state stays warm -
    switching is seamless with no cold-start penalty.
    """

    @property
    def active_optimizer(self) -> str:
        """Name of the currently active sub-optimizer."""
```

See [Tune hyperparameters online](../guides/hyperparameter-optimization.md) for usage guidance and a routing-condition summary table.
