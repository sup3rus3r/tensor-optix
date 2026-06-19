# tensor_optix.core.backoff_scheduler

## BackoffScheduler

```python
class BackoffScheduler:
    """
    Controls adaptation interval and state transitions.

    Improvement and degradation are detected via **linear trend** over the
    recent score window, not point-to-point comparison.

    Why trend instead of point:
        Point comparison (smoothed > best) fires on every local maximum and
        misses the actual direction of learning. A single unlucky eval (390→10)
        looks like a collapse even if the surrounding 8 evals are rising.
        A linear fit over the last N scores is robust to single-episode spikes
        and gives the loop a directional signal:

            slope > +floor_per_step  →  improving
            |slope| < floor_per_step →  stuck
            slope < -floor_per_step  →  degrading

        where floor_per_step = adaptive_floor / trend_window.

    Adaptive floor auto-scales to env reward range (noise_k × std of recent
    scores), capped at 50% of |best_score| to prevent early-chaos poisoning.

    State transitions:
        ACTIVE  → COOLING  : after plateau_threshold consecutive non-improving evals
        COOLING → DORMANT  : after dormant_threshold consecutive non-improving evals
        any     → ACTIVE   : on genuine improvement trend or degradation
    """
```

### Constructor

```python
def __init__(
    self,
    base_interval: int = 1,
    backoff_factor: float = 2.0,
    max_interval_episodes: int = 100,
    plateau_threshold: int = 5,
    dormant_threshold: int = 20,
    degradation_threshold: float = 0.95,
    min_degradation_drop: float = 1e-4,
    min_episodes_before_dormant: int = 0,
    min_episodes_before_degradation: int = 5,
    noise_k: float = 2.0,
    score_window: int = 20,
    trend_window: int = 8,
):
    """
    Args:
        noise_k:           Adaptive floor multiplier. floor = noise_k × std(scores).
        score_window:      Rolling window size for std / trend computation.
        trend_window:      Number of most-recent scores used for slope fit.
                           Must be ≤ score_window. Default 8 - enough to be
                           robust to single-episode spikes while responding
                           quickly to genuine trend changes.
        min_degradation_drop: Fallback floor before enough history exists.
        min_episodes_before_dormant:    Guard against premature DORMANT.
        min_episodes_before_degradation: Guard against early false positives.
    """
```

### Trend detection

```python
def is_improving(self) -> bool:
    """
    True when the recent score trend is meaningfully upward.

    Uses slope over the trend window. Falls back to checking whether the
    latest score beat the best (point comparison) when window is too short.

    Threshold: slope > adaptive_floor / trend_window
    """

def is_degrading(self) -> bool:
    """
    True when the recent score trend is meaningfully downward AND the
    current level is well below the best known score.

    Both conditions must hold simultaneously - a downward trend during
    exploration or after a SPSA probe is not degradation unless the
    absolute level has also dropped significantly.
    """

def is_converged(self, cv_threshold: float = 0.05, gap_threshold: float = 0.20) -> bool:
    """
    True when the policy has genuinely converged:
    flat trend AND low CV AND performing near its personal best.

    Three conditions (all must hold):
        |slope| < floor_per_step              flat - not improving or degrading
        cv < cv_threshold                     tightly clustered - not noisy
        |mean - best| / |best| < gap_threshold  near personal best - not just stuck

    The gap condition is what separates a stuck policy (flat + stable at a
    low level, well below its best) from a genuinely converged one (flat +
    stable near its peak). No external solve threshold is used - purely
    relative to the agent's own history.
    """

def check_degradation(self, score: float) -> bool:
    """
    Threshold-based degradation check called by loop_controller.

    Returns True when `score` has dropped significantly below the best
    known score (by more than (1 - degradation_threshold) × |best|).

    Works correctly for both positive and negative score regimes:
        positive best=100, threshold=0.95 → fires when score < 95
        negative best=-100, threshold=0.95 → fires when score < -105
    """
```

### State machine

```python
def record_improvement(self, score: float) -> None:
    """Called when a new best score is recorded. Resets backoff and state."""

def record_non_improvement(self) -> None:
    """Called when episode did not produce a new best. Advances backoff."""

def record_degradation(self) -> None:
    """
    Called when degradation is detected.

    Resets to ACTIVE without resetting the interval - resetting to
    base_interval=1 would cause the optimizer to fire every episode,
    cascading into repeated degradation and destabilising on-policy training.
    """

def record_restart(self) -> None:
    """
    Called after DORMANT fires and PolicyManager has acted.
    Gives the new policy variant a clean slate.
    Does NOT reset best_score - the new variant must beat the existing best.
    """

def record_score(self, score: float) -> None:
    """Append score to window without triggering state change (off-policy path)."""

def should_adapt(self, episode_count: int) -> bool: ...
```

### Properties

```python
@property
def current_state(self) -> LoopState: ...
@property
def current_interval(self) -> int: ...
@property
def consecutive_non_improvements(self) -> int: ...
@property
def best_score(self) -> float | None: ...
@property
def total_episodes(self) -> int: ...
```
