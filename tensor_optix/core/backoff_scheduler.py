from collections import deque
import numpy as np
from .types import LoopState


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
                               Must be ≤ score_window. Default 8 — enough to be
                               robust to single-episode spikes while responding
                               quickly to genuine trend changes.
            min_degradation_drop: Fallback floor before enough history exists.
            min_episodes_before_dormant:    Guard against premature DORMANT.
            min_episodes_before_degradation: Guard against early false positives.
        """
        self._base_interval = base_interval
        self._backoff_factor = backoff_factor
        self._max_interval = max_interval_episodes
        self._plateau_threshold = plateau_threshold
        self._dormant_threshold = dormant_threshold
        self._degradation_threshold = degradation_threshold
        self._min_degradation_drop = min_degradation_drop
        self._min_episodes_before_dormant = min_episodes_before_dormant
        self._min_episodes_before_degradation = min_episodes_before_degradation
        self._noise_k = noise_k
        self._trend_window = max(3, trend_window)
        self._score_window: deque = deque(maxlen=max(score_window, self._trend_window))
        self._dormant_fired: bool = False

        self._state = LoopState.ACTIVE
        self._consecutive_non_improvements = 0
        self._current_interval = base_interval
        self._best_score: float | None = None
        self._total_episodes: int = 0

    # ------------------------------------------------------------------
    # Adaptive floor
    # ------------------------------------------------------------------

    def _adaptive_floor(self) -> float:
        """
        Adaptive absolute floor for improvement margin and degradation detection.
        Returns noise_k × std(recent scores), capped at 50% of |best_score|.
        Falls back to min_degradation_drop until at least 5 scores exist.
        """
        if len(self._score_window) < 5:
            return self._min_degradation_drop
        floor = self._noise_k * float(np.std(list(self._score_window)))
        if self._best_score is not None and abs(self._best_score) > 1e-8:
            floor = min(floor, 0.5 * abs(self._best_score))
        return max(floor, self._min_degradation_drop)

    # ------------------------------------------------------------------
    # Trend detection
    # ------------------------------------------------------------------

    def _slope(self) -> float | None:
        """
        Linear regression slope over the most recent trend_window scores.
        Returns None when fewer than 3 scores exist (not enough to fit a line).
        Units: score-change per eval step.
        """
        n = min(len(self._score_window), self._trend_window)
        if n < 3:
            return None
        recent = list(self._score_window)[-n:]
        x = np.arange(n, dtype=np.float64)
        y = np.array(recent, dtype=np.float64)
        # Simple linear regression via closed form
        x_mean = x.mean()
        y_mean = y.mean()
        denom = ((x - x_mean) ** 2).sum()
        if denom < 1e-12:
            return 0.0
        return float(((x - x_mean) * (y - y_mean)).sum() / denom)

    def is_improving(self) -> bool:
        """
        True when the recent score trend is meaningfully upward.

        Uses slope over the trend window. Falls back to checking whether the
        latest score beat the best (point comparison) when window is too short.

        Threshold: slope > adaptive_floor / trend_window
        (floor_per_step scales the absolute floor into per-eval units)
        """
        slope = self._slope()
        if slope is None:
            # Not enough history — fall back to point comparison
            if len(self._score_window) == 0:
                return False
            latest = self._score_window[-1]
            if self._best_score is None:
                return True
            return latest > self._best_score + self._adaptive_floor()
        floor_per_step = self._adaptive_floor() / self._trend_window
        return slope > floor_per_step

    def is_degrading(self) -> bool:
        """
        True when the recent score trend is meaningfully downward AND the
        current level is well below the best known score.

        Both conditions must hold simultaneously — a downward trend during
        exploration or after a SPSA probe is not degradation unless the
        absolute level has also dropped significantly.

        Conditions:
            slope < -floor_per_step           (trend is falling)
            latest < best_score - allowed_drop (absolute level is low)
        """
        if self._best_score is None:
            return False
        if self._total_episodes < self._min_episodes_before_degradation:
            return False

        slope = self._slope()
        if slope is None:
            return False  # not enough history to call degradation

        floor_per_step = self._adaptive_floor() / self._trend_window
        if slope >= -floor_per_step:
            return False  # trend is flat or rising — not degrading

        # Trend is falling. Check absolute level.
        latest = self._score_window[-1]
        relative_drop = abs(self._best_score) * (1.0 - self._degradation_threshold)
        allowed_drop = max(relative_drop, self._adaptive_floor())
        return latest < self._best_score - allowed_drop

    # ------------------------------------------------------------------
    # State machine
    # ------------------------------------------------------------------

    def record_improvement(self, score: float) -> None:
        """Called when a new best score is recorded. Resets backoff and state."""
        self._score_window.append(score)
        if self._best_score is None or len(self._score_window) >= 5:
            self._best_score = score
        self._consecutive_non_improvements = 0
        self._current_interval = self._base_interval
        self._state = LoopState.ACTIVE
        self._dormant_fired = False
        self._total_episodes += 1

    def record_non_improvement(self) -> None:
        """Called when episode did not produce a new best. Advances backoff."""
        self._consecutive_non_improvements += 1
        self._total_episodes += 1
        self._current_interval = min(
            int(self._current_interval * self._backoff_factor),
            self._max_interval,
        )
        if self._consecutive_non_improvements >= self._dormant_threshold:
            if self._total_episodes >= self._min_episodes_before_dormant:
                self._state = LoopState.DORMANT
        elif self._consecutive_non_improvements >= self._plateau_threshold:
            self._state = LoopState.COOLING

    def record_degradation(self) -> None:
        """
        Called when degradation is detected.

        Resets to ACTIVE without resetting the interval — resetting to
        base_interval=1 would cause the optimizer to fire every episode,
        cascading into repeated degradation and destabilising on-policy training.

        Does NOT increment total_episodes: the episode was already counted by
        the preceding record_improvement() / record_non_improvement() call.
        """
        self._state = LoopState.ACTIVE

    def record_restart(self) -> None:
        """
        Called after DORMANT fires and PolicyManager has acted.
        Gives the new policy variant a clean slate.
        Does NOT reset best_score — the new variant must beat the existing best.
        """
        self._consecutive_non_improvements = 0
        self._current_interval = self._base_interval
        self._state = LoopState.ACTIVE
        self._dormant_fired = False
        self._score_window.clear()

    def record_score(self, score: float) -> None:
        """Append score to window without triggering state change (off-policy path)."""
        self._score_window.append(score)

    def is_converged(self, cv_threshold: float = 0.05, gap_threshold: float = 0.20) -> bool:
        """
        True when the policy has genuinely converged:
        flat trend AND low CV AND performing near its personal best.

        Three conditions (all must hold):
            |slope| < floor_per_step              flat — not improving or degrading
            cv < cv_threshold                     tightly clustered — not noisy
            |mean - best| / |best| < gap_threshold  near personal best — not just stuck

        The gap condition is what separates a stuck policy (flat + stable at a
        low level, well below its best) from a genuinely converged one (flat +
        stable near its peak). No external solve threshold is used — purely
        relative to the agent's own history.
        """
        slope = self._slope()
        if slope is None:
            return False
        floor_per_step = self._adaptive_floor() / self._trend_window
        if abs(slope) >= floor_per_step:
            return False  # still moving — not converged
        if len(self._score_window) < 5:
            return False
        scores = list(self._score_window)
        mean = float(np.mean(scores))
        std = float(np.std(scores))
        cv = std / max(abs(mean), 1e-8)
        if cv >= cv_threshold:
            return False  # too noisy — stuck, not converged
        # Gap check: current mean must be near personal best.
        # Prevents firing when the policy is stuck well below its historical peak.
        if self._best_score is not None and abs(self._best_score) > 1e-8:
            gap = abs(mean - self._best_score) / abs(self._best_score)
            if gap > gap_threshold:
                return False  # performing well below personal best — stuck, not converged
        return True

    def check_degradation(self, score: float) -> bool:
        """
        Threshold-based degradation check called by loop_controller.

        Returns True when `score` has dropped significantly below the best
        known score (by more than (1 - degradation_threshold) × |best|).

        Works correctly for both positive and negative score regimes:
            positive best=100, threshold=0.95 → fires when score < 95
            negative best=-100, threshold=0.95 → fires when score < -105
        """
        if self._best_score is None:
            return False
        allowed_drop = abs(self._best_score) * (1.0 - self._degradation_threshold)
        return score < self._best_score - allowed_drop

    def should_adapt(self, episode_count: int) -> bool:
        if episode_count == 0:
            return True
        return episode_count % self._current_interval == 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def current_state(self) -> LoopState:
        return self._state

    @property
    def current_interval(self) -> int:
        return self._current_interval

    @property
    def consecutive_non_improvements(self) -> int:
        return self._consecutive_non_improvements

    @property
    def best_score(self) -> float | None:
        return self._best_score

    @property
    def total_episodes(self) -> int:
        return self._total_episodes
