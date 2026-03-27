from .types import LoopState


class BackoffScheduler:
    """
    Controls adaptation interval and state transitions.

    The scheduler is the nervous system of the loop. It decides:
    - How long to wait between adaptation attempts
    - When to transition between ACTIVE / COOLING / DORMANT / WATCHDOG
    - When degradation triggers re-activation

    Backoff strategy: exponential with configurable base and cap.
    - Each consecutive non-improvement multiplies interval by backoff_factor
    - Interval is capped at max_interval_episodes
    - On new improvement: interval resets to base_interval

    State transitions:
    - ACTIVE       → COOLING  : after plateau_threshold consecutive non-improvements
    - COOLING      → DORMANT  : after dormant_threshold consecutive non-improvements
    - DORMANT      → WATCHDOG : automatically (DORMANT is the quiet watchdog state)
    - any state    → ACTIVE   : on degradation or improvement

    Degradation detection:
    - If primary_score drops below (best_score * degradation_threshold),
      transition back to ACTIVE from any state.
    """

    def __init__(
        self,
        base_interval: int = 1,
        backoff_factor: float = 2.0,
        max_interval_episodes: int = 100,
        plateau_threshold: int = 5,
        dormant_threshold: int = 20,
        degradation_threshold: float = 0.95,
    ):
        self._base_interval = base_interval
        self._backoff_factor = backoff_factor
        self._max_interval = max_interval_episodes
        self._plateau_threshold = plateau_threshold
        self._dormant_threshold = dormant_threshold
        self._degradation_threshold = degradation_threshold

        self._state = LoopState.ACTIVE
        self._consecutive_non_improvements = 0
        self._current_interval = base_interval
        self._best_score: float | None = None

    def record_improvement(self, score: float) -> None:
        """Call when a new best score is achieved. Resets backoff and state."""
        self._best_score = score
        self._consecutive_non_improvements = 0
        self._current_interval = self._base_interval
        self._state = LoopState.ACTIVE

    def record_non_improvement(self) -> None:
        """Call when episode did not beat best. Advances backoff and may change state."""
        self._consecutive_non_improvements += 1
        self._current_interval = min(
            int(self._current_interval * self._backoff_factor),
            self._max_interval,
        )
        if self._consecutive_non_improvements >= self._dormant_threshold:
            self._state = LoopState.DORMANT
        elif self._consecutive_non_improvements >= self._plateau_threshold:
            self._state = LoopState.COOLING

    def record_degradation(self) -> None:
        """Call when watchdog detects score drop below threshold. Resets to ACTIVE."""
        self._consecutive_non_improvements = 0
        self._current_interval = self._base_interval
        self._state = LoopState.ACTIVE

    def check_degradation(self, score: float) -> bool:
        """
        Returns True if score represents degradation relative to best known score.
        Does NOT record anything — caller decides whether to act.

        Uses an absolute drop threshold to handle negative scores correctly:
            degraded if: score < best - abs(best) * (1 - threshold)
        This works for both positive scores (e.g. 100 → drop of 5)
        and negative scores (e.g. -100 → drop of 5, meaning -105).
        """
        if self._best_score is None:
            return False
        allowed_drop = abs(self._best_score) * (1.0 - self._degradation_threshold)
        return score < self._best_score - allowed_drop

    def should_adapt(self, episode_count: int) -> bool:
        """
        Returns True if this episode should trigger an eval+tune cycle.
        Uses episode_count modulo current_interval.
        Always adapts on episode 0 (cold start).
        """
        if episode_count == 0:
            return True
        return episode_count % self._current_interval == 0

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
