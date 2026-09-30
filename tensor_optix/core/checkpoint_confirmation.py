"""
Statistical evidence gate for checkpoint acceptance.

This does NOT distinguish "big" jumps from "small" ones — a big jump isn't
inherently suspicious, and composite scoring (reward_shape.py / BaseEvaluator
.criteria()) is what actually measures whether an improvement is narrow
(one lucky axis) or broad (genuinely better). This tracker's only job is the
one BackoffScheduler already does for improvement/degradation *state*
detection, applied here to checkpoint *acceptance*: don't trust a single
measurement — even an already-composite one — until there's enough evidence
that it isn't just this run's own noise.

Deliberately NOT a BackoffScheduler instance/subclass: BackoffScheduler's
window is scoped to backoff/DORMANT state and gets cleared by
record_restart(), which must never wipe checkpoint-acceptance history.
"""
from collections import deque
from typing import Deque, Optional, Tuple

import numpy as np


class CheckpointConfirmationTracker:
    """
    Tracks recent checkpoint-candidate scores and decides whether a
    candidate has been sufficiently corroborated to become the new best.

    Philosophy (mirrors BackoffScheduler's adaptive floor + trend idiom):
    - No history yet → first checkpoint always accepted.
    - Low noise (adaptive floor is small relative to the run's own score
      scale) → a single sample clearly above best is already strong
      evidence; accept immediately, same as BackoffScheduler.is_improving()'s
      point-comparison fallback under sparse history.
    - Otherwise → require `confirm_window` CONSECUTIVE candidates to all
      exceed best (each individually, via this same evaluate_candidate call)
      before crowning the most recent one — so one noisy spike that
      immediately regresses next episode never gets confirmed. The streak is
      measured strictly relative to `best_score`, not to a self-referential
      "pending level," so noisy oscillation that never actually clears best
      can never accumulate a false streak.
    """

    def __init__(
        self,
        noise_k: float = 2.0,
        confirm_window: int = 3,
        score_window: int = 20,
        min_samples_for_floor: int = 5,
        min_degradation_drop: float = 1e-4,
    ):
        self._noise_k = noise_k
        self._confirm_window = max(1, confirm_window)
        self._min_samples_for_floor = min_samples_for_floor
        self._min_degradation_drop = min_degradation_drop
        self._window: Deque[float] = deque(maxlen=max(score_window, self._confirm_window))
        self._above_best_streak: int = 0
        self.last_reason: Optional[str] = None

    def _adaptive_floor(self, best_score: Optional[float]) -> float:
        if len(self._window) < self._min_samples_for_floor:
            return self._min_degradation_drop
        floor = self._noise_k * float(np.std(list(self._window)))
        if best_score is not None and abs(best_score) > 1e-8:
            floor = min(floor, 0.5 * abs(best_score))
        return max(floor, self._min_degradation_drop)

    def observe(self, score: float) -> None:
        """Record a checkpoint-candidate score. Call on every eval episode,
        regardless of whether it becomes the new best — this keeps the
        adaptive floor's noise estimate current."""
        self._window.append(score)

    def evaluate_candidate(
        self, candidate_score: float, best_score: Optional[float]
    ) -> Tuple[bool, str]:
        """
        Returns (accept, reason). Call once per eval episode where a best
        already exists (the caller handles the "no best yet" case itself,
        typically at cold start); when called with best_score=None this
        method still returns the correct first_checkpoint answer for
        standalone/direct use.

        reason is one of:
          "first_checkpoint"    — no best exists yet
          "confirmed_by_floor"  — noise floor is small enough that a single
                                  sample above best is already strong evidence
          "confirmed_by_streak" — confirm_window consecutive candidates have
                                  each individually exceeded best
          "below_best"          — candidate did not exceed best at all —
                                  this is a genuine magnitude rejection, not
                                  a confirmation-pending one
          "unconfirmed"         — candidate exceeds best but hasn't yet been
                                  corroborated — NOT a magnitude rejection;
                                  callers must not log this as "< best"

        `self.last_reason` mirrors the returned reason, for callers (e.g.
        LoopController's verbose/log output) that need to distinguish a
        genuine "below best" rejection from a merely "not yet confirmed"
        one after calling this indirectly through BaseEvaluator.compare().
        """
        if best_score is None:
            self._above_best_streak = 0
            self.last_reason = "first_checkpoint"
            return True, self.last_reason

        if candidate_score <= best_score + self._min_degradation_drop:
            self._above_best_streak = 0
            self.last_reason = "below_best"
            return False, self.last_reason

        floor = self._adaptive_floor(best_score)
        if candidate_score > best_score + floor and len(self._window) >= self._min_samples_for_floor:
            self._above_best_streak = 0
            self.last_reason = "confirmed_by_floor"
            return True, self.last_reason

        # Above best, but within this run's own noise floor — corroborate
        # over consecutive evals before trusting it.
        self._above_best_streak += 1
        if self._above_best_streak >= self._confirm_window:
            self._above_best_streak = 0
            self.last_reason = "confirmed_by_streak"
            return True, self.last_reason
        self.last_reason = "unconfirmed"
        return False, self.last_reason
