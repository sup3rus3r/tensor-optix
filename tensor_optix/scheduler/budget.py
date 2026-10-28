"""Global and per-trial wall-clock budget tracking for the concurrent scheduler.

Pure bookkeeping, no multiprocessing. The scheduler's parent process is the
sole owner of a BudgetTracker instance.
"""

import time
from typing import Dict, Optional


class BudgetTracker:
    def __init__(self, *, max_wall_clock_s: Optional[float] = None, trial_timeout_s: Optional[float] = None):
        self._max_wall_clock_s = max_wall_clock_s
        self._trial_timeout_s = trial_timeout_s
        self._start = time.monotonic()
        self._trial_started_at: Dict[int, float] = {}

    def start_trial(self, trial_number: int) -> None:
        self._trial_started_at[trial_number] = time.monotonic()

    def global_budget_exhausted(self) -> bool:
        if self._max_wall_clock_s is None:
            return False
        return (time.monotonic() - self._start) >= self._max_wall_clock_s

    def trial_timed_out(self, trial_number: int) -> bool:
        if self._trial_timeout_s is None:
            return False
        started = self._trial_started_at.get(trial_number)
        if started is None:
            return False
        return (time.monotonic() - started) >= self._trial_timeout_s

    def elapsed(self) -> float:
        return time.monotonic() - self._start
