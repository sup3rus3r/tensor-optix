"""
Discovery budgets: how much time, data and compute discovery may spend.

``Budget`` is what the user asks for; ``BudgetLedger`` tracks what has been
spent and predicts whether the next stage fits. When a stage would overrun,
discovery shrinks it in a fixed, documented order (fewer configurations, then
fewer repeats, then fewer rows) and records each cut as a BUDGET finding --
it never silently runs over or silently does less.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class Budget:
    """
    Args:
        time_s:     wall-clock seconds for the whole discovery run.
        max_rows:   cap on rows used for model fitting (ML). None = decided by
                    the sample-size rule and ``time_s``.
        max_fits:   cap on individual model fits (ML) / short trials (RL).
        env_steps:  cap on environment steps across all probes and trials (RL).
    """

    time_s: float = 600.0
    max_rows: Optional[int] = None
    max_fits: Optional[int] = None
    env_steps: Optional[int] = None

    def __post_init__(self):
        if not (self.time_s > 0 and math.isfinite(self.time_s)):
            raise ValueError(f"time_s must be a positive finite number, got {self.time_s!r}")
        for name in ("max_rows", "max_fits", "env_steps"):
            v = getattr(self, name)
            if v is not None and (not isinstance(v, int) or v <= 0):
                raise ValueError(f"{name} must be a positive int or None, got {v!r}")

    @classmethod
    def default_for_rows(cls, n_rows: int) -> "Budget":
        """About 60 s at 10k rows, growing with log size, capped at 15 minutes."""
        n = max(int(n_rows), 1)
        t = 60.0 * (1.0 + max(0.0, math.log10(n / 10_000.0)) * 2.0)
        return cls(time_s=float(min(max(t, 30.0), 900.0)))

    def to_dict(self) -> Dict[str, Any]:
        return {"time_s": self.time_s, "max_rows": self.max_rows,
                "max_fits": self.max_fits, "env_steps": self.env_steps}


@dataclass
class BudgetLedger:
    """Running account of a discovery run's spend."""

    budget: Budget
    started_at: float = field(default_factory=time.perf_counter)
    fits: int = 0
    env_steps: int = 0
    rows_fitted: int = 0
    cuts: list = field(default_factory=list)    # (stage, what, reason)

    def elapsed(self) -> float:
        return time.perf_counter() - self.started_at

    def remaining_s(self) -> float:
        return max(0.0, self.budget.time_s - self.elapsed())

    def exhausted(self) -> bool:
        if self.remaining_s() <= 0.0:
            return True
        if self.budget.max_fits is not None and self.fits >= self.budget.max_fits:
            return True
        if self.budget.env_steps is not None and self.env_steps >= self.budget.env_steps:
            return True
        return False

    def fits_left(self) -> Optional[int]:
        if self.budget.max_fits is None:
            return None
        return max(0, self.budget.max_fits - self.fits)

    def record_fit(self, n_rows: int = 0) -> None:
        self.fits += 1
        self.rows_fitted += int(n_rows)

    def record_steps(self, n: int) -> None:
        self.env_steps += int(n)

    def record_cut(self, stage: str, what: str, reason: str) -> None:
        self.cuts.append((stage, what, reason))

    def to_dict(self) -> Dict[str, Any]:
        return {"elapsed_s": round(self.elapsed(), 3), "fits": self.fits,
                "env_steps": self.env_steps, "rows_fitted": self.rows_fitted,
                "cuts": [{"stage": s, "what": w, "reason": r} for s, w, r in self.cuts]}


def predict_cost(t_ref: float, n_ref: int, n: int, exponent: float = 1.0) -> float:
    """
    Extrapolate a measured fit time ``t_ref`` at ``n_ref`` rows to ``n`` rows
    with ``t(n) = t_ref * (n / n_ref) ** exponent``; the exponent is clamped to
    [1, 2] (no estimator here is sub-linear or worse than quadratic in rows).
    """
    if n_ref <= 0 or t_ref < 0:
        raise ValueError("predict_cost needs n_ref > 0 and t_ref >= 0")
    a = min(max(float(exponent), 1.0), 2.0)
    return float(t_ref) * (max(n, 1) / n_ref) ** a
