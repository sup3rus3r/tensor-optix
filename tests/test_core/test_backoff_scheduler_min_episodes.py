"""
Tests for min_episodes_before_dormant in BackoffScheduler.
"""
import pytest
from tensor_optix.core.backoff_scheduler import BackoffScheduler
from tensor_optix.core.types import LoopState


def make_scheduler(**kwargs):
    defaults = dict(
        base_interval=1,
        backoff_factor=2.0,
        max_interval_episodes=100,
        plateau_threshold=3,
        dormant_threshold=6,
        degradation_threshold=0.95,
    )
    defaults.update(kwargs)
    return BackoffScheduler(**defaults)


# ── total_episodes tracking ───────────────────────────────────────────────────

def test_total_episodes_starts_at_zero():
    s = make_scheduler()
    assert s.total_episodes == 0


def test_total_episodes_increments_on_improvement():
    s = make_scheduler()
    s.record_improvement(1.0)
    s.record_improvement(2.0)
    assert s.total_episodes == 2


def test_total_episodes_increments_on_non_improvement():
    s = make_scheduler()
    s.record_non_improvement()
    s.record_non_improvement()
    assert s.total_episodes == 2


def test_total_episodes_mixed():
    s = make_scheduler()
    s.record_improvement(1.0)
    s.record_non_improvement()
    s.record_improvement(2.0)
    s.record_non_improvement()
    assert s.total_episodes == 4


# ── min_episodes_before_dormant = 0 (default) ────────────────────────────────

def test_default_no_guard_goes_dormant_at_threshold():
    """Without guard, DORMANT triggers at dormant_threshold."""
    s = make_scheduler(dormant_threshold=6)
    for _ in range(6):
        s.record_non_improvement()
    assert s.current_state == LoopState.DORMANT


# ── min_episodes_before_dormant guard ────────────────────────────────────────

def test_guard_prevents_dormant_before_min_episodes():
    """
    dormant_threshold consecutive non-improvements reached, but
    total_episodes < min_episodes_before_dormant → stays COOLING.
    """
    s = make_scheduler(
        plateau_threshold=3,
        dormant_threshold=6,
        min_episodes_before_dormant=20,
    )
    # Hit dormant_threshold (6) but only 6 total episodes
    for _ in range(6):
        s.record_non_improvement()
    # Dormant threshold exceeded but guard blocks it
    assert s.current_state == LoopState.COOLING
    assert s.total_episodes == 6


def test_guard_allows_dormant_after_min_episodes():
    """
    After enough total episodes, DORMANT is declared normally.
    """
    s = make_scheduler(
        plateau_threshold=3,
        dormant_threshold=6,
        min_episodes_before_dormant=10,
    )
    # Record some improvements to rack up episode count
    for _ in range(5):
        s.record_improvement(float(_))
    # Now record enough non-improvements to hit dormant_threshold
    for _ in range(6):
        s.record_non_improvement()
    # total_episodes = 5 + 6 = 11 >= 10, so DORMANT is allowed
    assert s.current_state == LoopState.DORMANT
    assert s.total_episodes == 11


def test_guard_exact_boundary():
    """Guard triggers dormant exactly when total_episodes == min_episodes."""
    s = make_scheduler(
        plateau_threshold=3,
        dormant_threshold=6,
        min_episodes_before_dormant=6,
    )
    # 6 non-improvements → total_episodes=6, min=6 → DORMANT allowed
    for _ in range(6):
        s.record_non_improvement()
    assert s.current_state == LoopState.DORMANT


def test_guard_one_below_boundary():
    """total_episodes == min - 1 → DORMANT still blocked."""
    s = make_scheduler(
        plateau_threshold=3,
        dormant_threshold=6,
        min_episodes_before_dormant=10,
    )
    # 3 improvements + 6 non-improvements = 9 total (< 10)
    for _ in range(3):
        s.record_improvement(float(_))
    for _ in range(6):
        s.record_non_improvement()
    assert s.current_state != LoopState.DORMANT
    assert s.total_episodes == 9


def test_improvement_resets_non_improvement_counter_but_keeps_total():
    """
    An improvement resets consecutive_non_improvements and state back to ACTIVE,
    but total_episodes continues accumulating.
    """
    s = make_scheduler(
        plateau_threshold=3,
        dormant_threshold=6,
        min_episodes_before_dormant=20,
    )
    for _ in range(5):
        s.record_non_improvement()
    assert s.current_state == LoopState.COOLING
    assert s.total_episodes == 5

    s.record_improvement(99.0)
    assert s.current_state == LoopState.ACTIVE
    assert s.consecutive_non_improvements == 0
    assert s.total_episodes == 6   # total keeps counting


def test_degradation_does_not_reset_total_episodes():
    """record_degradation() must not reset total_episodes."""
    s = make_scheduler(dormant_threshold=6, min_episodes_before_dormant=20)
    s.record_improvement(100.0)
    for _ in range(6):
        s.record_non_improvement()
    total_before = s.total_episodes
    s.record_degradation()
    assert s.total_episodes == total_before
