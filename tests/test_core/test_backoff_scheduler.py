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


def test_initial_state_is_active():
    s = make_scheduler()
    assert s.current_state == LoopState.ACTIVE


def test_improvement_resets_to_active():
    s = make_scheduler()
    s.record_non_improvement()
    s.record_non_improvement()
    s.record_non_improvement()
    assert s.current_state == LoopState.COOLING
    s.record_improvement(score=10.0)
    assert s.current_state == LoopState.ACTIVE
    assert s.current_interval == 1
    assert s.consecutive_non_improvements == 0


def test_plateau_threshold_transitions_to_cooling():
    s = make_scheduler(plateau_threshold=3)
    for _ in range(3):
        s.record_non_improvement()
    assert s.current_state == LoopState.COOLING


def test_dormant_threshold_transitions_to_dormant():
    s = make_scheduler(dormant_threshold=6)
    for _ in range(6):
        s.record_non_improvement()
    assert s.current_state == LoopState.DORMANT


def test_interval_grows_exponentially():
    s = make_scheduler(base_interval=1, backoff_factor=2.0, max_interval_episodes=100)
    s.record_non_improvement()
    assert s.current_interval == 2
    s.record_non_improvement()
    assert s.current_interval == 4
    s.record_non_improvement()
    assert s.current_interval == 8


def test_interval_capped_at_max():
    s = make_scheduler(base_interval=1, backoff_factor=2.0, max_interval_episodes=5)
    for _ in range(10):
        s.record_non_improvement()
    assert s.current_interval == 5


def test_degradation_resets_to_active():
    s = make_scheduler()
    s.record_improvement(score=100.0)
    for _ in range(6):
        s.record_non_improvement()
    assert s.current_state == LoopState.DORMANT
    s.record_degradation()
    assert s.current_state == LoopState.ACTIVE
    # Interval is intentionally NOT reset by record_degradation — resetting to 1
    # would fire the optimizer every episode and cause cascading degradation.
    # record_restart() resets it (called after PolicyManager acts on DORMANT).


def test_check_degradation():
    s = make_scheduler(degradation_threshold=0.95)
    # Positive scores: best=100, allowed_drop=5, threshold=95
    s.record_improvement(score=100.0)
    assert s.check_degradation(94.0)
    assert not s.check_degradation(96.0)
    assert not s.check_degradation(100.0)

def test_check_degradation_negative_scores():
    s = make_scheduler(degradation_threshold=0.95)
    # Negative scores: best=-100, allowed_drop=5, threshold=-105
    s.record_improvement(score=-100.0)
    assert s.check_degradation(-106.0)   # worse by 6
    assert not s.check_degradation(-104.0)  # worse by only 4
    assert not s.check_degradation(-100.0)  # same


def test_check_degradation_with_no_best():
    s = make_scheduler()
    assert not s.check_degradation(0.0)


def test_should_adapt_episode_zero():
    s = make_scheduler(base_interval=5)
    assert s.should_adapt(0)


def test_should_adapt_respects_interval():
    s = make_scheduler(base_interval=1, backoff_factor=2.0, max_interval_episodes=100)
    s.record_non_improvement()  # interval → 2
    assert not s.should_adapt(1)
    assert s.should_adapt(2)
    assert not s.should_adapt(3)
    assert s.should_adapt(4)
