import time

from tensor_optix.scheduler.budget import BudgetTracker


def test_global_wall_clock_budget_stops_new_dispatch():
    tracker = BudgetTracker(max_wall_clock_s=0.05)
    assert tracker.global_budget_exhausted() is False
    time.sleep(0.07)
    assert tracker.global_budget_exhausted() is True


def test_global_budget_unbounded_when_none():
    tracker = BudgetTracker(max_wall_clock_s=None)
    time.sleep(0.01)
    assert tracker.global_budget_exhausted() is False


def test_per_trial_timeout_independent_of_global_budget():
    tracker = BudgetTracker(max_wall_clock_s=100.0, trial_timeout_s=0.05)
    tracker.start_trial(1)
    assert tracker.trial_timed_out(1) is False
    time.sleep(0.07)
    assert tracker.trial_timed_out(1) is True
    # Global budget is a much larger window and must remain unaffected.
    assert tracker.global_budget_exhausted() is False


def test_trial_timeout_unbounded_when_none():
    tracker = BudgetTracker(trial_timeout_s=None)
    tracker.start_trial(1)
    time.sleep(0.01)
    assert tracker.trial_timed_out(1) is False


def test_trial_timed_out_false_for_unknown_trial():
    tracker = BudgetTracker(trial_timeout_s=0.01)
    assert tracker.trial_timed_out(999) is False
