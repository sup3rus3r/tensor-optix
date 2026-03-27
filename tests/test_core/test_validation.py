import pytest
import numpy as np
from tensor_optix.core.types import EvalMetrics, EpisodeData
from tensor_optix.core.base_evaluator import BaseEvaluator
from tensor_optix.core.policy_manager import PolicyManager


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def make_metrics(primary, train=None, val=None, episode_id=0):
    m = {"score": primary}
    if train is not None:
        m["train_score"] = train
    if val is not None:
        m["val_score"] = val
    return EvalMetrics(primary_score=primary, metrics=m, episode_id=episode_id)


class DummyEvaluator(BaseEvaluator):
    def __init__(self, fixed_score=10.0):
        self._score = fixed_score

    def score(self, episode_data, train_diagnostics):
        return EvalMetrics(
            primary_score=self._score,
            metrics={"mean_reward": self._score},
            episode_id=episode_data.episode_id,
        )


# -----------------------------------------------------------------------
# EvalMetrics.generalization_gap
# -----------------------------------------------------------------------

def test_generalization_gap_present():
    m = make_metrics(8.0, train=10.0, val=8.0)
    assert m.generalization_gap == pytest.approx(2.0)


def test_generalization_gap_zero_when_equal():
    m = make_metrics(5.0, train=5.0, val=5.0)
    assert m.generalization_gap == pytest.approx(0.0)


def test_generalization_gap_none_without_val():
    m = make_metrics(5.0)
    assert m.generalization_gap is None


def test_generalization_gap_negative_means_val_exceeds_train():
    m = make_metrics(12.0, train=10.0, val=12.0)
    assert m.generalization_gap == pytest.approx(-2.0)


# -----------------------------------------------------------------------
# BaseEvaluator.combine()
# -----------------------------------------------------------------------

def test_combine_primary_score_is_val():
    ev = DummyEvaluator()
    train = EvalMetrics(primary_score=10.0, metrics={"mean_reward": 10.0}, episode_id=1)
    val = EvalMetrics(primary_score=6.0, metrics={"mean_reward": 6.0}, episode_id=1)
    combined = ev.combine(train, val)
    assert combined.primary_score == pytest.approx(6.0)


def test_combine_includes_gap():
    ev = DummyEvaluator()
    train = EvalMetrics(primary_score=10.0, metrics={}, episode_id=1)
    val = EvalMetrics(primary_score=7.0, metrics={}, episode_id=1)
    combined = ev.combine(train, val)
    assert combined.metrics["generalization_gap"] == pytest.approx(3.0)


def test_combine_includes_both_scores():
    ev = DummyEvaluator()
    train = EvalMetrics(primary_score=10.0, metrics={}, episode_id=1)
    val = EvalMetrics(primary_score=8.0, metrics={}, episode_id=1)
    combined = ev.combine(train, val)
    assert combined.metrics["train_score"] == pytest.approx(10.0)
    assert combined.metrics["val_score"] == pytest.approx(8.0)


def test_combine_preserves_episode_id():
    ev = DummyEvaluator()
    train = EvalMetrics(primary_score=5.0, metrics={}, episode_id=42)
    val = EvalMetrics(primary_score=5.0, metrics={}, episode_id=42)
    combined = ev.combine(train, val)
    assert combined.episode_id == 42


def test_score_validation_delegates_to_score():
    ev = DummyEvaluator(fixed_score=7.0)
    episode = EpisodeData(
        observations=np.zeros((3, 4)),
        actions=np.zeros(3),
        rewards=[1.0, 1.0, 1.0],
        terminated=[False, False, True],
        truncated=[False, False, False],
        infos=[{}, {}, {}],
        episode_id=0,
    )
    result = ev.score_validation(episode)
    assert result.primary_score == pytest.approx(7.0)


# -----------------------------------------------------------------------
# adaptive_noise_scale — val+correlation aware
# -----------------------------------------------------------------------

def _make_pm():
    pm = PolicyManager.__new__(PolicyManager)
    pm._registry = None
    pm._ensemble = []
    pm._score_history = {}
    pm._score_window = 10
    pm._max_spawns = None
    pm._max_ensemble_size = None
    pm._spawn_count = 0
    pm._prune_count = 0
    pm._current_regime = None
    return pm


def test_adaptive_noise_high_gap_increases_noise():
    pm = _make_pm()
    # Train improving, val flat → large gap → more noise than pure slope would give
    history = [
        make_metrics(v, train=10.0 + i, val=v, episode_id=i)
        for i, v in enumerate([5.0, 5.1, 5.0, 5.1, 5.0])
    ]
    scale_with_gap = pm.adaptive_noise_scale(history, min_scale=0.001, max_scale=0.1)

    # Without gap (val matches train, corr=1, gap=0)
    history_no_gap = [
        make_metrics(5.0 + i * 0.5, train=5.0 + i * 0.5, val=5.0 + i * 0.5, episode_id=i)
        for i in range(5)
    ]
    scale_no_gap = pm.adaptive_noise_scale(history_no_gap, min_scale=0.001, max_scale=0.1)

    assert scale_with_gap > scale_no_gap


def test_adaptive_noise_low_correlation_increases_noise():
    pm = _make_pm()
    # Train and val moving in opposite directions → low/negative correlation
    history = [
        make_metrics(val, train=10.0 + i, val=val, episode_id=i)
        for i, val in enumerate([10.0, 8.0, 10.0, 8.0, 10.0])
    ]
    scale_low_corr = pm.adaptive_noise_scale(history, min_scale=0.001, max_scale=0.1)

    # Perfectly correlated
    history_high_corr = [
        make_metrics(5.0 + i, train=5.0 + i, val=5.0 + i, episode_id=i)
        for i in range(5)
    ]
    scale_high_corr = pm.adaptive_noise_scale(history_high_corr, min_scale=0.001, max_scale=0.1)

    assert scale_low_corr > scale_high_corr


def test_adaptive_noise_stays_within_bounds_with_val():
    pm = _make_pm()
    history = [
        make_metrics(float(i), train=float(i) * 2, val=float(i), episode_id=i)
        for i in range(1, 8)
    ]
    scale = pm.adaptive_noise_scale(history, min_scale=0.001, max_scale=0.1)
    assert 0.001 <= scale <= 0.1


def test_adaptive_noise_without_val_uses_slope_only():
    pm = _make_pm()
    # No train/val keys in metrics — falls back to slope mode
    history = [
        EvalMetrics(primary_score=float(i), metrics={}, episode_id=i)
        for i in range(5)
    ]
    scale = pm.adaptive_noise_scale(history, min_scale=0.001, max_scale=0.1)
    assert 0.001 <= scale <= 0.1
