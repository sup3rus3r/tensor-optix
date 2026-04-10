import pytest
import numpy as np
from tensor_optix.adapters.tensorflow.tf_evaluator import TFEvaluator
from tensor_optix.core.types import EpisodeData


@pytest.fixture
def episode():
    return EpisodeData(
        observations=np.zeros((10, 4)),
        actions=np.zeros(10),
        rewards=[1.0] * 10,
        terminated=[False] * 9 + [True],
        truncated=[False] * 10,
        infos=[{}] * 10,
        episode_id=0,
    )


def test_default_score_is_float(episode):
    ev = TFEvaluator()
    metrics = ev.score(episode, {})
    assert isinstance(metrics.primary_score, float)


def test_default_score_higher_for_better_rewards():
    ev = TFEvaluator()
    good = EpisodeData(
        observations=np.zeros((5, 4)), actions=np.zeros(5),
        rewards=[10.0] * 5, terminated=[False] * 4 + [True],
        truncated=[False] * 5, infos=[{}] * 5, episode_id=0,
    )
    bad = EpisodeData(
        observations=np.zeros((5, 4)), actions=np.zeros(5),
        rewards=[1.0] * 5, terminated=[False] * 4 + [True],
        truncated=[False] * 5, infos=[{}] * 5, episode_id=1,
    )
    good_score = ev.score(good, {}).primary_score
    bad_score = ev.score(bad, {}).primary_score
    assert good_score > bad_score


def test_metrics_dict_contains_expected_keys(episode):
    ev = TFEvaluator()
    metrics = ev.score(episode, {"loss": 0.5})
    assert "mean_reward" in metrics.metrics
    assert "total_reward" in metrics.metrics
    assert "reward_std" in metrics.metrics
    assert "loss" in metrics.metrics


def test_custom_primary_score_fn(episode):
    ev = TFEvaluator(primary_score_fn=lambda ep, _: 999.0)
    metrics = ev.score(episode, {})
    assert metrics.primary_score == 999.0


def test_compare_uses_primary_score(episode):
    ev = TFEvaluator()
    from tensor_optix.core.types import EvalMetrics
    a = EvalMetrics(primary_score=10.0, metrics={}, episode_id=0)
    b = EvalMetrics(primary_score=5.0, metrics={}, episode_id=1)
    assert ev.compare(a, b)
    assert not ev.compare(b, a)
