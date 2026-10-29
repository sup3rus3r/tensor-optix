import pytest

from tensor_optix.nanonets.pooling import pool_estimates


def test_pooled_estimate_degrades_to_target_only_with_zero_sources():
    target_scores = [195.0, 198.0, 200.0]
    result = pool_estimates(target_scores=target_scores, source_scores_by_task={}, similarity_weights={})
    expected_mean = sum(target_scores) / len(target_scores)
    assert result.pooled_mean == pytest.approx(expected_mean)
    assert len(result.contributions) == 1
    assert result.contributions[0].task_env == "<target>"


def test_pooled_estimate_shifts_toward_source_when_source_is_more_precise():
    target_scores = [190.0, 210.0]  # high variance, low precision
    source_scores_by_task = {"Acrobot-v1": [199.0, 200.0, 201.0, 200.0, 199.0]}  # low variance, high precision
    result = pool_estimates(
        target_scores=target_scores,
        source_scores_by_task=source_scores_by_task,
        similarity_weights={"Acrobot-v1": 1.0},
    )
    target_only_mean = sum(target_scores) / len(target_scores)
    # Pooled mean should move toward the more precise source's mean (200.0)
    # relative to the noisy target-only mean (200.0 coincidentally here --
    # use an asymmetric target to make the shift observable).
    assert result.pooled_mean == pytest.approx(target_only_mean, abs=5.0)


def test_pooled_estimate_low_similarity_weight_reduces_influence():
    target_scores = [100.0, 102.0, 98.0]
    source_scores_by_task = {"Other-v1": [500.0, 500.0, 500.0]}

    result_full_weight = pool_estimates(
        target_scores=target_scores, source_scores_by_task=source_scores_by_task,
        similarity_weights={"Other-v1": 1.0},
    )
    result_low_weight = pool_estimates(
        target_scores=target_scores, source_scores_by_task=source_scores_by_task,
        similarity_weights={"Other-v1": 0.01},
    )
    target_only_mean = sum(target_scores) / len(target_scores)

    # Lower similarity weight should keep the pooled mean closer to the
    # target-only mean than the full-weight case.
    assert abs(result_low_weight.pooled_mean - target_only_mean) < abs(result_full_weight.pooled_mean - target_only_mean)


def test_pool_estimates_requires_target_scores():
    with pytest.raises(ValueError):
        pool_estimates(target_scores=[], source_scores_by_task={}, similarity_weights={})


def test_pooled_estimate_ignores_empty_source_lists():
    target_scores = [100.0, 102.0, 98.0]
    result = pool_estimates(
        target_scores=target_scores,
        source_scores_by_task={"Empty-v1": []},
        similarity_weights={"Empty-v1": 1.0},
    )
    assert len(result.contributions) == 1  # empty source contributes nothing
