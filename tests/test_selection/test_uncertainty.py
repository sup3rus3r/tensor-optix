import statistics

import pytest

from tensor_optix.selection.uncertainty import score_interval


def test_score_interval_single_value():
    interval = score_interval([5.0])
    assert interval.mean == 5.0
    assert interval.std == 0.0
    assert interval.n == 1
    assert interval.lo95 == interval.hi95 == 5.0


def test_score_interval_multiple_values():
    scores = [1.0, 2.0, 3.0, 4.0, 5.0]
    interval = score_interval(scores)
    assert interval.mean == statistics.mean(scores)
    assert interval.std == statistics.stdev(scores)
    assert interval.lo95 < interval.mean < interval.hi95


def test_score_interval_empty_raises():
    with pytest.raises(ValueError):
        score_interval([])
