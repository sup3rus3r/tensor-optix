import math

import pytest

from tensor_optix.nanonets.smalln import small_n_interval, t_critical_975


def _naive_margin(scores):
    n = len(scores)
    mean = sum(scores) / n
    if n > 1:
        variance = sum((s - mean) ** 2 for s in scores) / (n - 1)
        std = math.sqrt(variance)
    else:
        std = 0.0
    return 1.96 * std / math.sqrt(n) if n > 1 else 0.0


def test_small_n_interval_wider_than_naive_at_low_n():
    scores = [100.0, 150.0, 90.0, 180.0]  # n=4
    interval = small_n_interval(scores)
    naive_margin = _naive_margin(scores)
    t_margin = interval.hi95 - interval.mean
    assert t_margin > naive_margin


def test_small_n_interval_converges_to_z_approximation_at_high_n():
    scores = [float(i) for i in range(1, 41)]  # n=40, df=39 > table range
    interval = small_n_interval(scores)
    naive_margin = _naive_margin(scores)
    t_margin = interval.hi95 - interval.mean
    assert t_margin == pytest.approx(naive_margin, rel=1e-9)


def test_small_n_interval_single_value():
    interval = small_n_interval([5.0])
    assert interval.mean == 5.0
    assert interval.std == 0.0
    assert interval.lo95 == interval.hi95 == 5.0


def test_small_n_interval_empty_raises():
    with pytest.raises(ValueError):
        small_n_interval([])


def test_t_critical_975_known_values():
    assert t_critical_975(1) == pytest.approx(12.7062, abs=1e-3)
    assert t_critical_975(30) == pytest.approx(2.0423, abs=1e-3)


def test_t_critical_975_beyond_table_uses_z():
    assert t_critical_975(100) == 1.96


def test_t_critical_975_rejects_invalid_df():
    with pytest.raises(ValueError):
        t_critical_975(0)
