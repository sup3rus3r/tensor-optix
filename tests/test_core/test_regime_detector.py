import pytest
from tensor_optix.core.regime_detector import RegimeDetector
from tensor_optix.core.types import EvalMetrics


def make_metrics(scores):
    return [
        EvalMetrics(primary_score=s, metrics={}, episode_id=i)
        for i, s in enumerate(scores)
    ]


# -----------------------------------------------------------------------
# Edge cases
# -----------------------------------------------------------------------

def test_empty_history_returns_ranging():
    d = RegimeDetector()
    assert d.detect([]) == "ranging"


def test_fewer_than_three_points_returns_ranging():
    d = RegimeDetector()
    assert d.detect(make_metrics([1.0, 2.0])) == "ranging"


# -----------------------------------------------------------------------
# Volatile regime
# -----------------------------------------------------------------------

def test_volatile_when_high_variance():
    d = RegimeDetector(volatility_threshold=0.2)
    # CV = std/mean — these scores have very high variance relative to mean
    scores = [1.0, 100.0, 2.0, 90.0, 3.0, 80.0]
    assert d.detect(make_metrics(scores)) == "volatile"


def test_not_volatile_when_low_variance():
    d = RegimeDetector(volatility_threshold=0.2)
    # Scores very close together — low CV
    scores = [10.0, 10.1, 9.9, 10.05, 10.02]
    result = d.detect(make_metrics(scores))
    assert result != "volatile"


# -----------------------------------------------------------------------
# Trending regime
# -----------------------------------------------------------------------

def test_trending_when_consistently_improving():
    d = RegimeDetector(volatility_threshold=0.2, trend_threshold=0.05)
    # Steadily increasing scores with low variance
    scores = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
    assert d.detect(make_metrics(scores)) == "trending"


def test_not_trending_when_flat():
    d = RegimeDetector(volatility_threshold=0.2, trend_threshold=0.05)
    scores = [10.0, 10.0, 10.0, 10.0, 10.0]
    assert d.detect(make_metrics(scores)) == "ranging"


# -----------------------------------------------------------------------
# Ranging regime
# -----------------------------------------------------------------------

def test_ranging_when_stable_flat():
    d = RegimeDetector(volatility_threshold=0.2, trend_threshold=0.05)
    scores = [10.0, 10.1, 9.95, 10.05, 10.0, 9.98]
    assert d.detect(make_metrics(scores)) == "ranging"


# -----------------------------------------------------------------------
# Window parameter
# -----------------------------------------------------------------------

def test_only_last_window_points_considered():
    d = RegimeDetector(volatility_threshold=0.2, trend_threshold=0.05, window=3)
    # First 10 points are flat/ranging; last 3 are trending
    old = [10.0] * 10
    recent = [20.0, 25.0, 30.0]
    result = d.detect(make_metrics(old + recent))
    assert result == "trending"


# -----------------------------------------------------------------------
# Custom subclass
# -----------------------------------------------------------------------

def test_subclass_can_override_detect():
    class AlwaysVolatile(RegimeDetector):
        def detect(self, metrics_history):
            return "volatile"

    d = AlwaysVolatile()
    assert d.detect(make_metrics([10.0, 10.0, 10.0])) == "volatile"
