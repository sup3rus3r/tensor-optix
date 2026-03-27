import logging
from typing import List
import numpy as np

from .types import EvalMetrics

logger = logging.getLogger(__name__)


class RegimeDetector:
    """
    Classifies the current performance regime from EvalMetrics history.

    Regime labels:
    - "trending"  : scores consistently improving (positive normalized slope)
    - "ranging"   : scores stable with low volatility
    - "volatile"  : scores show high variance

    Uses coefficient of variation (CV = std/|mean|) for volatility and a
    linear slope normalized by the mean for trend direction. Both metrics
    are scale-independent, so they work across different scoring ranges.

    For domain-specific signals (e.g. Sharpe ratio, VIX, ATR percentile),
    subclass this and override detect().

    Parameters:
        volatility_threshold: CV above this → "volatile" (default 0.2 = 20%)
        trend_threshold: normalized slope above this → "trending" (default 0.05)
        window: number of recent EvalMetrics to consider (default 10)

    Usage:
        detector = RegimeDetector()
        regime = detector.detect(metrics_history)
        if regime == "volatile":
            pm.update_weights({2: 2.0})  # boost the volatile-regime agent
    """

    def __init__(
        self,
        volatility_threshold: float = 0.2,
        trend_threshold: float = 0.05,
        window: int = 10,
    ):
        self._vol_threshold = volatility_threshold
        self._trend_threshold = trend_threshold
        self._window = window

    def detect(self, metrics_history: List[EvalMetrics]) -> str:
        """
        Classify the current regime from recent EvalMetrics history.

        Returns "trending", "ranging", or "volatile".
        Returns "ranging" if fewer than 3 data points are available.
        """
        if not metrics_history:
            return "ranging"

        recent = metrics_history[-self._window :]
        if len(recent) < 3:
            return "ranging"

        scores = np.array([m.primary_score for m in recent], dtype=float)
        mean = float(np.mean(scores))

        if mean == 0.0:
            return "ranging"

        cv = float(np.std(scores)) / abs(mean)
        if cv > self._vol_threshold:
            logger.debug("RegimeDetector: volatile (CV=%.3f > threshold=%.3f)", cv, self._vol_threshold)
            return "volatile"

        x = np.arange(len(scores), dtype=float)
        slope = float(np.polyfit(x, scores, 1)[0])
        normalized_slope = slope / abs(mean)

        if normalized_slope > self._trend_threshold:
            logger.debug(
                "RegimeDetector: trending (normalized_slope=%.4f > threshold=%.4f)",
                normalized_slope,
                self._trend_threshold,
            )
            return "trending"

        logger.debug("RegimeDetector: ranging")
        return "ranging"
