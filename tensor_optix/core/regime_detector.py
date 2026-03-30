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
    - "volatile"  : scores show high variance around the trend

    Volatility is measured as the *detrended* coefficient of variation:
        1. Fit a linear trend to the score window via least squares.
        2. Compute residuals = scores - trend.
        3. CV_detrended = std(residuals) / (|mean(scores)| + ε)

    This is strictly better than raw CV because:
    - Raw CV conflates genuine noise with a strong upward/downward trend.
      A steadily declining score has low raw CV but is clearly not "ranging".
    - Detrended CV measures noise *around* the trend, independent of direction.
    - A single polyfit call produces both the slope (for trend detection) and
      the residuals (for volatility), so there is no redundant computation.

    For domain-specific signals (e.g. Sharpe ratio, VIX, ATR percentile),
    subclass this and override detect().

    Parameters:
        volatility_threshold: detrended CV above this → "volatile" (default 0.15)
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
        volatility_threshold: float = 0.15,
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

        recent = metrics_history[-self._window:]
        if len(recent) < 3:
            return "ranging"

        scores = np.array([m.primary_score for m in recent], dtype=float)
        mean = float(np.mean(scores))
        eps = 1e-8

        x = np.arange(len(scores), dtype=float)
        coeffs = np.polyfit(x, scores, 1)
        slope = float(coeffs[0])
        trend_line = np.polyval(coeffs, x)

        # Detrended residuals — noise that is not explained by the linear trend
        residuals = scores - trend_line
        cv_detrended = float(np.std(residuals)) / (abs(mean) + eps)

        if cv_detrended > self._vol_threshold:
            logger.debug(
                "RegimeDetector: volatile (CV_detrended=%.3f > threshold=%.3f)",
                cv_detrended, self._vol_threshold,
            )
            return "volatile"

        normalized_slope = slope / (abs(mean) + eps)
        if normalized_slope > self._trend_threshold:
            logger.debug(
                "RegimeDetector: trending (normalized_slope=%.4f > threshold=%.4f)",
                normalized_slope, self._trend_threshold,
            )
            return "trending"

        logger.debug(
            "RegimeDetector: ranging (CV_detrended=%.3f slope=%.4f)",
            cv_detrended, normalized_slope,
        )
        return "ranging"
