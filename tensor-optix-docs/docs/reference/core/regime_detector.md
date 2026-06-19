# tensor_optix.core.regime_detector

## RegimeDetector

```python
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
    ): ...

    def detect(self, metrics_history: List[EvalMetrics]) -> str:
        """
        Classify the current regime from recent EvalMetrics history.

        Returns "trending", "ranging", or "volatile".
        Returns "ranging" if fewer than 3 data points are available.
        """
```

### Classification order

1. Fewer than 3 recent scores → `"ranging"`.
2. `detrended_cv > volatility_threshold` → `"volatile"`.
3. `normalized_slope > trend_threshold` → `"trending"`.
4. Otherwise → `"ranging"`.

Used by `NeuromodulatorSignal` (in the [neuroevo](../neuroevo/neuromodulator.md) subsystem) to modulate Hebbian learning rate, entropy coefficient, and topology grow/prune thresholds, and standalone with `PolicyManager.boost()`/`set_regime()` for non-neuroevo regime-conditional ensembles.
