import logging
from enum import Enum, auto
from typing import Dict, List, Optional
import numpy as np

from .types import EvalMetrics

logger = logging.getLogger(__name__)


class MetaAction(Enum):
    """Actions the MetaController can instruct the system to take on DORMANT."""
    NO_OP = auto()   # system is healthy — do nothing
    SPAWN = auto()   # exploration needed — clone and mutate a new variant
    PRUNE = auto()   # ensemble too large or most overfit agent should be removed
    STOP  = auto()   # spawn budget exhausted or convergence confirmed — halt


class MetaController:
    """
    Decides what the system should do when it reaches DORMANT state.

    Observes three signals derived from EvalMetrics history and pm.status():

    1. Generalization gap level: mean (train - val) / |val|
       Large gap → model is overfitting → PRUNE

    2. Generalization gap slope: is the gap widening?
       A model where train improves but val stagnates is actively overfitting
       even if the current gap is below the level threshold. Slope of the
       normalized gap series catches this early.

       This replaces the former Pearson correlation signal. Pearson measured
       whether train and val move *together in shape*, not whether they
       diverge in *level*. A perfectly correlated (r=1.0) pair like
       train=[0.9,0.91,0.92] vs val=[0.3,0.31,0.32] is catastrophically
       overfit. Gap slope is the correct overfitting-progression signal.

    3. Normalized val slope (improvement rate)
       Flat or declining val performance → genuine plateau → SPAWN

    Priority: gap level → gap slope → improvement rate. If the budget
    is exhausted, STOP is returned regardless of other signals.

    This is a rule-based controller. It implements no learning of its own.
    The interface is intentionally minimal so it can be swapped for a learned
    policy (e.g. an RL agent whose observation is pm.status() + metrics
    features and whose action space is MetaAction) without any API change.

    Parameters:
        gap_threshold:       normalized gap above which PRUNE fires (default 0.2)
        gap_slope_threshold: normalized gap slope above which PRUNE fires (default 0.02)
                             i.e. gap is widening by >2% of |val| per episode
        improvement_threshold: normalized val slope below which SPAWN fires (default 0.01)
        window:              number of recent EvalMetrics to consider (default 10)
    """

    def __init__(
        self,
        gap_threshold: float = 0.2,
        gap_slope_threshold: float = 0.02,
        improvement_threshold: float = 0.01,
        window: int = 10,
    ):
        self._gap_threshold = gap_threshold
        self._gap_slope_threshold = gap_slope_threshold
        self._improvement_threshold = improvement_threshold
        self._window = window

    def decide(self, metrics_history: List[EvalMetrics], pm_status: dict) -> MetaAction:
        """
        Return a MetaAction based on current system state.

        metrics_history: full history of EvalMetrics from the loop
        pm_status: output of PolicyManager.status()
        """
        if pm_status.get("budget_exhausted", False):
            logger.info("MetaController: budget exhausted → STOP")
            return MetaAction.STOP

        if len(metrics_history) < 3:
            return MetaAction.NO_OP

        recent = metrics_history[-self._window:]

        # --- Signal 1: generalization gap level ---
        gap = self._normalized_gap(recent)
        if gap > self._gap_threshold:
            logger.info(
                "MetaController: gap=%.3f > %.3f → PRUNE (overfitting level)",
                gap, self._gap_threshold,
            )
            return MetaAction.PRUNE

        # --- Signal 2: generalization gap slope (widening gap) ---
        gap_slope = self._gap_slope(recent)
        if gap_slope is not None and gap_slope > self._gap_slope_threshold:
            logger.info(
                "MetaController: gap_slope=%.4f > %.4f → PRUNE (gap widening)",
                gap_slope, self._gap_slope_threshold,
            )
            return MetaAction.PRUNE

        # --- Signal 3: improvement rate ---
        slope = self._normalized_slope(recent)
        if slope < self._improvement_threshold:
            logger.info(
                "MetaController: slope=%.4f < %.4f → SPAWN (plateau)",
                slope, self._improvement_threshold,
            )
            return MetaAction.SPAWN

        logger.debug(
            "MetaController: gap=%.3f gap_slope=%s slope=%.4f → NO_OP",
            gap,
            f"{gap_slope:.4f}" if gap_slope is not None else "n/a",
            slope,
        )
        return MetaAction.NO_OP

    # ------------------------------------------------------------------
    # Internal signal extractors
    # ------------------------------------------------------------------

    def _normalized_gap(self, recent: List[EvalMetrics]) -> float:
        """Mean (train-val)/|val| over recent window. 0 if no val data."""
        values = []
        for m in recent:
            g = m.generalization_gap
            if g is not None:
                val = m.metrics.get("val_score", 0.0)
                if val != 0.0:
                    values.append(g / abs(val))
        return float(np.mean(values)) if values else 0.0

    def _gap_slope(self, recent: List[EvalMetrics]) -> Optional[float]:
        """
        Linear slope of the normalized generalization gap series.

        Returns None if fewer than 3 points have val data (not enough to fit).
        A positive slope means the gap is actively widening — the model is
        overfitting progressively even if the current gap level is tolerable.

        Normalized by |mean(val)| to be scale-independent.
        """
        gap_series = []
        val_mean_abs = []
        for m in recent:
            g = m.generalization_gap
            val = m.metrics.get("val_score")
            if g is not None and val is not None and val != 0.0:
                gap_series.append(g / abs(val))
                val_mean_abs.append(abs(val))

        if len(gap_series) < 3:
            return None

        x = np.arange(len(gap_series), dtype=float)
        slope = float(np.polyfit(x, gap_series, 1)[0])
        return slope

    def _normalized_slope(self, recent: List[EvalMetrics]) -> float:
        """Linear slope of primary_score, normalized by |mean|."""
        scores = np.array([m.primary_score for m in recent], dtype=float)
        mean = float(np.mean(scores))
        if mean == 0.0:
            return 0.0
        x = np.arange(len(scores), dtype=float)
        slope = float(np.polyfit(x, scores, 1)[0])
        return slope / abs(mean)
