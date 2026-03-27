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

    1. Generalization gap (train - val) / |val|
       Large gap → model is overfitting → PRUNE (remove the most overfit agent)

    2. train / val Pearson correlation over recent window
       Low correlation → train is moving but val is not following → SPAWN
       (explore a different region of the policy space)

    3. Normalized val slope (improvement rate)
       Flat or declining → genuine plateau → SPAWN

    Priority: gap check runs before correlation before slope. If the budget
    is exhausted, STOP is returned regardless of other signals.

    This is a rule-based controller. It implements no learning of its own.
    The interface is intentionally minimal so it can be swapped for a learned
    policy (e.g. an RL agent whose observation is pm.status() + metrics
    features and whose action space is MetaAction) without any API change.

    Parameters:
        gap_threshold:         normalized gap above which PRUNE fires (default 0.2)
        corr_threshold:        train/val correlation below which SPAWN fires (default 0.5)
        improvement_threshold: normalized val slope below which SPAWN fires (default 0.01)
        window:                number of recent EvalMetrics to consider (default 10)
    """

    def __init__(
        self,
        gap_threshold: float = 0.2,
        corr_threshold: float = 0.5,
        improvement_threshold: float = 0.01,
        window: int = 10,
    ):
        self._gap_threshold = gap_threshold
        self._corr_threshold = corr_threshold
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

        # --- Signal 1: generalization gap ---
        gap = self._normalized_gap(recent)
        if gap > self._gap_threshold:
            logger.info(
                "MetaController: gap=%.3f > %.3f → PRUNE (overfitting)",
                gap, self._gap_threshold,
            )
            return MetaAction.PRUNE

        # --- Signal 2: train/val correlation ---
        corr = self._train_val_corr(recent)
        if corr is not None and corr < self._corr_threshold:
            logger.info(
                "MetaController: corr=%.3f < %.3f → SPAWN (train/val diverging)",
                corr, self._corr_threshold,
            )
            return MetaAction.SPAWN

        # --- Signal 3: improvement rate ---
        slope = self._normalized_slope(recent)
        if slope < self._improvement_threshold:
            logger.info(
                "MetaController: slope=%.4f < %.4f → SPAWN (plateau)",
                slope, self._improvement_threshold,
            )
            return MetaAction.SPAWN

        logger.debug(
            "MetaController: gap=%.3f corr=%s slope=%.4f → NO_OP",
            gap, f"{corr:.3f}" if corr is not None else "n/a", slope,
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

    def _train_val_corr(self, recent: List[EvalMetrics]) -> Optional[float]:
        """
        Pearson correlation between train and val score series.
        Returns None if val data is not present or variance is zero.
        """
        train = [m.metrics.get("train_score") for m in recent]
        val   = [m.metrics.get("val_score")   for m in recent]
        if any(v is None for v in train + val):
            return None
        t = np.array(train, dtype=float)
        v = np.array(val,   dtype=float)
        if np.std(t) == 0.0 or np.std(v) == 0.0:
            return 1.0  # no variance → assume correlated
        return float(np.corrcoef(t, v)[0, 1])

    def _normalized_slope(self, recent: List[EvalMetrics]) -> float:
        """Linear slope of primary_score, normalized by |mean|."""
        scores = np.array([m.primary_score for m in recent], dtype=float)
        mean = float(np.mean(scores))
        if mean == 0.0:
            return 0.0
        x = np.arange(len(scores), dtype=float)
        slope = float(np.polyfit(x, scores, 1)[0])
        return slope / abs(mean)
