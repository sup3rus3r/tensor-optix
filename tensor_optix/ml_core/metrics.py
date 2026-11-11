"""
tensor_optix.ml_core.metrics — MLMetrics dataclass and the metrics registry.

Mirrors tensor_optix.ml.loss_registry's resolve_loss()/available_losses()/
"auto" pattern exactly, for the same reason that module exists: a small,
explicit registry the user can override, with auto-detection as the
default so the common case needs zero configuration.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

from tensor_optix.core.types import EvalMetrics
from tensor_optix.utils.console import safe_print

logger = logging.getLogger(__name__)


@dataclass
class MLMetrics:
    """
    Scored output from a BaseMLModel's evaluate() call.

    primary_score: higher-is-better scalar, feeding CheckpointRegistry and
        BackoffScheduler exactly as EvalMetrics.primary_score does for RL --
        for a loss-based task this is -loss (matching MLAgent's existing,
        already-proven convention), so "higher is better" holds universally.
    loss: the raw loss value (for display/logging -- NOT what checkpointing
        compares; primary_score is).
    task_type: "classification" | "regression" | "other" -- records which
        metric family metrics was built from.
    metrics: flat Dict[str, float] of every computed metric (accuracy,
        f1_macro, confusion_matrix_<i>_<j>, mae, r2, ...) -- see
        metrics_impl/classification.py and metrics_impl/regression.py for
        the exact key sets.
    epoch: which training epoch this evaluation was taken at.
    """
    primary_score: float
    loss: float
    task_type: str
    metrics: Dict[str, float]
    epoch: int
    timestamp: float = field(default_factory=time.time)

    def to_eval_metrics(self) -> EvalMetrics:
        """
        Bridge to the existing, reused-as-is CheckpointRegistry, which only
        knows EvalMetrics. MLMetrics is the richer ML-native type; this is
        the one-line adapter so CheckpointRegistry needs zero changes.
        """
        return EvalMetrics(
            primary_score=self.primary_score,
            metrics=self.metrics,
            episode_id=self.epoch,
            timestamp=self.timestamp,
        )


_METRIC_FAMILIES = {
    "classification": "accuracy, precision_macro, recall_macro, f1_macro, "
                       "per-class precision/recall/f1, flattened confusion_matrix",
    "regression": "mae, rmse, r2",
}


def available_metrics() -> str:
    """Human-readable list of all supported metric task types."""
    lines = ["Available metric task types:", ""]
    for k, desc in _METRIC_FAMILIES.items():
        lines.append(f'  "{k}"  ->  {desc}')
    lines.append("")
    lines.append('  "auto"  ->  detect from targets (default): integer/class-index '
                 "targets -> classification, float targets -> regression")
    lines.append("  A list of custom metric callables (y_true, y_pred) -> float "
                 "can also be passed, each named by its __name__.")
    return "\n".join(lines)


def resolve_metrics(
    metrics: Union[str, List[Callable]],
    y_true: Optional[np.ndarray] = None,
    y_pred: Optional[np.ndarray] = None,
    n_classes: Optional[int] = None,
    loss: float = 0.0,
    epoch: int = 0,
) -> MLMetrics:
    """
    Compute an MLMetrics from (y_true, y_pred), auto-detecting task type
    from y_true's dtype/values when metrics="auto" (default) -- same
    auto-detection spirit as loss_registry.resolve_loss()'s dataset-sampling
    heuristic, but simpler since the targets are already in hand here
    rather than needing to be sampled from a Dataset first.

    Parameters
    ----------
    metrics:
        "auto" (default), "classification", "regression", or a list of
        custom callables (y_true, y_pred) -> float, each contributing one
        key to MLMetrics.metrics named by its __name__.
    y_true, y_pred:
        Flat numpy arrays of true/predicted values. Required unless
        metrics is a list of custom callables that don't need them (not a
        supported case here -- y_true/y_pred are always required).
    n_classes:
        Required for classification (explicit or auto-detected). If None
        when needed, inferred as max(y_true.max(), y_pred.max()) + 1.
    loss, epoch:
        Passed straight through into the returned MLMetrics.
    """
    if y_true is None or y_pred is None:
        raise ValueError("resolve_metrics() requires both y_true and y_pred.")

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if isinstance(metrics, list):
        computed = {}
        for fn in metrics:
            name = getattr(fn, "__name__", repr(fn))
            computed[name] = float(fn(y_true, y_pred))
        # Custom metrics: no principled "higher is better" default --
        # primary_score falls back to -loss, same convention as every
        # other task_type, so checkpointing still has a comparable score.
        return MLMetrics(
            primary_score=-float(loss), loss=float(loss),
            task_type="other", metrics=computed, epoch=epoch,
        )

    key = metrics.lower() if isinstance(metrics, str) else metrics

    if key == "auto":
        key = _detect_task_type(y_true)
        logger.info("metrics='auto' detected: '%s'", key)
        safe_print(
            f"  [tensor-optix] metrics='auto' detected: '{key}'  "
            f"(set metrics='{key}' explicitly to suppress this message)",
            flush=True,
        )

    if key == "classification":
        from tensor_optix.ml_core.metrics_impl.classification import classification_metrics
        # Class-index arrays must be flat and integer-valued. Two shapes
        # reach here that are not: targets of shape (N, 1), and the single-
        # output binary head, whose "prediction" is a raw logit/probability
        # per sample rather than a class index.
        y_true = y_true.reshape(-1)
        y_pred = y_pred.reshape(-1)
        if np.issubdtype(y_pred.dtype, np.floating) and not np.allclose(y_pred, np.round(y_pred)):
            # Probabilities live in [0, 1] -> threshold 0.5; otherwise logits -> 0.
            is_prob = y_pred.min() >= 0.0 and y_pred.max() <= 1.0
            y_pred = (y_pred > (0.5 if is_prob else 0.0))
        y_true = y_true.astype(np.int64)
        y_pred = y_pred.astype(np.int64)
        resolved_n_classes = n_classes
        if resolved_n_classes is None:
            resolved_n_classes = int(max(y_true.max(), y_pred.max())) + 1
        computed = classification_metrics(y_true, y_pred, resolved_n_classes)
        return MLMetrics(
            primary_score=computed["accuracy"], loss=float(loss),
            task_type="classification", metrics=computed, epoch=epoch,
        )

    if key == "regression":
        from tensor_optix.ml_core.metrics_impl.regression import regression_metrics
        computed = regression_metrics(y_true, y_pred)
        # Regression: higher-is-better score is r2 (bounded above by 1.0,
        # unlike -mae/-rmse which have no natural ceiling) -- matches how
        # CheckpointRegistry/BackoffScheduler expect "improvement" to mean
        # approaching a known-good value, not an unbounded negative loss.
        return MLMetrics(
            primary_score=computed["r2"], loss=float(loss),
            task_type="regression", metrics=computed, epoch=epoch,
        )

    raise ValueError(f"Unknown metrics task type '{metrics}'.\n{available_metrics()}")


def _detect_task_type(y_true: np.ndarray) -> str:
    """
    Integer/class-index targets -> classification.
    Float targets -> regression.

    Mirrors loss_registry._auto_detect()'s dtype-based heuristic (int/long
    targets -> cross_entropy there, classification here).
    """
    if np.issubdtype(y_true.dtype, np.integer):
        return "classification"
    if np.issubdtype(y_true.dtype, np.floating):
        # Float but every value is a whole number close to an int index
        # (e.g. class labels stored as float32) -- still classification.
        if np.allclose(y_true, np.round(y_true)) and y_true.max() < 1000:
            unique_count = len(np.unique(y_true))
            if unique_count <= 1000 and unique_count < len(y_true):
                return "classification"
        return "regression"
    return "regression"
