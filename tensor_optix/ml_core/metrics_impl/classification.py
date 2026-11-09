"""
tensor_optix.ml_core.metrics_impl.classification — pure-numpy classification
metrics: accuracy, precision, recall, F1, confusion matrix.

Framework-agnostic by design (operates on plain numpy arrays of predicted
and true class indices) so the same implementation serves both the Torch
and TF adapters without duplication -- each adapter's evaluate() converts
its framework's tensors to numpy before calling these.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    """
    Standard confusion matrix, rows = true class, columns = predicted class.
    cm[i, j] = count of samples with true label i predicted as j.
    """
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0:
        return 0.0
    return float(np.mean(y_true == y_pred))


def precision_recall_f1(cm: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Per-class precision/recall/F1 from a confusion matrix.

    precision_i = cm[i,i] / sum(cm[:,i])  (of predicted i, how many were right)
    recall_i    = cm[i,i] / sum(cm[i,:])  (of true i, how many were found)
    f1_i        = 2 * p * r / (p + r)

    Division-by-zero (an unpredicted or unseen class) yields 0.0 for that
    class rather than NaN/inf, so downstream aggregation (macro average)
    stays a finite, usable number.
    """
    n = cm.shape[0]
    tp = np.diag(cm).astype(np.float64)
    predicted_totals = cm.sum(axis=0).astype(np.float64)
    true_totals = cm.sum(axis=1).astype(np.float64)

    precision = np.divide(tp, predicted_totals, out=np.zeros(n), where=predicted_totals > 0)
    recall = np.divide(tp, true_totals, out=np.zeros(n), where=true_totals > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros(n), where=denom > 0)

    return {"precision": precision, "recall": recall, "f1": f1}


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> Dict[str, float]:
    """
    Full classification metric set as a flat Dict[str, float], suitable for
    MLMetrics.metrics (and, transitively, CheckpointRegistry's
    json.dump(metadata) -- every value here is already a plain Python
    float/int, never a numpy scalar or nested array, except confusion_matrix
    which is pre-flattened into per-cell keys for the same JSON-safety
    reason).

    Keys:
        accuracy
        precision_macro, recall_macro, f1_macro   (unweighted mean over classes)
        precision_<i>, recall_<i>, f1_<i>          (per-class, i = 0..n_classes-1)
        confusion_matrix_<i>_<j>                    (flattened cm cells)
    """
    cm = confusion_matrix(y_true, y_pred, n_classes)
    per_class = precision_recall_f1(cm)

    out: Dict[str, float] = {
        "accuracy": accuracy(y_true, y_pred),
        "precision_macro": float(np.mean(per_class["precision"])),
        "recall_macro": float(np.mean(per_class["recall"])),
        "f1_macro": float(np.mean(per_class["f1"])),
    }
    for i in range(n_classes):
        out[f"precision_{i}"] = float(per_class["precision"][i])
        out[f"recall_{i}"] = float(per_class["recall"][i])
        out[f"f1_{i}"] = float(per_class["f1"][i])
    for i in range(n_classes):
        for j in range(n_classes):
            out[f"confusion_matrix_{i}_{j}"] = int(cm[i, j])
    return out


def unflatten_confusion_matrix(metrics: Dict[str, float], n_classes: int) -> np.ndarray:
    """Inverse of classification_metrics()'s confusion_matrix_<i>_<j>
    flattening -- reconstructs the 2-D confusion matrix for display/plotting
    from a stored MLMetrics.metrics dict."""
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for i in range(n_classes):
        for j in range(n_classes):
            cm[i, j] = int(metrics.get(f"confusion_matrix_{i}_{j}", 0))
    return cm
