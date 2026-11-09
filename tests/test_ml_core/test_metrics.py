import numpy as np
import pytest

from tensor_optix.ml_core.metrics import MLMetrics, resolve_metrics, _detect_task_type
from tensor_optix.core.types import EvalMetrics


def test_detect_task_type_int_labels_is_classification():
    y = np.array([0, 1, 2, 1, 0], dtype=np.int64)
    assert _detect_task_type(y) == "classification"


def test_detect_task_type_float_class_like_is_classification():
    y = np.array([0.0, 1.0, 2.0, 1.0, 0.0] * 20, dtype=np.float32)
    assert _detect_task_type(y) == "classification"


def test_detect_task_type_continuous_float_is_regression():
    y = np.random.default_rng(0).standard_normal(100).astype(np.float32)
    assert _detect_task_type(y) == "regression"


def test_resolve_metrics_classification_accuracy_and_confusion_matrix():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 1, 0, 2, 2])
    m = resolve_metrics("classification", y_true=y_true, y_pred=y_pred, loss=0.5, epoch=1)

    assert isinstance(m, MLMetrics)
    assert m.task_type == "classification"
    assert m.primary_score == pytest.approx(m.metrics["accuracy"])
    assert m.metrics["accuracy"] == pytest.approx(4 / 6)
    # 4 correct, 2 wrong -- confusion matrix diagonal sums to 4
    diag_sum = sum(m.metrics[f"confusion_matrix_{i}_{i}"] for i in range(3))
    assert diag_sum == 4


def test_resolve_metrics_regression_mae_rmse_r2():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 1.9, 3.2, 3.8])
    m = resolve_metrics("regression", y_true=y_true, y_pred=y_pred, loss=0.1, epoch=2)

    assert m.task_type == "regression"
    assert m.primary_score == pytest.approx(m.metrics["r2"])
    assert m.metrics["mae"] > 0
    assert m.metrics["r2"] > 0.9  # predictions are close to y_true


def test_resolve_metrics_auto_detects_classification():
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])
    m = resolve_metrics("auto", y_true=y_true, y_pred=y_pred)
    assert m.task_type == "classification"


def test_resolve_metrics_auto_detects_regression():
    rng = np.random.default_rng(1)
    y_true = rng.standard_normal(50)
    y_pred = y_true + rng.standard_normal(50) * 0.1
    m = resolve_metrics("auto", y_true=y_true, y_pred=y_pred)
    assert m.task_type == "regression"


def test_resolve_metrics_requires_y_true_and_y_pred():
    with pytest.raises(ValueError):
        resolve_metrics("classification")


def test_mlmetrics_to_eval_metrics_round_trip():
    m = MLMetrics(
        primary_score=0.9, loss=0.2, task_type="classification",
        metrics={"accuracy": 0.9}, epoch=5,
    )
    em = m.to_eval_metrics()
    assert isinstance(em, EvalMetrics)
    assert em.primary_score == m.primary_score
    assert em.episode_id == m.epoch
    assert em.metrics == m.metrics


def test_resolve_metrics_custom_callables():
    def my_metric(y_true, y_pred):
        return float(np.mean(y_true == y_pred))

    y_true = np.array([0, 1, 1])
    y_pred = np.array([0, 1, 0])
    m = resolve_metrics([my_metric], y_true=y_true, y_pred=y_pred, loss=0.3)
    assert m.task_type == "other"
    assert "my_metric" in m.metrics
    assert m.metrics["my_metric"] == pytest.approx(2 / 3)
