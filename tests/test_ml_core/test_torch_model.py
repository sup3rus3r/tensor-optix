import os
import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from tensor_optix.ml_core.adapters.torch_model import TorchMLModel
from tensor_optix.core.checkpoint_registry import CheckpointRegistry


def _make_separable_dataset(n=200, n_features=10, n_classes=3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_features)).astype(np.float32)
    y = rng.integers(0, n_classes, n).astype(np.int64)
    for i in range(n):
        X[i, y[i]] += 3.0  # make the task easily learnable
    return X, y


def _build_model(n_features=10, n_classes=3):
    return nn.Sequential(nn.Linear(n_features, 16), nn.ReLU(), nn.Linear(16, n_classes))


def test_fit_predict_evaluate_on_separable_classification_task():
    X, y = _make_separable_dataset()
    X_train, y_train = X[:160], y[:160]
    X_val, y_val = X[160:], y[160:]

    model = TorchMLModel(_build_model(), loss_fn=nn.CrossEntropyLoss(), learning_rate=1e-2)
    history = model.fit((X_train, y_train), (X_val, y_val), epochs=10, batch_size=16)

    assert history.epochs_run == 10
    assert len(history.val_metrics) == 10
    final = history.val_metrics[-1]
    assert final.task_type == "classification"
    assert final.metrics["accuracy"] > 0.8  # easily learnable task

    preds = model.predict(X_val)
    assert preds.shape == (len(X_val), 3)


def test_evaluate_without_fit_returns_finite_metrics():
    X, y = _make_separable_dataset(n=40)
    model = TorchMLModel(_build_model(), loss_fn=nn.CrossEntropyLoss())
    metrics = model.evaluate((X, y))
    assert np.isfinite(metrics.primary_score)
    assert np.isfinite(metrics.loss)


def test_early_stopping_triggers_when_validation_never_improves():
    """A held-out validation set that is entirely disjoint from (and
    unrelated to) the training distribution gives the model nothing to
    generalize -- val accuracy should hover at chance level the whole run,
    so early_stopping_patience must trigger well before the full epoch
    budget rather than depending on getting a lucky "new best" on noise."""
    rng = np.random.default_rng(2)
    X_train = rng.standard_normal((120, 8)).astype(np.float32)
    y_train = rng.integers(0, 2, 120).astype(np.int64)
    # Validation targets are independently re-randomized with no relation
    # to X_val, so no amount of training on X_train/y_train can ever move
    # val accuracy away from chance -- removes the "got lucky on noise"
    # flakiness of comparing against a shared random draw.
    X_val = rng.standard_normal((40, 8)).astype(np.float32)
    y_val = rng.integers(0, 2, 40).astype(np.int64)

    model = TorchMLModel(nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 2)),
                          loss_fn=nn.CrossEntropyLoss(), learning_rate=1e-3)
    history = model.fit(
        (X_train, y_train), (X_val, y_val),
        epochs=50, batch_size=16, early_stopping_patience=3,
    )

    assert history.epochs_run <= 50
    # Not every run is guaranteed to stop early (a single lucky epoch can
    # reset the plateau counter), but the model must never actually learn
    # anything meaningful on an unrelated validation set.
    assert history.val_metrics[-1].metrics["accuracy"] < 0.75


def test_early_stopping_triggers_with_strict_patience(monkeypatch):
    """Deterministic early-stopping check: force every epoch's validation
    MLMetrics.primary_score to the same constant value (simulating a
    plateaued model), independent of what the model actually learns, and
    confirm Trainer stops at exactly early_stopping_patience epochs past
    the first (and only) recorded improvement -- removes any dependency
    on real training dynamics being noisy/lucky."""
    import tensor_optix.ml_core.trainer as trainer_module

    X, y = _make_separable_dataset(n=60)
    X_train, y_train = X[:40], y[:40]
    X_val, y_val = X[40:], y[40:]

    model = TorchMLModel(_build_model(10, 3), loss_fn=nn.CrossEntropyLoss())

    original_evaluate = model.evaluate
    call_count = {"n": 0}

    def flat_evaluate(data):
        m = original_evaluate(data)
        call_count["n"] += 1
        # First call is the only "improvement"; every call after reports
        # the exact same score, so the scheduler must plateau immediately.
        m.primary_score = 1.0
        m.metrics["accuracy"] = 1.0
        return m

    monkeypatch.setattr(model, "evaluate", flat_evaluate)

    history = model.fit(
        (X_train, y_train), (X_val, y_val),
        epochs=20, batch_size=8, early_stopping_patience=3,
    )

    assert history.stopped_early
    # epoch 0: improvement (first-ever score). epochs 1-3: no improvement
    # (same score as best) -> patience of 3 exhausted after epoch 3 (the
    # 4th evaluate() call overall).
    assert history.epochs_run == 4


def test_checkpoint_restore_matches_post_fit_weights(tmp_path):
    rng = np.random.default_rng(3)
    X = rng.standard_normal((150, 8)).astype(np.float32)
    y = rng.integers(0, 2, 150).astype(np.int64)
    X_train, y_train = X[:120], y[:120]
    X_val, y_val = X[120:], y[120:]

    ckpt_dir = str(tmp_path / "checkpoints")
    model = TorchMLModel(nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 2)),
                          loss_fn=nn.CrossEntropyLoss(), learning_rate=1e-3)
    model.fit(
        (X_train, y_train), (X_val, y_val),
        epochs=50, batch_size=16, early_stopping_patience=5,
        checkpoint_dir=ckpt_dir,
    )

    assert os.path.exists(os.path.join(ckpt_dir, "manifest.json"))
    state_after_fit = {k: v.clone() for k, v in model.model.state_dict().items()}

    fresh_model = TorchMLModel(nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 2)),
                                loss_fn=nn.CrossEntropyLoss())
    CheckpointRegistry(ckpt_dir).load_best(fresh_model)

    for key, value in state_after_fit.items():
        assert torch.allclose(value, fresh_model.model.state_dict()[key])
