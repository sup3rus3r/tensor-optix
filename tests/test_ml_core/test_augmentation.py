import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from tensor_optix.ml_core.adapters.torch_model import TorchMLModel


def _make_dataset(n=40, n_features=10, n_classes=3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_features)).astype(np.float32)
    y = rng.integers(0, n_classes, n).astype(np.int64)
    return X, y


def _build_model(n_features=10, n_classes=3):
    return nn.Sequential(nn.Linear(n_features, 16), nn.ReLU(), nn.Linear(16, n_classes))


def test_augment_called_on_every_training_batch():
    X, y = _make_dataset()
    model = TorchMLModel(_build_model(), loss_fn=nn.CrossEntropyLoss())

    call_count = {"n": 0}

    def augment(X_batch, y_batch):
        call_count["n"] += 1
        return X_batch, y_batch

    model.fit((X, y), epochs=3, batch_size=10, augment=augment)
    assert call_count["n"] == 3 * 4  # 3 epochs * 4 batches/epoch (40/10)


def test_augment_actually_mutates_batch_content():
    X, y = _make_dataset()
    model = TorchMLModel(_build_model(), loss_fn=nn.CrossEntropyLoss())

    seen_batches = []
    original_train_step = model.train_step

    def recording_train_step(batch):
        seen_batches.append(batch[0].clone())
        return original_train_step(batch)

    model.train_step = recording_train_step

    def zero_out(X_batch, y_batch):
        return torch.zeros_like(X_batch), y_batch

    model.fit((X, y), epochs=1, batch_size=10, augment=zero_out)
    assert all(torch.all(b == 0) for b in seen_batches)


def test_augment_not_applied_during_evaluate():
    X, y = _make_dataset()
    model = TorchMLModel(_build_model(), loss_fn=nn.CrossEntropyLoss())

    def exploding_augment(X_batch, y_batch):
        raise AssertionError("augment must not be called during evaluate()")

    # Directly call evaluate() -- it never receives augment at all, since
    # it's not plumbed through evaluate()'s own iter_batches() call.
    metrics = model.evaluate((X, y))
    assert np.isfinite(metrics.loss)


def test_augment_not_applied_to_validation_during_fit():
    X, y = _make_dataset(n=60)
    X_train, y_train = X[:40], y[:40]
    X_val, y_val = X[40:], y[40:]
    model = TorchMLModel(_build_model(), loss_fn=nn.CrossEntropyLoss())

    val_call_count = {"n": 0}
    original_evaluate = model.evaluate

    def counting_evaluate(data):
        val_call_count["n"] += 1
        return original_evaluate(data)

    model.evaluate = counting_evaluate

    def augment(X_batch, y_batch):
        return X_batch, y_batch

    model.fit((X_train, y_train), (X_val, y_val), epochs=2, batch_size=10, augment=augment)
    assert val_call_count["n"] == 2  # evaluate() ran once per epoch, untouched by augment
