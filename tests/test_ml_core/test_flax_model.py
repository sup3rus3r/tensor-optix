import os
import numpy as np
import pytest

flax = pytest.importorskip("flax")
jax = pytest.importorskip("jax")
optax = pytest.importorskip("optax")

from flax import nnx
import jax.numpy as jnp

from tensor_optix.ml_core.adapters.flax_model import FlaxMLModel
from tensor_optix.core.checkpoint_registry import CheckpointRegistry


def _make_separable_dataset(n=200, n_features=10, n_classes=3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_features)).astype(np.float32)
    y = rng.integers(0, n_classes, n).astype(np.int64)
    for i in range(n):
        X[i, y[i]] += 3.0
    return X, y


class _MLP(nnx.Module):
    def __init__(self, din, dh, dout, rngs):
        self.fc1 = nnx.Linear(din, dh, rngs=rngs)
        self.fc2 = nnx.Linear(dh, dout, rngs=rngs)

    def __call__(self, x):
        return self.fc2(jnp.tanh(self.fc1(x)))


def _build_model(n_features=10, n_classes=3, seed=0):
    return _MLP(n_features, 16, n_classes, nnx.Rngs(seed))


def _cross_entropy_loss(pred, y):
    return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(pred, y))


def test_fit_predict_evaluate_on_separable_classification_task():
    X, y = _make_separable_dataset()
    X_train, y_train = X[:160], y[:160]
    X_val, y_val = X[160:], y[160:]

    model = FlaxMLModel(_build_model(), loss_fn=_cross_entropy_loss, learning_rate=1e-2)
    history = model.fit((X_train, y_train), (X_val, y_val), epochs=10, batch_size=16)

    assert history.epochs_run == 10
    assert len(history.val_metrics) == 10
    final = history.val_metrics[-1]
    assert final.task_type == "classification"
    assert final.metrics["accuracy"] > 0.8

    preds = model.predict(X_val)
    assert preds.shape == (len(X_val), 3)


def test_evaluate_without_fit_returns_finite_metrics():
    X, y = _make_separable_dataset(n=40)
    model = FlaxMLModel(_build_model(), loss_fn=_cross_entropy_loss)
    metrics = model.evaluate((X, y))
    assert np.isfinite(metrics.primary_score)
    assert np.isfinite(metrics.loss)


def test_train_step_reduces_loss_over_steps():
    X, y = _make_separable_dataset(n=60)
    model = FlaxMLModel(_build_model(), loss_fn=_cross_entropy_loss, learning_rate=1e-1)

    first_loss = model.train_step((X[:20], y[:20]))
    for _ in range(20):
        last_loss = model.train_step((X[:20], y[:20]))

    assert last_loss < first_loss


def test_checkpoint_restore_matches_post_fit_weights(tmp_path):
    rng = np.random.default_rng(3)
    X = rng.standard_normal((150, 8)).astype(np.float32)
    y = rng.integers(0, 2, 150).astype(np.int64)
    X_train, y_train = X[:120], y[:120]
    X_val, y_val = X[120:], y[120:]

    ckpt_dir = str(tmp_path / "checkpoints")
    model = FlaxMLModel(_build_model(8, 2), loss_fn=_cross_entropy_loss, learning_rate=1e-2)
    model.fit(
        (X_train, y_train), (X_val, y_val),
        epochs=20, batch_size=16, early_stopping_patience=5,
        checkpoint_dir=ckpt_dir,
    )

    assert os.path.exists(os.path.join(ckpt_dir, "manifest.json"))

    state_after_fit = nnx.to_pure_dict(nnx.state(model.model))

    fresh_model = FlaxMLModel(_build_model(8, 2), loss_fn=_cross_entropy_loss)
    CheckpointRegistry(ckpt_dir).load_best(fresh_model)

    state_restored = nnx.to_pure_dict(nnx.state(fresh_model.model))

    def _assert_close(a, b):
        for k in a:
            if isinstance(a[k], dict):
                _assert_close(a[k], b[k])
            else:
                assert np.allclose(np.asarray(a[k]), np.asarray(b[k]))

    _assert_close(state_after_fit, state_restored)


def test_lr_schedule_and_augment_work_with_flax_model():
    from tensor_optix.ml_core.lr_schedules import StepLR

    X, y = _make_separable_dataset(n=40)
    model = FlaxMLModel(_build_model(), loss_fn=_cross_entropy_loss, learning_rate=1.0)

    call_count = {"n": 0}

    def augment(X_batch, y_batch):
        call_count["n"] += 1
        return X_batch, y_batch

    schedule = StepLR(base_lr=1.0, step_size=2, gamma=0.5)
    model.fit((X, y), epochs=4, batch_size=10, lr_schedule=schedule, augment=augment)

    assert call_count["n"] == 4 * 4  # 4 epochs * 4 batches/epoch (40/10)
    assert model._lr == pytest.approx(schedule(3))
