import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from tensor_optix.ml_core.adapters.tf_model import TFMLModel


def _make_dataset(n=40, n_features=10, n_classes=3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_features)).astype(np.float32)
    y = rng.integers(0, n_classes, n).astype(np.int64)
    return X, y


def _build_model(n_features=10, n_classes=3):
    inputs = tf.keras.Input(shape=(n_features,))
    x = tf.keras.layers.Dense(16, activation="relu")(inputs)
    outputs = tf.keras.layers.Dense(n_classes)(x)
    return tf.keras.Model(inputs, outputs)


def test_augment_called_on_every_training_batch():
    X, y = _make_dataset()
    model = TFMLModel(_build_model(), loss_fn=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))

    call_count = {"n": 0}

    def augment(X_batch, y_batch):
        call_count["n"] += 1
        return X_batch, y_batch

    model.fit((X, y), epochs=3, batch_size=10, augment=augment)
    assert call_count["n"] == 3 * 4


def test_augment_not_applied_to_validation_during_fit():
    X, y = _make_dataset(n=60)
    X_train, y_train = X[:40], y[:40]
    X_val, y_val = X[40:], y[40:]
    model = TFMLModel(_build_model(), loss_fn=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))

    val_call_count = {"n": 0}
    original_evaluate = model.evaluate

    def counting_evaluate(data):
        val_call_count["n"] += 1
        return original_evaluate(data)

    model.evaluate = counting_evaluate

    def augment(X_batch, y_batch):
        return X_batch, y_batch

    model.fit((X_train, y_train), (X_val, y_val), epochs=2, batch_size=10, augment=augment)
    assert val_call_count["n"] == 2
