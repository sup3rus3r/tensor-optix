"""
tensor_optix.ml_core.adapters.tf_model — TFMLModel(BaseMLModel).

TF/Keras counterpart of torch_model.py -- see that module's docstring for
the overall rationale (logic ported from the proven MLAgent pattern, just
invoked through the real ML contract instead of a fake RL episode).
"""
from __future__ import annotations

import os
from typing import Any, List, Union

import numpy as np

from tensor_optix.core.types import HyperparamSet
from tensor_optix.ml_core.base_model import BaseMLModel
from tensor_optix.ml_core.metrics import MLMetrics, resolve_metrics


class TFMLModel(BaseMLModel):
    """
    Parameters
    ----------
    model:
        Any tf.keras.Model. Forward pass must accept the batched input
        tensor and return predictions compatible with loss_fn.
    loss_fn:
        A tf.keras.losses.Loss instance or any callable (y_true, y_pred)
        -> scalar tensor.
    learning_rate, weight_decay, max_grad_norm:
        Same meaning as TorchMLModel. weight_decay is applied via Keras'
        AdamW when nonzero, plain Adam otherwise (TF's base Adam has no
        weight_decay kwarg, unlike torch.optim.Adam).
    metrics:
        Passed through to evaluate() as resolve_metrics()'s `metrics` arg.
    """

    default_param_bounds = {
        "learning_rate": (1e-5, 1e-2),
        "weight_decay": (0.0, 1e-2),
    }
    default_log_params = ["learning_rate"]

    def __init__(
        self,
        model,
        loss_fn,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        max_grad_norm: float = 10.0,
        metrics: Union[str, List] = "auto",
    ) -> None:
        import tensorflow as tf

        self._tf = tf
        self.model = model
        self.loss_fn = loss_fn
        self.max_grad_norm = max_grad_norm
        self._metrics = metrics

        self._lr = learning_rate
        self._wd = weight_decay
        self._optimizer = self._build_optimizer(learning_rate, weight_decay)
        self._step_count = 0

    def _build_optimizer(self, lr, wd):
        if wd and wd > 0:
            return self._tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=wd)
        return self._tf.keras.optimizers.Adam(learning_rate=lr)

    # ------------------------------------------------------------------
    # BaseMLModel contract
    # ------------------------------------------------------------------

    def predict(self, X: Any) -> np.ndarray:
        x = self._to_tensor(X)
        out = self.model(x, training=False)
        return out.numpy()

    def train_step(self, batch) -> float:
        X, y = batch
        X_t = self._to_tensor(X)
        y_t = self._to_target_tensor(y)

        with self._tf.GradientTape() as tape:
            pred = self.model(X_t, training=True)
            loss = self.loss_fn(y_t, pred)

        grads = tape.gradient(loss, self.model.trainable_variables)
        if self.max_grad_norm is not None:
            grads, _ = self._tf.clip_by_global_norm(grads, self.max_grad_norm)
        self._optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        self._step_count += 1
        return float(loss.numpy())

    def evaluate(self, data) -> MLMetrics:
        y_true_parts, y_pred_parts = [], []
        total_loss, n_batches = 0.0, 0

        for X, y in self.iter_batches(data, batch_size=256):
            X_t = self._to_tensor(X)
            y_t = self._to_target_tensor(y)
            pred = self.model(X_t, training=False)
            loss = self.loss_fn(y_t, pred)
            total_loss += float(loss.numpy())
            n_batches += 1

            y_true_parts.append(y_t.numpy())
            pred_np = pred.numpy()
            if pred_np.ndim == 2 and pred_np.shape[1] > 1:
                pred_np = pred_np.argmax(axis=-1)
            else:
                pred_np = pred_np.reshape(-1)
            y_pred_parts.append(pred_np)

        y_true = np.concatenate(y_true_parts)
        y_pred = np.concatenate(y_pred_parts)
        avg_loss = total_loss / max(n_batches, 1)

        return resolve_metrics(
            self._metrics, y_true=y_true, y_pred=y_pred,
            loss=avg_loss, epoch=self._step_count,
        )

    def iter_batches(self, data, batch_size: int, augment=None):
        """
        Accepts either a raw (X, y) array pair or a native tf.data.Dataset
        -- auto-detected. Mirrors torch_model.py's iter_batches().

        augment: optional callable(X_batch, y_batch) -> (X_batch, y_batch),
        applied per-batch. Only takes effect for the raw (X, y) array
        form -- a tf.data.Dataset already has a natural place to put
        augmentation (.map()), so this hook doesn't duplicate that path.
        """
        tf = self._tf

        if isinstance(data, tf.data.Dataset):
            for batch in data.batch(batch_size):
                yield batch[0].numpy(), batch[1].numpy()
            return

        if isinstance(data, (tuple, list)) and len(data) == 2:
            X, y = data
            X = np.asarray(X)
            y = np.asarray(y)
            n = len(X)
            for start in range(0, n, batch_size):
                X_batch, y_batch = X[start:start + batch_size], y[start:start + batch_size]
                if augment is not None:
                    X_batch, y_batch = augment(X_batch, y_batch)
                yield X_batch, y_batch
            return

        raise TypeError(
            f"TFMLModel.iter_batches() expects a (X, y) pair or a "
            f"tf.data.Dataset, got {type(data).__name__}."
        )

    def get_hyperparams(self) -> HyperparamSet:
        return HyperparamSet(
            params={"learning_rate": self._lr, "weight_decay": self._wd},
            episode_id=self._step_count,
        )

    def set_hyperparams(self, hyperparams: HyperparamSet) -> None:
        self._lr = hyperparams.params.get("learning_rate", self._lr)
        self._wd = hyperparams.params.get("weight_decay", self._wd)
        self._optimizer.learning_rate.assign(self._lr)
        if hasattr(self._optimizer, "weight_decay") and self._optimizer.weight_decay is not None:
            self._optimizer.weight_decay = self._wd

    def save_weights(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        self.model.save_weights(os.path.join(path, "model.weights.h5"))
        np.save(os.path.join(path, "step.npy"), self._step_count)
        np.save(os.path.join(path, "lr.npy"), self._lr)
        np.save(os.path.join(path, "wd.npy"), self._wd)

    def load_weights(self, path: str) -> None:
        self.model.load_weights(os.path.join(path, "model.weights.h5"))
        self._step_count = int(np.load(os.path.join(path, "step.npy")))
        self._lr = float(np.load(os.path.join(path, "lr.npy")))
        self._wd = float(np.load(os.path.join(path, "wd.npy")))
        self._optimizer.learning_rate.assign(self._lr)

    def average_weights(self, paths: List[str]) -> None:
        if not paths:
            return
        all_weights = []
        for p in paths:
            self.model.load_weights(os.path.join(p, "model.weights.h5"))
            all_weights.append([w.numpy().copy() for w in self.model.weights])
        avg = [np.mean([w[i] for w in all_weights], axis=0) for i in range(len(all_weights[0]))]
        for var, avg_val in zip(self.model.weights, avg):
            var.assign(avg_val)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _to_tensor(self, x):
        return self._tf.cast(np.asarray(x), self._tf.float32)

    def _to_target_tensor(self, y):
        arr = np.asarray(y)
        if np.issubdtype(arr.dtype, np.integer):
            return self._tf.cast(arr, self._tf.int64)
        return self._tf.cast(arr, self._tf.float32)
