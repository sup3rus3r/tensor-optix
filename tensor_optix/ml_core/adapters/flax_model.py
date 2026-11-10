"""
tensor_optix.ml_core.adapters.flax_model — FlaxMLModel(BaseMLModel).

JAX/Flax NNX counterpart of torch_model.py/tf_model.py -- see those
modules' docstrings for the overall rationale. Conventions (nnx.Optimizer,
nnx.value_and_grad, nnx.state/to_pure_dict for weight persistence) mirror
tensor_optix.algorithms.flax_ppo, the only other place in the codebase
that uses Flax NNX.
"""
from __future__ import annotations

import os
import pickle
from typing import Any, List, Union

import numpy as np

from tensor_optix.core.types import HyperparamSet
from tensor_optix.ml_core.base_model import BaseMLModel
from tensor_optix.ml_core.metrics import MLMetrics, resolve_metrics


class FlaxMLModel(BaseMLModel):
    """
    Parameters
    ----------
    model:
        Any flax.nnx.Module. Called as model(x) -- must accept the batched
        input array and return predictions compatible with loss_fn.
    loss_fn:
        callable(pred, target) -> scalar jax array. There is no Flax
        equivalent of torch.nn.Module/tf.keras.losses.Loss instances, so
        this is always a plain function here (e.g. optax.softmax_cross_entropy
        wrapped to reduce to a scalar, or a custom function).
    learning_rate, weight_decay, max_grad_norm:
        Same meaning as TorchMLModel/TFMLModel. weight_decay is applied via
        optax.adamw when nonzero, plain optax.adam otherwise.
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
        import optax
        from flax import nnx

        self._nnx = nnx
        self._optax = optax
        self.model = model
        self.loss_fn = loss_fn
        self.max_grad_norm = max_grad_norm
        self._metrics = metrics

        self._lr = learning_rate
        self._wd = weight_decay
        self._optimizer = self._build_optimizer(learning_rate, weight_decay, max_grad_norm)
        self._step_count = 0

    def _build_optimizer(self, lr, wd, max_grad_norm):
        tx_parts = []
        if max_grad_norm is not None:
            tx_parts.append(self._optax.clip_by_global_norm(max_grad_norm))
        if wd and wd > 0:
            tx_parts.append(self._optax.adamw(learning_rate=lr, weight_decay=wd))
        else:
            tx_parts.append(self._optax.adam(learning_rate=lr))
        tx = self._optax.chain(*tx_parts)
        return self._nnx.Optimizer(self.model, tx, wrt=self._nnx.Param)

    # ------------------------------------------------------------------
    # BaseMLModel contract
    # ------------------------------------------------------------------

    def predict(self, X: Any) -> np.ndarray:
        x = self._to_array(X)
        out = self.model(x)
        return np.asarray(out)

    def train_step(self, batch) -> float:
        import jax

        X, y = batch
        X_arr = self._to_array(X)
        y_arr = self._to_array(y)

        def loss_fn(model):
            pred = model(X_arr)
            return self.loss_fn(pred, y_arr)

        loss, grads = self._nnx.value_and_grad(loss_fn)(self.model)
        self._optimizer.update(self.model, grads)
        self._step_count += 1
        return float(jax.numpy.asarray(loss))

    def evaluate(self, data) -> MLMetrics:
        y_true_parts, y_pred_parts = [], []
        total_loss, n_batches = 0.0, 0

        for X, y in self.iter_batches(data, batch_size=256):
            X_arr = self._to_array(X)
            y_arr = self._to_array(y)
            pred = self.model(X_arr)
            loss = float(np.asarray(self.loss_fn(pred, y_arr)))
            total_loss += loss
            n_batches += 1

            y_true_parts.append(np.asarray(y_arr))
            pred_np = np.asarray(pred)
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
        Accepts a raw (X, y) array pair only -- JAX/Flax has no native
        Dataset/DataLoader equivalent in this codebase's dependency set
        (grain/tf.data could be used by callers directly, but ml_core does
        not assume either is installed).

        augment: optional callable(X_batch, y_batch) -> (X_batch, y_batch),
        applied per-batch. Mirrors torch_model.py/tf_model.py's hook.
        """
        if not (isinstance(data, (tuple, list)) and len(data) == 2):
            raise TypeError(
                f"FlaxMLModel.iter_batches() expects a (X, y) pair, "
                f"got {type(data).__name__}."
            )

        X, y = data
        X = np.asarray(X)
        y = np.asarray(y)
        n = len(X)
        for start in range(0, n, batch_size):
            X_batch, y_batch = X[start:start + batch_size], y[start:start + batch_size]
            if augment is not None:
                X_batch, y_batch = augment(X_batch, y_batch)
            yield X_batch, y_batch

    def get_hyperparams(self) -> HyperparamSet:
        return HyperparamSet(
            params={"learning_rate": self._lr, "weight_decay": self._wd},
            episode_id=self._step_count,
        )

    def set_hyperparams(self, hyperparams: HyperparamSet) -> None:
        self._lr = hyperparams.params.get("learning_rate", self._lr)
        self._wd = hyperparams.params.get("weight_decay", self._wd)
        # optax schedules/transforms are immutable once built -- rebuilding
        # the optimizer is the same approach flax_ppo.py uses when hyperparams
        # change (see its own set_hyperparams()), not a new pattern here.
        self._optimizer = self._build_optimizer(self._lr, self._wd, self.max_grad_norm)

    def save_weights(self, path: str) -> None:
        from flax import nnx
        os.makedirs(path, exist_ok=True)
        state = nnx.state(self.model)
        pure_dict = nnx.to_pure_dict(state)
        with open(os.path.join(path, "model.pkl"), "wb") as f:
            pickle.dump(pure_dict, f, protocol=4)
        with open(os.path.join(path, "meta.pkl"), "wb") as f:
            pickle.dump({"step": self._step_count, "lr": self._lr, "wd": self._wd}, f, protocol=4)

    def load_weights(self, path: str) -> None:
        from flax import nnx
        with open(os.path.join(path, "model.pkl"), "rb") as f:
            pure_dict = pickle.load(f)
        state = nnx.state(self.model)
        nnx.replace_by_pure_dict(state, pure_dict)
        nnx.update(self.model, state)

        meta_path = os.path.join(path, "meta.pkl")
        if os.path.exists(meta_path):
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
            self._step_count = meta.get("step", 0)
            self._lr = meta.get("lr", self._lr)
            self._wd = meta.get("wd", self._wd)
            self._optimizer = self._build_optimizer(self._lr, self._wd, self.max_grad_norm)

    def average_weights(self, paths: List[str]) -> None:
        from flax import nnx
        if not paths:
            return
        pure_dicts = []
        for p in paths:
            with open(os.path.join(p, "model.pkl"), "rb") as f:
                pure_dicts.append(pickle.load(f))

        def _average_leaves(*leaves):
            return np.mean(np.stack([np.asarray(leaf) for leaf in leaves]), axis=0)

        import jax
        averaged = jax.tree_util.tree_map(_average_leaves, *pure_dicts)
        state = nnx.state(self.model)
        nnx.replace_by_pure_dict(state, averaged)
        nnx.update(self.model, state)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _to_array(self, x):
        import jax.numpy as jnp
        if hasattr(x, "shape") and hasattr(x, "dtype") and not isinstance(x, np.ndarray):
            return x  # already a jax/numpy-like array
        return jnp.asarray(np.asarray(x))
