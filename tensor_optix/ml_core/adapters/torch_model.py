"""
tensor_optix.ml_core.adapters.torch_model — TorchMLModel(BaseMLModel).

Wraps an arbitrary nn.Module with the ML-native fit/predict/evaluate
contract. The device-handling, optimizer construction, gradient-step, and
save/load/SWA logic below is ported near-verbatim from
tensor_optix.ml.ml_agent.MLAgent -- that logic was already correct, it was
just reachable only through the wrong (RL-episode-shaped) contract. The
only real change is train_step()/evaluate() working on a plain (X, y)
batch instead of a fake EpisodeData.
"""
from __future__ import annotations

import os
from typing import Any, List, Optional, Union

import numpy as np

from tensor_optix.core.types import HyperparamSet
from tensor_optix.ml_core.base_model import BaseMLModel
from tensor_optix.ml_core.metrics import MLMetrics, resolve_metrics


class TorchMLModel(BaseMLModel):
    """
    Parameters
    ----------
    model:
        Any nn.Module. Forward pass must accept the batched input tensor
        and return predictions compatible with loss_fn.
    loss_fn:
        Instantiated loss nn.Module (e.g. from tensor_optix.ml.loss_registry
        .resolve_loss(), reused here since it's already framework-correct
        and task-agnostic), or any callable (pred, target) -> scalar tensor.
    learning_rate, weight_decay, max_grad_norm, device:
        Same meaning and defaults as MLAgent.
    metrics:
        Passed through to evaluate() as resolve_metrics()'s `metrics` arg
        (default "auto") -- can be overridden per-call to evaluate() too.
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
        device: str = "auto",
        metrics: Union[str, List] = "auto",
    ) -> None:
        import torch

        self._torch = torch
        self.model = model
        self.loss_fn = loss_fn
        self.max_grad_norm = max_grad_norm
        self._metrics = metrics

        _device = (
            ("cuda" if torch.cuda.is_available() else "cpu")
            if device == "auto" else device
        )
        self.device = torch.device(_device)
        self.model.to(self.device)
        if hasattr(self.loss_fn, "to"):
            self.loss_fn.to(self.device)

        self._lr = learning_rate
        self._wd = weight_decay
        self._optimizer = torch.optim.Adam(
            (p for p in self.model.parameters() if p.requires_grad),
            lr=learning_rate, weight_decay=weight_decay,
        )
        self._step_count = 0

    # ------------------------------------------------------------------
    # BaseMLModel contract
    # ------------------------------------------------------------------

    def predict(self, X: Any) -> np.ndarray:
        x = self._to_tensor(X)
        self.model.eval()
        with self._torch.no_grad():
            out = self.model(x)
        return out.cpu().numpy()

    def train_step(self, batch) -> float:
        X, y = batch
        X = self._to_tensor(X)
        y = self._to_target_tensor(y)

        self.model.train()
        self._optimizer.zero_grad()
        pred = self.model(X)
        loss = self.loss_fn(pred, y)
        loss.backward()

        if self.max_grad_norm is not None:
            self._torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

        self._optimizer.step()
        self._step_count += 1
        return float(loss.item())

    def evaluate(self, data) -> MLMetrics:
        y_true_parts, y_pred_parts = [], []
        total_loss, n_batches = 0.0, 0

        self.model.eval()
        with self._torch.no_grad():
            for X, y in self.iter_batches(data, batch_size=256):
                X_t = self._to_tensor(X)
                y_t = self._to_target_tensor(y)
                pred = self.model(X_t)
                loss = self.loss_fn(pred, y_t)
                total_loss += float(loss.item())
                n_batches += 1

                y_true_parts.append(y_t.cpu().numpy())
                pred_np = pred.cpu().numpy()
                # Classification heads typically output per-class logits
                # ([B, n_classes]); reduce to a predicted class index so
                # resolve_metrics() always receives flat (y_true, y_pred)
                # arrays of the same shape, matching what
                # metrics_impl/classification.py expects.
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
        Accepts either a raw (X, y) array pair or a native
        torch.utils.data.Dataset/DataLoader -- auto-detected, mirroring
        tensor_optix.ml.loss_registry's auto-detection style.

        augment: optional callable(X_batch, y_batch) -> (X_batch, y_batch),
        applied per-batch. Only takes effect for the raw (X, y) array
        form -- a Dataset/DataLoader already has a natural place to put
        augmentation (__getitem__ or a transform), so this hook doesn't
        duplicate that path.
        """
        from torch.utils.data import DataLoader, Dataset, TensorDataset

        if isinstance(data, DataLoader):
            yield from data
            return

        if isinstance(data, Dataset):
            yield from DataLoader(data, batch_size=batch_size, shuffle=False)
            return

        if isinstance(data, (tuple, list)) and len(data) == 2:
            X, y = data
            X_t = self._to_tensor(X)
            y_t = self._to_target_tensor(y)
            dataset = TensorDataset(X_t.cpu(), y_t.cpu())
            for X_batch, y_batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
                if augment is not None:
                    X_batch, y_batch = augment(X_batch, y_batch)
                yield X_batch, y_batch
            return

        raise TypeError(
            f"TorchMLModel.iter_batches() expects a (X, y) pair or a "
            f"torch Dataset/DataLoader, got {type(data).__name__}."
        )

    def get_hyperparams(self) -> HyperparamSet:
        return HyperparamSet(
            params={"learning_rate": self._lr, "weight_decay": self._wd},
            episode_id=self._step_count,
        )

    def set_hyperparams(self, hyperparams: HyperparamSet) -> None:
        self._lr = hyperparams.params.get("learning_rate", self._lr)
        self._wd = hyperparams.params.get("weight_decay", self._wd)
        for pg in self._optimizer.param_groups:
            pg["lr"] = self._lr
            pg["weight_decay"] = self._wd

    def save_weights(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        self._torch.save({
            "model": self.model.state_dict(),
            "step": self._step_count,
            "lr": self._lr,
            "wd": self._wd,
        }, path)

    def load_weights(self, path: str) -> None:
        state = self._torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state["model"])
        self._step_count = state.get("step", 0)
        self._lr = state.get("lr", self._lr)
        self._wd = state.get("wd", self._wd)
        for pg in self._optimizer.param_groups:
            pg["lr"] = self._lr
            pg["weight_decay"] = self._wd

    def average_weights(self, paths: List[str]) -> None:
        if not paths:
            return
        states = [
            self._torch.load(p, map_location=self.device, weights_only=False)["model"]
            for p in paths
        ]
        avg = {k: self._torch.stack([s[k].float() for s in states]).mean(0) for k in states[0]}
        self.model.load_state_dict(avg)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _to_tensor(self, x):
        if isinstance(x, self._torch.Tensor):
            return x.to(self.device)
        return self._torch.as_tensor(np.asarray(x), dtype=self._torch.float32, device=self.device)

    def _to_target_tensor(self, y):
        t = y if isinstance(y, self._torch.Tensor) else self._torch.as_tensor(np.asarray(y))
        if t.dtype in (self._torch.int32, self._torch.int64):
            return t.to(device=self.device, dtype=self._torch.int64)
        return t.to(device=self.device, dtype=self._torch.float32)
