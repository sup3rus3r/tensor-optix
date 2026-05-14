from __future__ import annotations

"""
MLAgent — a single BaseAgent adapter for all non-RL ML tasks.

Wraps any nn.Module and gives it the full tensor-optix feature set:
  - SPSA online hyperparameter tuning
  - Rollback on validation degradation
  - Convergence / plateau detection
  - Checkpointing (save / load weights)
  - Weight averaging (SWA)

Works for:
  - Supervised:         classification, regression
  - Unsupervised:       autoencoders, VAE
  - Self-supervised:    contrastive learning (SimCLR)
  - Anything with a loss: just pass loss= as a string or nn.Module

Usage::

    import tensor_optix as optix

    # Supervised
    model = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))
    opt = optix.Optimizer(model, train_dataset, loss="cross_entropy")
    opt.run()

    # Autoencoder
    opt = optix.Optimizer(autoencoder, train_dataset, loss="reconstruction")
    opt.run()

    # VAE — model must return (reconstruction, mu, logvar)
    opt = optix.Optimizer(vae, train_dataset, loss="vae")
    opt.run()

    # Custom loss
    opt = optix.Optimizer(model, train_dataset, loss=my_loss_fn)
    opt.run()
"""

import logging
import os
from typing import Any, Union

import torch
import torch.nn as nn

from tensor_optix.core.base_agent import BaseAgent
from tensor_optix.core.types import EpisodeData, HyperparamSet

logger = logging.getLogger(__name__)


class MLAgent(BaseAgent):
    """
    BaseAgent adapter for any nn.Module training task.

    Parameters
    ----------
    model:
        Any nn.Module. Forward pass must accept the input tensor produced by
        DatasetPipeline and return predictions compatible with loss_fn.
    loss_fn:
        Instantiated loss nn.Module. Build from a string via
        ``loss_registry.resolve_loss()``, or pass an nn.Module directly.
    learning_rate:
        Initial Adam learning rate. Tunable online via SPSA.
    weight_decay:
        Adam L2 regularisation. Tunable online via SPSA.
    max_grad_norm:
        Gradient clipping norm. Set to None to disable.
    device:
        "auto", "cpu", or "cuda".
    """

    is_on_policy = True  # safe for rollback — each batch is independent data

    default_param_bounds = {
        "learning_rate": (1e-5, 1e-2),
        "weight_decay":  (0.0,  1e-2),
    }
    default_log_params = ["learning_rate"]

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        max_grad_norm: float = 10.0,
        device: str = "auto",
    ) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.max_grad_norm = max_grad_norm

        _device = (
            ("cuda" if torch.cuda.is_available() else "cpu")
            if device == "auto" else device
        )
        self.device = torch.device(_device)
        self.model.to(self.device)
        self.loss_fn.to(self.device)

        self._lr = learning_rate
        self._wd = weight_decay
        self._optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self._step_count = 0

    # ------------------------------------------------------------------
    # BaseAgent contract
    # ------------------------------------------------------------------

    def act(self, observation) -> Any:
        """Inference — returns raw model output as a numpy array."""
        x = self._to_tensor(observation)
        with torch.no_grad():
            out = self.model(x)
        if isinstance(out, (tuple, list)):
            return tuple(o.cpu().numpy() for o in out)
        return out.cpu().numpy()

    def learn(self, episode_data: EpisodeData) -> dict:
        """
        One gradient update on the batch packed into episode_data.

        Mapping from EpisodeData fields:
          observations  → model input  (X)
          actions       → target       (y, or X again for reconstruction/VAE)

        For unsupervised tasks (reconstruction, VAE, contrastive) the pipeline
        sets actions = observations so the loss receives the original input as
        the target automatically.
        """
        X = self._to_tensor(episode_data.observations)
        y = self._to_target_tensor(episode_data.actions)

        self._optimizer.zero_grad()
        pred = self.model(X)
        loss = self.loss_fn(pred, y)
        loss.backward()

        if self.max_grad_norm is not None:
            grad_norm = nn.utils.clip_grad_norm_(
                self.model.parameters(), self.max_grad_norm
            )
        else:
            grad_norm = 0.0

        self._optimizer.step()
        self._step_count += 1

        return {
            "loss":      loss.item(),
            "grad_norm": float(grad_norm),
        }

    def get_hyperparams(self) -> HyperparamSet:
        return HyperparamSet(
            params={
                "learning_rate": self._lr,
                "weight_decay":  self._wd,
            },
            episode_id=self._step_count,
        )

    def set_hyperparams(self, hyperparams: HyperparamSet) -> None:
        self._lr = hyperparams.params.get("learning_rate", self._lr)
        self._wd = hyperparams.params.get("weight_decay",  self._wd)
        for pg in self._optimizer.param_groups:
            pg["lr"]           = self._lr
            pg["weight_decay"] = self._wd

    def save_weights(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save({
            "model":   self.model.state_dict(),
            "step":    self._step_count,
            "lr":      self._lr,
            "wd":      self._wd,
        }, path)

    def load_weights(self, path: str) -> None:
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state["model"])
        self._step_count = state.get("step", 0)
        self._lr = state.get("lr", self._lr)
        self._wd = state.get("wd", self._wd)
        for pg in self._optimizer.param_groups:
            pg["lr"]           = self._lr
            pg["weight_decay"] = self._wd

    def perturb_weights(self, noise_scale: float) -> None:
        with torch.no_grad():
            for p in self.model.parameters():
                p.mul_(1 + noise_scale * torch.randn_like(p))

    def average_weights(self, paths: list) -> None:
        if not paths:
            return
        states = [
            torch.load(p, map_location=self.device, weights_only=False)["model"]
            for p in paths
        ]
        avg = {
            k: torch.stack([s[k].float() for s in states]).mean(0)
            for k in states[0]
        }
        self.model.load_state_dict(avg)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _to_tensor(self, x) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.to(self.device)
        return torch.as_tensor(x, dtype=torch.float32, device=self.device)

    def _to_target_tensor(self, y) -> torch.Tensor:
        """Convert target to tensor, preserving integer dtypes for CrossEntropyLoss etc."""
        t = torch.as_tensor(y) if not isinstance(y, torch.Tensor) else y
        if t.dtype in (torch.int32, torch.int64):
            return t.to(device=self.device, dtype=torch.int64)
        return t.to(device=self.device, dtype=torch.float32)
