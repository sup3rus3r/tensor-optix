"""
tensor_optix.ml_core.base_model — BaseMLModel: the ML-native training
contract, genuinely separate from BaseAgent's RL episode/reward contract.

No act()/learn()/is_on_policy/average_weights() RL vocabulary here --
fit()/predict()/evaluate() instead, matching sklearn's convention, which is
what the vast majority of ML practitioners already expect. The per-epoch
training loop fit() runs is an internal implementation detail (built on
tensor_optix.ml_core.trainer.Trainer), not an object the caller drives
step by step.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from tensor_optix.ml_core.metrics import MLMetrics


@dataclass
class FitHistory:
    """Per-epoch record of training, returned by fit()."""
    train_loss: List[float] = field(default_factory=list)
    val_metrics: List[MLMetrics] = field(default_factory=list)
    best_epoch: Optional[int] = None
    stopped_early: bool = False
    epochs_run: int = 0
    # Why the run ended: "epochs", "max_epochs", "early_stopping" or
    # "converged" (the ConvergenceMonitor reached DORMANT).
    stop_reason: Optional[str] = None
    # One ConvergenceReport per validated epoch.
    convergence: List[Any] = field(default_factory=list)


class BaseMLModel(ABC):
    """
    The only contract between a trained ML model and tensor-optix's
    ML-native checkpointing/metrics machinery (CheckpointRegistry,
    ConvergenceMonitor, SPSAOptimizer -- all reused as-is from the RL core,
    since they are genuinely algorithm-agnostic; see ml_core's module docs
    for which RL-core pieces are and are not reused).
    """

    @abstractmethod
    def predict(self, X: Any) -> Any:
        """Run inference on a batch of inputs, returning model outputs
        (e.g. logits for classification, raw values for regression).
        Must not update weights."""

    @abstractmethod
    def train_step(self, batch: Any) -> float:
        """
        One gradient step on a single batch. Returns the scalar loss value
        for that batch (a Python float, not a framework tensor).

        batch shape is framework-adapter-specific (e.g. a (X, y) tuple of
        tensors) -- Trainer never inspects batch contents itself, only
        passes through whatever the data iterator yields.
        """

    @abstractmethod
    def evaluate(self, data) -> MLMetrics:
        """
        Run inference over an entire dataset/loader and compute MLMetrics
        (via tensor_optix.ml_core.metrics.resolve_metrics()).
        """

    @abstractmethod
    def iter_batches(self, data, batch_size: int):
        """
        Yield framework-appropriate batches from `data`, which may be a
        raw (X, y) array pair or a native Dataset/DataLoader/tf.data.Dataset
        (see this package's "Data input" design note) -- auto-detected and
        normalized here so Trainer itself never needs to know which form
        was passed. Each yielded batch is whatever train_step()/evaluate()
        expect to receive.
        """

    @abstractmethod
    def save_weights(self, path: str) -> None:
        """Persist model weights to the given path."""

    @abstractmethod
    def load_weights(self, path: str) -> None:
        """Restore model weights from the given path."""

    def get_hyperparams(self):
        """Current hyperparameters, for SPSA tuning -- default: no tunable
        hyperparameters. Override to enable online SPSA tuning during
        fit(), matching MLAgent's existing, already-proven pattern."""
        from tensor_optix.core.types import HyperparamSet
        return HyperparamSet(params={}, episode_id=0)

    def set_hyperparams(self, hyperparams) -> None:
        """Apply new hyperparameters. Default: no-op (see get_hyperparams)."""

    default_param_bounds: Dict[str, tuple] = {}
    default_log_params: List[str] = []

    def average_weights(self, paths: List[str]) -> None:
        """Stochastic Weight Averaging across checkpoints. Default: no-op.
        Override in framework-specific adapters (see MLAgent's version,
        reused as the implementation pattern)."""

    def fit(
        self,
        train_data,
        val_data=None,
        epochs: Optional[int] = 10,
        batch_size: int = 32,
        early_stopping_patience: Optional[int] = None,
        metrics="auto",
        callbacks: Optional[List] = None,
        checkpoint_dir: Optional[str] = None,
        lr_schedule: Optional[object] = None,
        augment: Optional[object] = None,
        max_epochs: Optional[int] = None,
        stop_on_convergence: bool = True,
        convergence: Optional[object] = None,
    ) -> FitHistory:
        """
        Train for up to `epochs` epochs, or -- with ``epochs=None`` -- until
        tensor-optix judges the run converged. The best checkpoint's weights
        are restored before returning, however the run ended.

        epochs: a fixed budget, or None to train until convergence (requires
        val_data, or a max_epochs cap).

        max_epochs: optional hard cap for an open-ended run. Off by default.

        early_stopping_patience: stop once the validation primary_score hasn't
        set a new best for that many consecutive epochs. Off by default.

        stop_on_convergence: when val_data is given, stop as soon as the
        ConvergenceMonitor reaches DORMANT (no new ground for its dormant
        patience, and no slow rise still in progress). On by default.
        FitHistory.stop_reason says which rule ended the run, and
        FitHistory.convergence holds one ConvergenceReport per epoch.

        convergence: a tensor_optix.core.convergence.ConvergenceMonitor to use
        instead of the ML defaults (to change its timescales, patience or
        min_delta).

        lr_schedule: an optional callable(epoch) -> float (see
        tensor_optix.ml_core.lr_schedules for StepLR/CosineAnnealingLR/
        WarmupLR), applied once per epoch before that epoch's training
        batches run. When set, learning_rate is no longer handed to SPSA
        for online tuning during this fit() call -- the schedule owns it.

        augment: an optional callable(X_batch, y_batch) -> (X_batch, y_batch),
        applied to every training batch (not validation). Only takes
        effect when train_data is a raw (X, y) array pair -- a native
        Dataset/DataLoader/tf.data.Dataset already has its own place for
        augmentation (a transform or .map()), so this hook doesn't
        duplicate that path.

        See tensor_optix.ml_core.trainer.Trainer for the actual loop --
        this method only constructs and runs it, matching the sklearn-style
        "the loop is an implementation detail" decision.
        """
        from tensor_optix.ml_core.trainer import Trainer

        trainer = Trainer(
            model=self,
            epochs=epochs,
            batch_size=batch_size,
            early_stopping_patience=early_stopping_patience,
            metrics=metrics,
            callbacks=callbacks,
            checkpoint_dir=checkpoint_dir,
            lr_schedule=lr_schedule,
            augment=augment,
            max_epochs=max_epochs,
            stop_on_convergence=stop_on_convergence,
            convergence=convergence,
        )
        return trainer.run(train_data, val_data)
