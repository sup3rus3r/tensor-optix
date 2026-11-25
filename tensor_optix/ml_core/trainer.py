"""
tensor_optix.ml_core.trainer — the ML-native training loop, analogous to
LoopController but genuinely epoch/batch-shaped instead of episode-shaped,
and with no RL-specific concerns (on-policy/off-policy rollback gating,
SPSA-probe suppression, RND eta updates -- none of these have an ML
analogue; see this package's module docs for the full reusable/new split).

Reuses, unmodified, from the RL core:
  - CheckpointRegistry   (core/checkpoint_registry.py)  -- save/load_best
  - ConvergenceMonitor   (core/convergence.py)          -- convergence / stop
  - SPSAOptimizer        (optimizers/spsa_optimizer.py)  -- hyperparameter tuning
All three operate purely on scores / HyperparamSet, so none of them need to
know this is an ML trainer rather than an RL loop.
"""
from __future__ import annotations

import logging
import warnings
from typing import List, Optional

import numpy as np

from tensor_optix.core.checkpoint_registry import CheckpointRegistry
from tensor_optix.core.convergence import ConvergenceMonitor
from tensor_optix.core.types import HyperparamSet

logger = logging.getLogger(__name__)

# ConvergenceMonitor settings for per-epoch validation scores. Validation
# metrics are far smoother than per-episode RL returns and a run is tens of
# epochs, not thousands of episodes, so every window is short. Patience keeps
# the historical ML values: COOLING after 5 epochs without new ground,
# DORMANT (stop) after 20.
ML_CONVERGENCE_DEFAULTS = dict(
    timescales=(3, 8, 24),
    cooling_patience=5,
    dormant_patience=20,
    persist=2,
)


def _call(cb, name, *args) -> None:
    fn = getattr(cb, name, None)
    if callable(fn):
        fn(*args)


class Trainer:
    """
    Drives BaseMLModel.fit()'s actual epoch loop. Not part of the public
    BaseMLModel API -- constructed and run internally by fit(); see that
    method's docstring.

    Stopping, in order of precedence:
      * ``epochs`` reached (when given),
      * ``max_epochs`` reached (when given; an optional cap, off by default),
      * ``early_stopping_patience`` epochs without a new best (when given),
      * the ConvergenceMonitor reaching DORMANT (when ``stop_on_convergence``
        and validation data are given).
    The best checkpoint's weights are restored before returning, however the
    run ended.
    """

    def __init__(
        self,
        model,
        epochs: Optional[int],
        batch_size: int,
        early_stopping_patience: Optional[int],
        metrics,
        callbacks: Optional[List],
        checkpoint_dir: Optional[str],
        lr_schedule: Optional[object] = None,
        augment: Optional[object] = None,
        max_epochs: Optional[int] = None,
        stop_on_convergence: bool = True,
        convergence: Optional[ConvergenceMonitor] = None,
    ):
        if epochs is not None and int(epochs) < 0:
            raise ValueError(f"epochs must be >= 0 or None, got {epochs}")
        if max_epochs is not None and int(max_epochs) < 0:
            raise ValueError(f"max_epochs must be >= 0 or None, got {max_epochs}")
        self._model = model
        self._epochs = None if epochs is None else int(epochs)
        self._max_epochs = None if max_epochs is None else int(max_epochs)
        self._batch_size = batch_size
        self._early_stopping_patience = early_stopping_patience
        self._metrics = metrics
        self._callbacks = callbacks or []
        self._checkpoint_dir = checkpoint_dir
        self._lr_schedule = lr_schedule
        self._augment = augment
        self._stop_on_convergence = bool(stop_on_convergence)

        self._monitor = convergence or ConvergenceMonitor(**ML_CONVERGENCE_DEFAULTS)
        self._registry = (
            CheckpointRegistry(checkpoint_dir) if checkpoint_dir is not None else None
        )

        self._spsa = None
        param_bounds = dict(getattr(model, "default_param_bounds", None) or {})
        if lr_schedule is not None:
            # A schedule and SPSA both mutating learning_rate every epoch
            # would fight each other -- the schedule owns learning_rate
            # for the whole run; SPSA still tunes any other bounded param.
            param_bounds.pop("learning_rate", None)
        if param_bounds:
            from tensor_optix.optimizers.spsa_optimizer import SPSAOptimizer
            self._spsa = SPSAOptimizer(
                param_bounds=param_bounds,
                log_params=[p for p in getattr(model, "default_log_params", []) if p in param_bounds],
            )

    @property
    def monitor(self) -> ConvergenceMonitor:
        return self._monitor

    def _check_open_ended(self, val_data) -> None:
        if self._epochs is not None:
            return
        if val_data is None and self._max_epochs is None:
            raise ValueError(
                "epochs=None trains until convergence, which is judged on "
                "validation scores: pass val_data (or set max_epochs)."
            )
        if val_data is not None and not self._stop_on_convergence and (
            self._early_stopping_patience is None and self._max_epochs is None
        ):
            raise ValueError(
                "epochs=None with stop_on_convergence=False and no "
                "early_stopping_patience / max_epochs would never stop."
            )
        sched_epochs = getattr(self._lr_schedule, "epochs", None)
        if sched_epochs is not None:
            warnings.warn(
                f"lr_schedule has a fixed length ({sched_epochs} epochs) but "
                "epochs=None: the learning rate reaches its final value at "
                f"epoch {sched_epochs} and stays there.",
                UserWarning, stacklevel=3,
            )

    def run(self, train_data, val_data) -> "FitHistory":
        from tensor_optix.ml_core.base_model import FitHistory

        self._check_open_ended(val_data)
        history = FitHistory()
        epochs_without_improvement = 0
        best_score_seen: Optional[float] = None
        metrics_history = []  # List[EvalMetrics], grows every eval -- SPSAOptimizer.suggest()'s contract (same as LoopController's usage)

        epoch = 0
        while True:
            if self._epochs is not None and epoch >= self._epochs:
                history.stop_reason = "epochs"
                break
            if self._max_epochs is not None and epoch >= self._max_epochs:
                history.stop_reason = "max_epochs"
                break

            if self._lr_schedule is not None:
                current = self._model.get_hyperparams()
                new_params = dict(current.params)
                new_params["learning_rate"] = self._lr_schedule(epoch)
                self._model.set_hyperparams(
                    HyperparamSet(params=new_params, episode_id=current.episode_id)
                )

            epoch_loss = self._run_train_epoch(train_data)
            history.train_loss.append(epoch_loss)
            history.epochs_run = epoch + 1

            for cb in self._callbacks:
                _call(cb, "on_epoch_end", epoch, epoch_loss)

            if val_data is None:
                epoch += 1
                continue

            eval_metrics = self._model.evaluate(val_data)
            # loss isn't known to evaluate() unless the model records it --
            # fall back to the epoch's own train loss when evaluate() left
            # it at its dataclass default (0.0), so FitHistory/logging still
            # show a meaningful number rather than silently zero.
            if eval_metrics.loss == 0.0 and epoch_loss != 0.0:
                eval_metrics.loss = epoch_loss
            history.val_metrics.append(eval_metrics)

            # "New best" (the checkpoint decision) is judged against the best
            # score observed so far THIS run, independent of whether
            # checkpointing is enabled -- checkpoint_dir=None must still let
            # early stopping work; it only disables persisting weights.
            is_new_best = best_score_seen is None or eval_metrics.primary_score > best_score_seen

            if is_new_best:
                best_score_seen = eval_metrics.primary_score
                epochs_without_improvement = 0
                history.best_epoch = epoch
                if self._registry is not None:
                    self._registry.save(
                        self._model, eval_metrics.to_eval_metrics(),
                        self._model.get_hyperparams(),
                    )
                for cb in self._callbacks:
                    _call(cb, "on_improvement", eval_metrics)
            else:
                epochs_without_improvement += 1
                for cb in self._callbacks:
                    _call(cb, "on_no_improvement", eval_metrics)

            # Convergence: what the validation curve is doing, and whether
            # the run has stopped making real progress.
            report = self._monitor.observe(eval_metrics.primary_score, at=epoch)
            history.convergence.append(report)
            if report.regime_changed or report.entered_cooling or report.entered_dormant:
                for cb in self._callbacks:
                    _call(cb, "on_convergence", epoch, report)
            if report.improved and self._spsa is not None:
                self._spsa.on_improvement(eval_metrics.to_eval_metrics())

            metrics_history.append(eval_metrics.to_eval_metrics())
            if report.entered_cooling:
                if self._spsa is not None:
                    self._spsa.on_cooling(metrics_history)
                for cb in self._callbacks:
                    _call(cb, "on_plateau", eval_metrics)
            if self._spsa is not None:
                suggestion = self._spsa.suggest(
                    self._model.get_hyperparams(), metrics_history,
                )
                self._model.set_hyperparams(suggestion)

            if (
                self._early_stopping_patience is not None
                and epochs_without_improvement >= self._early_stopping_patience
            ):
                logger.info(
                    "Early stopping at epoch %d: no improvement for %d epochs.",
                    epoch, epochs_without_improvement,
                )
                history.stopped_early = True
                history.stop_reason = "early_stopping"
                break

            if report.entered_dormant:
                for cb in self._callbacks:
                    _call(cb, "on_dormant", epoch)
                if self._stop_on_convergence:
                    logger.info(
                        "Converged at epoch %d (%s): no new ground for %d epochs.",
                        epoch, report.regime.name, int(report.since_improvement),
                    )
                    history.stopped_early = True
                    history.stop_reason = "converged"
                    break

            epoch += 1

        if self._registry is not None and self._registry.best is not None:
            self._registry.load_best(self._model)

        return history

    def _run_train_epoch(self, train_data) -> float:
        total_loss = 0.0
        n_batches = 0
        for batch in self._iter_batches(self._shuffled(train_data)):
            loss = self._model.train_step(batch)
            total_loss += float(loss)
            n_batches += 1
        return total_loss / max(n_batches, 1)

    @staticmethod
    def _shuffled(data):
        """
        Re-order a raw (X, y) pair with a fresh random permutation each epoch.

        The adapters' iter_batches() walk (X, y) in storage order, so without
        this every epoch sees identical minibatches — and data stored sorted
        by class would train on single-class batches. Native datasets
        (DataLoader / tf.data.Dataset) are passed through untouched: they
        carry their own shuffle configuration.
        """
        if not (isinstance(data, (tuple, list)) and len(data) == 2):
            return data
        X, y = data
        try:
            n = len(X)
            if n != len(y) or n < 2:
                return data
        except TypeError:
            return data
        perm = np.random.permutation(n)

        def _take(a):
            if isinstance(a, np.ndarray):
                return a[perm]
            if type(a).__module__.split(".")[0] == "torch":
                return a[perm]  # torch tensors accept an int64 ndarray index
            return np.asarray(a)[perm]

        return (_take(X), _take(y))

    def _iter_batches(self, data):
        """
        Yield batches from train_data. Delegates entirely to the model
        adapter's own batch iterator (TorchMLModel/TFMLModel each know how
        to turn a (X, y) pair or a native Dataset/DataLoader/tf.data.Dataset
        into framework-appropriate batches) -- Trainer itself stays
        framework-agnostic and never inspects batch contents, per the
        plan's "both input forms, auto-detected" decision.
        """
        return self._model.iter_batches(data, self._batch_size, augment=self._augment)
