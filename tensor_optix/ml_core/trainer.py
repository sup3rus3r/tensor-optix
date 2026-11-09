"""
tensor_optix.ml_core.trainer — the ML-native training loop, analogous to
LoopController but genuinely epoch/batch-shaped instead of episode-shaped,
and with no RL-specific concerns (on-policy/off-policy rollback gating,
SPSA-probe suppression, RND eta updates -- none of these have an ML
analogue; see this package's module docs for the full reusable/new split).

Reuses, unmodified, from the RL core:
  - CheckpointRegistry   (core/checkpoint_registry.py)  -- save/load_best
  - BackoffScheduler     (core/backoff_scheduler.py)     -- plateau/early-stop
  - SPSAOptimizer        (optimizers/spsa_optimizer.py)  -- hyperparameter tuning
All three operate purely on EvalMetrics.primary_score / HyperparamSet, so
none of them need to know this is an ML trainer rather than an RL loop.
"""
from __future__ import annotations

import logging
from typing import List, Optional

from tensor_optix.core.backoff_scheduler import BackoffScheduler
from tensor_optix.core.checkpoint_registry import CheckpointRegistry
from tensor_optix.core.types import HyperparamSet

logger = logging.getLogger(__name__)

# BackoffScheduler defaults tuned for RL's noisy per-episode reward signal
# (see backoff_scheduler.py's own defaults: trend_window=8, score_window=20,
# noise_k=2.0). Per-epoch validation metrics are evaluated far less often
# and are typically much smoother than per-episode RL reward, so:
#   - trend_window/score_window are smaller (fewer epochs needed before the
#     scheduler has enough history to judge a trend at all -- an ML run
#     might only train for 10-30 epochs total, where RL runs for thousands
#     of episodes).
#   - noise_k is smaller (less inflation of the adaptive floor against
#     eval-to-eval noise, since that noise is genuinely lower for a
#     deterministic validation pass over a fixed held-out set than for a
#     stochastic RL rollout).
# plateau_threshold/dormant_threshold are left at the RL defaults (5/20)
# since both are already epoch/episode-count-agnostic -- they count
# *consecutive non-improving evaluations*, which is the right unit
# regardless of what an "evaluation" is. Flagged in the plan for real-
# training-curve verification, not assumed correct from first principles.
ML_BACKOFF_DEFAULTS = dict(
    trend_window=4,
    score_window=8,
    noise_k=1.0,
    plateau_threshold=5,
    dormant_threshold=20,
    min_episodes_before_degradation=3,
)


class Trainer:
    """
    Drives BaseMLModel.fit()'s actual epoch loop. Not part of the public
    BaseMLModel API -- constructed and run internally by fit(); see that
    method's docstring.
    """

    def __init__(
        self,
        model,
        epochs: int,
        batch_size: int,
        early_stopping_patience: Optional[int],
        metrics,
        callbacks: Optional[List],
        checkpoint_dir: Optional[str],
    ):
        self._model = model
        self._epochs = epochs
        self._batch_size = batch_size
        self._early_stopping_patience = early_stopping_patience
        self._metrics = metrics
        self._callbacks = callbacks or []
        self._checkpoint_dir = checkpoint_dir

        self._scheduler = BackoffScheduler(**ML_BACKOFF_DEFAULTS)
        self._registry = (
            CheckpointRegistry(checkpoint_dir) if checkpoint_dir is not None else None
        )

        self._spsa = None
        if getattr(model, "default_param_bounds", None):
            from tensor_optix.optimizers.spsa_optimizer import SPSAOptimizer
            self._spsa = SPSAOptimizer(
                param_bounds=model.default_param_bounds,
                log_params=getattr(model, "default_log_params", []),
            )

    def run(self, train_data, val_data) -> "FitHistory":
        from tensor_optix.ml_core.base_model import FitHistory

        history = FitHistory()
        epochs_without_improvement = 0
        best_score_seen: Optional[float] = None
        metrics_history = []  # List[EvalMetrics], grows every eval -- SPSAOptimizer.suggest()'s contract (same as LoopController's usage)

        for epoch in range(self._epochs):
            epoch_loss = self._run_train_epoch(train_data)
            history.train_loss.append(epoch_loss)
            history.epochs_run = epoch + 1

            for cb in self._callbacks:
                if hasattr(cb, "on_epoch_end"):
                    cb.on_epoch_end(epoch, epoch_loss)

            if val_data is None:
                continue

            eval_metrics = self._model.evaluate(val_data)
            # loss isn't known to evaluate() unless the model records it --
            # fall back to the epoch's own train loss when evaluate() left
            # it at its dataclass default (0.0), so FitHistory/logging still
            # show a meaningful number rather than silently zero.
            if eval_metrics.loss == 0.0 and epoch_loss != 0.0:
                eval_metrics.loss = epoch_loss
            history.val_metrics.append(eval_metrics)

            # "New best" is judged against the best score observed so far
            # THIS run, independent of whether checkpointing is enabled --
            # checkpoint_dir=None must still let early stopping/plateau
            # detection work correctly; it only disables persisting weights
            # to disk, not the Trainer's own notion of improvement.
            is_new_best = best_score_seen is None or eval_metrics.primary_score > best_score_seen

            if is_new_best:
                best_score_seen = eval_metrics.primary_score
                self._scheduler.record_improvement(eval_metrics.primary_score)
                epochs_without_improvement = 0
                history.best_epoch = epoch
                if self._registry is not None:
                    self._registry.save(
                        self._model, eval_metrics.to_eval_metrics(),
                        self._model.get_hyperparams(),
                    )
                for cb in self._callbacks:
                    if hasattr(cb, "on_improvement"):
                        cb.on_improvement(eval_metrics)
            else:
                self._scheduler.record_non_improvement()
                epochs_without_improvement += 1
                for cb in self._callbacks:
                    if hasattr(cb, "on_plateau"):
                        cb.on_plateau(eval_metrics)

            metrics_history.append(eval_metrics.to_eval_metrics())
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
                break

        if self._registry is not None and self._registry.best is not None:
            self._registry.load_best(self._model)

        return history

    def _run_train_epoch(self, train_data) -> float:
        total_loss = 0.0
        n_batches = 0
        for batch in self._iter_batches(train_data):
            loss = self._model.train_step(batch)
            total_loss += float(loss)
            n_batches += 1
        return total_loss / max(n_batches, 1)

    def _iter_batches(self, data):
        """
        Yield batches from train_data. Delegates entirely to the model
        adapter's own batch iterator (TorchMLModel/TFMLModel each know how
        to turn a (X, y) pair or a native Dataset/DataLoader/tf.data.Dataset
        into framework-appropriate batches) -- Trainer itself stays
        framework-agnostic and never inspects batch contents, per the
        plan's "both input forms, auto-detected" decision.
        """
        return self._model.iter_batches(data, self._batch_size)
