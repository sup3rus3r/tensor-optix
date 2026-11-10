"""
tensor_optix.ml_core.lr_schedules — per-epoch learning-rate schedules for
BaseMLModel.fit(). Each schedule is a plain callable: schedule(epoch) ->
float, called by Trainer once per epoch (epoch is 0-indexed) before that
epoch's training batches run.

Mutually exclusive with SPSA tuning of learning_rate within the same
fit() call: when lr_schedule is passed, Trainer drops "learning_rate" from
the bounds it hands SPSAOptimizer, since a schedule and a score-driven
probe both trying to own the same hyperparameter would fight each other
every epoch. weight_decay and any other SPSA-tunable hyperparameter are
unaffected.
"""
from __future__ import annotations

import math
from typing import Optional


class StepLR:
    """Multiplies base_lr by gamma every step_size epochs."""

    def __init__(self, base_lr: float, step_size: int, gamma: float = 0.1):
        self.base_lr = base_lr
        self.step_size = step_size
        self.gamma = gamma

    def __call__(self, epoch: int) -> float:
        return self.base_lr * (self.gamma ** (epoch // self.step_size))


class CosineAnnealingLR:
    """Cosine decay from base_lr to min_lr over `epochs` total epochs."""

    def __init__(self, base_lr: float, epochs: int, min_lr: float = 0.0):
        self.base_lr = base_lr
        self.epochs = epochs
        self.min_lr = min_lr

    def __call__(self, epoch: int) -> float:
        progress = min(epoch, self.epochs) / max(self.epochs, 1)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return self.min_lr + (self.base_lr - self.min_lr) * cosine


class WarmupLR:
    """
    Linear warmup from 0 to base_lr over warmup_epochs, then delegates to
    `after` (any other schedule, e.g. CosineAnnealingLR) for every epoch
    from warmup_epochs onward. `after` is called with the epoch re-based
    to 0 at the point warmup ends, so e.g.
    WarmupLR(1e-3, warmup_epochs=5, after=CosineAnnealingLR(1e-3, epochs=45))
    warms up for epochs 0-4, then anneals over its own epochs 0-44 (global
    epochs 5-49).
    """

    def __init__(self, base_lr: float, warmup_epochs: int, after: Optional[object] = None):
        self.base_lr = base_lr
        self.warmup_epochs = warmup_epochs
        self.after = after

    def __call__(self, epoch: int) -> float:
        if epoch < self.warmup_epochs:
            return self.base_lr * (epoch + 1) / max(self.warmup_epochs, 1)
        if self.after is None:
            return self.base_lr
        return self.after(epoch - self.warmup_epochs)
