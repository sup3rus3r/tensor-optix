"""Dummy baseline: class priors (classification) or the training mean
(regression). Every family must beat this to show any learnable signal."""
from __future__ import annotations

import numpy as np

from ..preprocess import PipelineOptions
from .base import CLASSIFICATION, REGRESSION, Estimator, Family, register


class Dummy(Estimator):
    def __init__(self, task: str, n_classes: int):
        self.task, self.k = task, n_classes

    def fit(self, X, y, X_val=None, y_val=None):
        if self.task in CLASSIFICATION:
            counts = np.bincount(y.astype(int), minlength=self.k).astype(np.float64)
            self.prior = (counts + 1.0) / (counts.sum() + self.k)      # Laplace: never exactly 0
        else:
            self.mean = float(np.mean(y))
        return self

    def predict_proba(self, X):
        return np.tile(self.prior, (len(X), 1))

    def predict(self, X):
        return np.full(len(X), self.mean)


register(Family(
    name="baseline", cost_rank=0, tasks=CLASSIFICATION + REGRESSION,
    make=lambda cfg, seed, task, k: Dummy(task, k),
    configs=lambda n, p, task: [{}],
    preprocessing=PipelineOptions(scale="none", categorical="ordinal", high_cardinality="ordinal"),
    landmark=True, candidate=False,
    description="predicts the class balance / the mean; the floor every family must beat",
))
