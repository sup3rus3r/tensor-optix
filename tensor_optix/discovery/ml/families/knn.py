"""
k-nearest neighbours on standardised features. A landmark for "local
similarity" structure and a candidate for small, low-dimensional data.

Training rows are subsampled to at most ``max_train`` (seeded) so the cost
stays bounded; distances are computed in blocks.
"""
from __future__ import annotations

import numpy as np

from ...rng import rng_for
from ..preprocess import PipelineOptions
from .base import CLASSIFICATION, REGRESSION, Estimator, Family, register

_BLOCK = 2048


class KNN(Estimator):
    def __init__(self, k: int, task: str, n_classes: int, seed: int, max_train: int = 5000):
        self.k, self.task, self.nc, self.seed, self.max_train = int(k), task, n_classes, seed, max_train

    def fit(self, X, y, X_val=None, y_val=None):
        if len(X) > self.max_train:
            pick = rng_for(self.seed, "knn_subsample").choice(len(X), self.max_train, replace=False)
            X, y = X[pick], y[pick]
        self.X = X.astype(np.float64)
        self.y = y
        self.sq = np.sum(self.X * self.X, axis=1)
        self.kk = max(1, min(self.k, len(self.X)))
        return self

    def _neighbours(self, X):
        out = np.empty((len(X), self.kk), dtype=np.int64)
        for s in range(0, len(X), _BLOCK):
            Q = X[s:s + _BLOCK].astype(np.float64)
            d = np.sum(Q * Q, axis=1)[:, None] - 2.0 * Q @ self.X.T + self.sq[None, :]
            if self.kk < self.X.shape[0]:
                part = np.argpartition(d, self.kk - 1, axis=1)[:, :self.kk]
            else:
                part = np.tile(np.arange(self.X.shape[0]), (len(Q), 1))
            out[s:s + _BLOCK] = part
        return out

    def predict_proba(self, X):
        nb = self._neighbours(X)
        counts = np.zeros((len(X), self.nc))
        lab = self.y.astype(int)[nb]
        for c in range(self.nc):
            counts[:, c] = np.sum(lab == c, axis=1)
        return (counts + 0.5) / (self.kk + 0.5 * self.nc)          # smoothed: never 0 / 1

    def predict(self, X):
        return np.mean(self.y.astype(np.float64)[self._neighbours(X)], axis=1)


def _configs(n, p, task):
    k = int(np.clip(round(np.sqrt(max(n, 1)) / 2), 5, 50))
    return [{"k": k}]


register(Family(
    name="knn", cost_rank=3, tasks=CLASSIFICATION + REGRESSION,
    make=lambda cfg, seed, task, k: KNN(int(cfg.get("k", 15)), task, k, seed),
    configs=_configs,
    preprocessing=PipelineOptions(scale="standard", categorical="onehot", high_cardinality="target"),
    landmark=True, candidate=True,
    description="k-nearest neighbours: local similarity, no global form assumed",
))
