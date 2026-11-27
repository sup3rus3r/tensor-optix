"""
Gaussian naive Bayes and shrinkage LDA (classification landmarks).

* naive Bayes: per-class means and variances, features independent given the
  class (variance floor = 1e-6 * mean feature variance);
* LDA: shared covariance shrunk toward its diagonal (fixed shrinkage 0.1),
  a linear boundary that accounts for correlated features.

Both are fast landmarks whose relative performance hints at the shape of the
class boundaries; naive Bayes is also a valid (very simple) candidate.
"""
from __future__ import annotations

import numpy as np

from ..preprocess import PipelineOptions
from .base import CLASSIFICATION, Estimator, Family, register, softmax


class GaussianNB(Estimator):
    def __init__(self, n_classes: int):
        self.k = n_classes

    def fit(self, X, y, X_val=None, y_val=None):
        y = y.astype(int)
        p = X.shape[1]
        floor = 1e-6 * max(float(np.mean(np.var(X, axis=0))), 1e-12)
        self.mu = np.zeros((self.k, p))
        self.var = np.ones((self.k, p))
        counts = np.bincount(y, minlength=self.k).astype(np.float64)
        self.logprior = np.log((counts + 1.0) / (counts.sum() + self.k))
        for c in range(self.k):
            Xc = X[y == c]
            if len(Xc):
                self.mu[c] = Xc.mean(axis=0)
                self.var[c] = Xc.var(axis=0) + floor
            else:
                self.var[c] = np.var(X, axis=0) + floor
        return self

    def predict_proba(self, X):
        # class by class: an (n, k, p) intermediate would not fit in memory for large data
        ll = np.empty((len(X), self.k))
        for c in range(self.k):
            ll[:, c] = -0.5 * (np.sum(np.log(2 * np.pi * self.var[c]))
                               + np.sum((X - self.mu[c]) ** 2 / self.var[c], axis=1))
        return softmax(ll + self.logprior[None, :])


class LDA(Estimator):
    def __init__(self, n_classes: int, shrinkage: float = 0.1):
        self.k, self.shrink = n_classes, float(shrinkage)

    def fit(self, X, y, X_val=None, y_val=None):
        y = y.astype(int)
        n, p = X.shape
        counts = np.bincount(y, minlength=self.k).astype(np.float64)
        self.logprior = np.log((counts + 1.0) / (counts.sum() + self.k))
        self.mu = np.array([X[y == c].mean(axis=0) if counts[c] else X.mean(axis=0) for c in range(self.k)])
        R = X - self.mu[y]
        S = (R.T @ R) / max(n - self.k, 1)
        S = (1 - self.shrink) * S + self.shrink * np.diag(np.diag(S))
        S += 1e-6 * (np.trace(S) / max(p, 1) + 1e-12) * np.eye(p)
        self.Sinv_mu = np.linalg.solve(S, self.mu.T).T                # (k, p)
        self.const = -0.5 * np.sum(self.mu * self.Sinv_mu, axis=1) + self.logprior
        return self

    def predict_proba(self, X):
        return softmax(X @ self.Sinv_mu.T + self.const[None, :])


register(Family(
    name="naive_bayes", cost_rank=2, tasks=CLASSIFICATION,
    make=lambda cfg, seed, task, k: GaussianNB(k),
    configs=lambda n, p, task: [{}],
    preprocessing=PipelineOptions(scale="standard", categorical="onehot", high_cardinality="target"),
    landmark=True, candidate=True,
    description="Gaussian naive Bayes: features treated as independent given the class",
))

register(Family(
    name="lda", cost_rank=2, tasks=CLASSIFICATION,
    make=lambda cfg, seed, task, k: LDA(k),
    configs=lambda n, p, task: [{}],
    preprocessing=PipelineOptions(scale="standard", categorical="onehot", high_cardinality="target"),
    landmark=True, candidate=False,
    description="linear discriminant analysis with covariance shrinkage",
))
