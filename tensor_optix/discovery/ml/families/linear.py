"""
Linear family: ridge regression (closed form) and L2-regularised logistic
regression (Newton / IRLS, one-vs-rest for multiclass).

Inputs are the standardised matrices from ``ColumnPipeline``; the intercept is
fitted separately and not penalised. ``alpha`` is the L2 strength per row
(the penalty is ``alpha * n``), so its meaning does not change with sample size.
"""
from __future__ import annotations

import numpy as np

from ..preprocess import PipelineOptions
from .base import CLASSIFICATION, REGRESSION, Estimator, Family, register, sigmoid

_NEWTON_MAX_FEATURES = 600


class Ridge(Estimator):
    def __init__(self, alpha: float = 1e-3):
        self.alpha = float(alpha)

    def fit(self, X, y, X_val=None, y_val=None):
        n, p = X.shape
        self.x_mean = X.mean(axis=0)
        self.y_mean = float(np.mean(y))
        Xc = X - self.x_mean
        yc = y.astype(np.float64) - self.y_mean
        A = Xc.T @ Xc + self.alpha * n * np.eye(p)
        self.w = np.linalg.solve(A, Xc.T @ yc) if p else np.zeros(0)
        return self

    def predict(self, X):
        return (X - self.x_mean) @ self.w + self.y_mean


def _logistic_binary(X: np.ndarray, y: np.ndarray, lam: float, max_iter: int = 50) -> np.ndarray:
    """Returns [w..., b]; penalty lam on w only."""
    n, p = X.shape
    Xb = np.hstack([X, np.ones((n, 1))])
    w = np.zeros(p + 1)
    pen = np.full(p + 1, lam)
    pen[-1] = 0.0
    if p + 1 <= _NEWTON_MAX_FEATURES:
        for _ in range(max_iter):
            mu = sigmoid(Xb @ w)
            g = Xb.T @ (mu - y) + pen * w
            s = mu * (1 - mu)
            H = (Xb * s[:, None]).T @ Xb + np.diag(pen) + 1e-10 * np.eye(p + 1)
            step = np.linalg.solve(H, g)
            w -= step
            if np.max(np.abs(step)) < 1e-6:
                break
        return w
    # Many features: full-batch gradient descent with a safe step (Lipschitz bound).
    L = 0.25 * float(np.linalg.norm(Xb, 2)) ** 2 + lam
    lr = 1.0 / max(L, 1e-12)
    for _ in range(2000):
        mu = sigmoid(Xb @ w)
        g = Xb.T @ (mu - y) + pen * w
        w -= lr * g
        if np.max(np.abs(lr * g)) < 1e-7:
            break
    return w


class Logistic(Estimator):
    def __init__(self, alpha: float, n_classes: int):
        self.alpha, self.k = float(alpha), n_classes

    def fit(self, X, y, X_val=None, y_val=None):
        n = len(X)
        lam = self.alpha * n
        y = y.astype(int)
        if self.k == 2:
            self.W = [_logistic_binary(X, (y == 1).astype(np.float64), lam)]
        else:
            self.W = [_logistic_binary(X, (y == c).astype(np.float64), lam) for c in range(self.k)]
        return self

    def predict_proba(self, X):
        Xb = np.hstack([X, np.ones((len(X), 1))])
        if self.k == 2:
            p1 = sigmoid(Xb @ self.W[0])
            return np.column_stack([1 - p1, p1])
        P = np.column_stack([sigmoid(Xb @ w) for w in self.W])
        P = np.clip(P, 1e-12, None)
        return P / P.sum(axis=1, keepdims=True)


def _make(cfg, seed, task, k):
    alpha = cfg.get("alpha", 1e-3)
    if task in CLASSIFICATION:
        return Logistic(alpha, k)
    return Ridge(alpha)


register(Family(
    name="linear", cost_rank=1, tasks=CLASSIFICATION + REGRESSION, make=_make,
    configs=lambda n, p, task: [{"alpha": 1e-3}, {"alpha": 1e-2}],
    preprocessing=PipelineOptions(scale="standard", categorical="onehot", high_cardinality="target"),
    landmark=True, candidate=True,
    description="ridge / L2 logistic regression: additive effects, fast, interpretable",
))
