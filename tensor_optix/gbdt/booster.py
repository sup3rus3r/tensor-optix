"""
Boosting loop, objectives and estimators.

Early stopping uses tensor-optix's ``ConvergenceMonitor`` on the negated
validation loss (one observation per boosting round): training stops when the
monitor reaches DORMANT -- no new ground for its patience and no slow rise in
progress -- and the model is truncated to the round with the lowest
validation loss. ``n_estimators`` is a model-size cap, like any GBDT library's.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence

import numpy as np

from .core import BinMapper, Tree, TreeParams, build_tree

GBDT_CONVERGENCE = dict(timescales=(8, 32, 128), cooling_patience=20, dormant_patience=60, persist=2,
                        rel_delta=1e-4)


def _sigmoid(z):
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def _softmax(Z):
    Z = Z - Z.max(axis=1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)


class Objective:
    n_outputs = 1

    def init(self, y): raise NotImplementedError
    def grad_hess(self, y, F): raise NotImplementedError
    def loss(self, y, F): raise NotImplementedError


class Binary(Objective):
    def init(self, y):
        p = float(np.clip(np.mean(y), 1e-6, 1 - 1e-6))
        return np.array([math.log(p / (1 - p))])

    def grad_hess(self, y, F):
        p = _sigmoid(F[:, 0])
        return (p - y)[:, None], np.maximum(p * (1 - p), 1e-16)[:, None]

    def loss(self, y, F):
        p = np.clip(_sigmoid(F[:, 0]), 1e-15, 1 - 1e-15)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


class Multiclass(Objective):
    def __init__(self, k):
        self.n_outputs = k

    def init(self, y):
        c = np.bincount(y.astype(int), minlength=self.n_outputs) + 1.0
        return np.log(c / c.sum())

    def grad_hess(self, y, F):
        P = _softmax(F)
        Y = np.zeros_like(P)
        Y[np.arange(len(y)), y.astype(int)] = 1.0
        k = self.n_outputs
        return P - Y, np.maximum(P * (1 - P) * k / (k - 1), 1e-16)

    def loss(self, y, F):
        P = np.clip(_softmax(F), 1e-15, 1.0)
        return float(-np.mean(np.log(P[np.arange(len(y)), y.astype(int)])))


class L2(Objective):
    def init(self, y):
        return np.array([float(np.mean(y))])

    def grad_hess(self, y, F):
        return (F[:, 0] - y)[:, None], np.ones((len(y), 1))

    def loss(self, y, F):
        return float(np.mean((F[:, 0] - y) ** 2))


class Poisson(Objective):
    MAX_RAW = 30.0

    def init(self, y):
        return np.array([math.log(max(float(np.mean(y)), 1e-9))])

    def grad_hess(self, y, F):
        mu = np.exp(np.minimum(F[:, 0], self.MAX_RAW))
        return (mu - y)[:, None], np.maximum(mu, 1e-16)[:, None] * math.exp(0.7)   # max_delta_step-like damping

    def loss(self, y, F):
        f = np.minimum(F[:, 0], self.MAX_RAW)
        return float(np.mean(np.exp(f) - y * f))


@dataclass
class GBDTParams:
    n_estimators: int = 2000
    learning_rate: float = 0.1
    max_depth: int = 6
    min_samples_leaf: int = 20
    min_child_weight: float = 1e-3
    reg_lambda: float = 1.0
    gamma: float = 0.0
    subsample: float = 1.0
    colsample: float = 1.0
    max_bins: int = 255
    early_stopping: bool = True
    validation_fraction: float = 0.1
    cat_smooth: float = 10.0
    cat_l2: float = 10.0
    min_data_per_group: int = 100
    max_cat_threshold: int = 32
    seed: int = 0


class Booster:
    def __init__(self, objective: Objective, params: GBDTParams, categorical: Sequence[int] = ()):
        self.obj, self.p = objective, params
        self.categorical = tuple(int(c) for c in categorical)
        self.trees: List[List[Tree]] = []
        self.init_score: Optional[np.ndarray] = None
        self.best_iteration: Optional[int] = None
        self.stop_reason = "not_fitted"
        self.history: List[float] = []

    def _raw(self, Xb: np.ndarray, n_iter: Optional[int] = None) -> np.ndarray:
        F = np.tile(self.init_score, (len(Xb), 1)).astype(np.float64)
        miss = np.asarray(self.mapper.n_bins, dtype=np.int64) - 1
        for round_trees in self.trees[: n_iter if n_iter is not None else len(self.trees)]:
            for k, t in enumerate(round_trees):
                F[:, k] += t.predict_binned(Xb, miss)
        return F

    def fit(self, X: np.ndarray, y: np.ndarray, X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None) -> "Booster":
        from ..core.convergence import ConvergenceMonitor
        from ..core.types import LoopState
        p = self.p
        rng = np.random.default_rng(p.seed)
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if p.early_stopping and X_val is None:
            n_val = max(1, int(round(p.validation_fraction * len(X))))
            perm = rng.permutation(len(X))
            va, tr = perm[:n_val], perm[n_val:]
            X, X_val, y, y_val = X[tr], X[va], y[tr], y[va]
        self.mapper = BinMapper(max_bins=p.max_bins, categorical=self.categorical).fit(X)
        Xb = self.mapper.transform(X)
        is_cat = np.zeros(X.shape[1], dtype=bool)
        is_cat[list(self.categorical)] = True
        self.init_score = self.obj.init(y)
        F = np.tile(self.init_score, (len(X), 1)).astype(np.float64)
        Xvb = None if X_val is None else self.mapper.transform(np.asarray(X_val, dtype=np.float64))
        Fv = None if Xvb is None else np.tile(self.init_score, (len(Xvb), 1)).astype(np.float64)
        yv = None if y_val is None else np.asarray(y_val, dtype=np.float64)
        miss = np.asarray(self.mapper.n_bins, dtype=np.int64) - 1
        tp = TreeParams(max_depth=p.max_depth, min_samples_leaf=p.min_samples_leaf,
                        min_child_weight=p.min_child_weight, reg_lambda=p.reg_lambda, gamma=p.gamma,
                        learning_rate=p.learning_rate, cat_smooth=p.cat_smooth, cat_l2=p.cat_l2,
                        min_data_per_group=p.min_data_per_group, max_cat_threshold=p.max_cat_threshold)
        monitor = ConvergenceMonitor(**GBDT_CONVERGENCE) if Xvb is not None else None
        best_loss, best_it = math.inf, 0
        self.trees, self.history = [], []
        n_feat = X.shape[1]
        self.stop_reason = "n_estimators"
        for it in range(p.n_estimators):
            g, h = self.obj.grad_hess(y, F)
            rows = np.arange(len(X)) if p.subsample >= 1.0 else np.sort(
                rng.choice(len(X), max(1, int(p.subsample * len(X))), replace=False))
            feats = np.arange(n_feat) if p.colsample >= 1.0 else np.sort(
                rng.choice(n_feat, max(1, int(round(p.colsample * n_feat))), replace=False))
            round_trees = []
            for k in range(self.obj.n_outputs):
                t = build_tree(Xb[rows], g[rows, k], h[rows, k], self.mapper.n_bins, is_cat, feats, tp)
                round_trees.append(t)
                F[:, k] += t.predict_binned(Xb, miss)
                if Fv is not None:
                    Fv[:, k] += t.predict_binned(Xvb, miss)
            self.trees.append(round_trees)
            if Fv is not None:
                loss = self.obj.loss(yv, Fv)
                self.history.append(loss)
                if loss < best_loss - 1e-12:
                    best_loss, best_it = loss, it + 1
                rep = monitor.observe(-loss, at=it + 1)
                if rep.state == LoopState.DORMANT:
                    self.stop_reason = "converged"
                    break
        if Fv is not None:
            self.best_iteration = best_it
            self.trees = self.trees[:best_it]
        else:
            self.best_iteration = len(self.trees)
        return self

    def raw_predict(self, X: np.ndarray) -> np.ndarray:
        return self._raw(self.mapper.transform(np.asarray(X, dtype=np.float64)))


# ---------------------------------------------------------------------------
# estimators
# ---------------------------------------------------------------------------

@dataclass
class _Base:
    n_estimators: int = 2000
    learning_rate: float = 0.1
    max_depth: int = 6
    min_samples_leaf: int = 20
    min_child_weight: float = 1e-3
    reg_lambda: float = 1.0
    gamma: float = 0.0
    subsample: float = 1.0
    colsample: float = 1.0
    max_bins: int = 255
    early_stopping: bool = True
    validation_fraction: float = 0.1
    cat_smooth: float = 10.0
    cat_l2: float = 10.0
    min_data_per_group: int = 100
    max_cat_threshold: int = 32
    categorical_features: Sequence[int] = ()
    seed: int = 0
    booster_: Any = field(default=None, init=False, repr=False)

    def _params(self) -> GBDTParams:
        return GBDTParams(**{k: getattr(self, k) for k in GBDTParams.__dataclass_fields__})

    @property
    def best_iteration_(self):
        return None if self.booster_ is None else self.booster_.best_iteration

    @property
    def stop_reason_(self):
        return None if self.booster_ is None else self.booster_.stop_reason


@dataclass
class GBDTRegressor(_Base):
    objective: str = "l2"                    # "l2" | "poisson"

    def fit(self, X, y, eval_set=None) -> "GBDTRegressor":
        obj = {"l2": L2(), "poisson": Poisson()}[self.objective]
        Xv, yv = (eval_set if eval_set is not None else (None, None))
        self.booster_ = Booster(obj, self._params(), self.categorical_features).fit(X, y, Xv, yv)
        return self

    def predict(self, X) -> np.ndarray:
        F = self.booster_.raw_predict(X)[:, 0]
        return np.exp(np.minimum(F, Poisson.MAX_RAW)) if self.objective == "poisson" else F


@dataclass
class GBDTClassifier(_Base):
    classes_: Optional[np.ndarray] = field(default=None, init=False)

    def fit(self, X, y, eval_set=None) -> "GBDTClassifier":
        y = np.asarray(y)
        self.classes_, yi = np.unique(y, return_inverse=True)
        k = len(self.classes_)
        if k < 2:
            raise ValueError("GBDTClassifier needs at least two classes")
        obj = Binary() if k == 2 else Multiclass(k)
        Xv = yv = None
        if eval_set is not None:
            Xv, yraw = eval_set
            index = {c: i for i, c in enumerate(self.classes_.tolist())}
            yv = np.array([index[v] for v in np.asarray(yraw).tolist()], dtype=np.float64)
        self.booster_ = Booster(obj, self._params(), self.categorical_features).fit(X, yi.astype(np.float64), Xv, yv)
        return self

    def predict_proba(self, X) -> np.ndarray:
        F = self.booster_.raw_predict(X)
        if F.shape[1] == 1:
            p = _sigmoid(F[:, 0])
            return np.column_stack([1 - p, p])
        return _softmax(F)

    def predict(self, X) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
