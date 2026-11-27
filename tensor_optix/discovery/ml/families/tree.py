"""
Shallow decision tree (CART on histogram bins), used as a landmark probe: a
depth-1 stump and a depth-3 tree. How much a shallow tree beats a linear
model says how much of the signal is thresholds / interactions -- a strong
hint that a tree ensemble family would fit.

Splits are searched over up to 32 quantile bins per feature (bins learned on
the training rows only). Classification uses Gini impurity and Laplace-smoothed
leaf probabilities; regression uses squared error and leaf means.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from ..preprocess import PipelineOptions
from .base import CLASSIFICATION, REGRESSION, Estimator, Family, register

_MAX_BINS = 32


def _bin_edges(X: np.ndarray) -> List[np.ndarray]:
    edges = []
    for j in range(X.shape[1]):
        e = np.unique(np.quantile(X[:, j], np.linspace(0, 1, _MAX_BINS + 1)[1:-1]))
        edges.append(e)
    return edges


def _digitize(X: np.ndarray, edges: List[np.ndarray]) -> np.ndarray:
    B = np.empty(X.shape, dtype=np.int32)
    for j, e in enumerate(edges):
        B[:, j] = np.searchsorted(e, X[:, j], side="right")
    return B


class Tree(Estimator):
    def __init__(self, depth: int, task: str, n_classes: int, min_leaf: int = 5):
        self.depth, self.task, self.k, self.min_leaf = depth, task, n_classes, min_leaf

    # node: (feature, bin_threshold, left, right) or ("leaf", value)
    def fit(self, X, y, X_val=None, y_val=None):
        self.edges = _bin_edges(X)
        B = _digitize(X, self.edges)
        if self.task in CLASSIFICATION:
            T = np.eye(self.k)[y.astype(int)]
        else:
            yf = y.astype(np.float64)
            T = np.column_stack([np.ones(len(yf)), yf, yf * yf])     # count, sum, sum of squares
        self.root = self._grow(B, T, np.arange(len(B)), 0)
        return self

    def _leaf(self, T: np.ndarray) -> np.ndarray:
        s = T.sum(axis=0)
        if self.task in CLASSIFICATION:
            return (s + 1.0) / (s.sum() + self.k)
        return np.array([s[1] / max(s[0], 1.0)])

    def _impurity(self, S: np.ndarray) -> np.ndarray:
        """S: (..., width) summary per side -> impurity * count."""
        if self.task in CLASSIFICATION:
            n = S.sum(axis=-1)
            with np.errstate(invalid="ignore", divide="ignore"):
                gini = 1.0 - np.sum((S / np.maximum(n[..., None], 1e-12)) ** 2, axis=-1)
            return gini * n
        n, s1, s2 = S[..., 0], S[..., 1], S[..., 2]
        return s2 - s1 * s1 / np.maximum(n, 1e-12)                 # SSE

    def _grow(self, B, T, idx, d):
        Tn = T[idx]
        if d >= self.depth or len(idx) < 2 * self.min_leaf:
            return ("leaf", self._leaf(Tn))
        total = Tn.sum(axis=0)
        parent = float(self._impurity(total))
        best: Optional[Tuple[float, int, int]] = None
        width = T.shape[1]
        for j in range(B.shape[1]):
            nb = len(self.edges[j]) + 1
            if nb < 2:
                continue
            H = np.zeros((nb, width))
            np.add.at(H, B[idx, j], Tn)
            left = np.cumsum(H, axis=0)[:-1]                        # split after bin b
            right = total - left
            cnt_l = left.sum(axis=1) if self.task in CLASSIFICATION else left[:, 0]
            cnt_r = right.sum(axis=1) if self.task in CLASSIFICATION else right[:, 0]
            ok = (cnt_l >= self.min_leaf) & (cnt_r >= self.min_leaf)
            if not ok.any():
                continue
            gain = parent - self._impurity(left) - self._impurity(right)
            gain[~ok] = -np.inf
            b = int(np.argmax(gain))
            if gain[b] > 1e-12 and (best is None or gain[b] > best[0]):
                best = (float(gain[b]), j, b)
        if best is None:
            return ("leaf", self._leaf(Tn))
        _, j, b = best
        go_left = B[idx, j] <= b
        return (j, b, self._grow(B, T, idx[go_left], d + 1), self._grow(B, T, idx[~go_left], d + 1))

    def _apply(self, X) -> np.ndarray:
        B = _digitize(X, self.edges)
        out = np.empty((len(X), self.k if self.task in CLASSIFICATION else 1))
        stack = [(self.root, np.arange(len(X)))]
        while stack:
            node, rows = stack.pop()
            if node[0] == "leaf":
                out[rows] = node[1]
                continue
            j, b, l, r = node
            m = B[rows, j] <= b
            stack.append((l, rows[m]))
            stack.append((r, rows[~m]))
        return out

    def predict_proba(self, X):
        return self._apply(X)

    def predict(self, X):
        return self._apply(X)[:, 0]


register(Family(
    name="tree", cost_rank=4, tasks=CLASSIFICATION + REGRESSION,
    make=lambda cfg, seed, task, k: Tree(int(cfg.get("depth", 3)), task, k),
    configs=lambda n, p, task: [{"depth": 1}, {"depth": 3}],
    preprocessing=PipelineOptions(scale="none", categorical="ordinal", high_cardinality="target"),
    landmark=True, candidate=False,
    description="shallow decision tree: reveals threshold / interaction structure",
))
