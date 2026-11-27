"""
Model families for discovery.

A ``Family`` knows which tasks it supports, what preprocessing it needs, a
small fixed portfolio of configurations (``default_configs``) and how to make
an ``Estimator``. Estimators work on the float matrices produced by
``ColumnPipeline`` and expose

    fit(X, y, X_val=None, y_val=None) -> self
    predict_proba(X) -> (n, K)        classification
    predict(X) -> (n,)                regression / count

Everything here is deterministic given the seed passed to ``make``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

from ..preprocess import PipelineOptions

CLASSIFICATION = ("binary", "multiclass")
REGRESSION = ("regression", "count")


class Estimator:
    classes_: int = 0

    def fit(self, X: np.ndarray, y: np.ndarray, X_val=None, y_val=None) -> "Estimator":
        raise NotImplementedError

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    # Filled by iterative estimators (e.g. MLP): "converged", "max_epochs", ...
    stop_reason: str = "closed_form"


@dataclass(frozen=True)
class Family:
    name: str
    cost_rank: int                              # simplicity order: lower = simpler
    tasks: Tuple[str, ...]
    make: Callable[[Dict[str, Any], int, str, int], Estimator]   # (config, seed, task, n_classes)
    configs: Callable[[int, int, str], List[Dict[str, Any]]]     # (n_rows, n_features, task) -> configs
    preprocessing: PipelineOptions = field(default_factory=PipelineOptions)
    landmark: bool = False                      # cheap probe family
    candidate: bool = True                      # can be recommended
    description: str = ""
    available: Callable[[], bool] = lambda: True

    def supports(self, task: str) -> bool:
        return task in self.tasks


_REGISTRY: Dict[str, Family] = {}


def register(family: Family) -> Family:
    _REGISTRY[family.name] = family
    return family


def get_family(name: str) -> Family:
    _ensure_loaded()
    try:
        return _REGISTRY[name]
    except KeyError:
        raise ValueError(f"unknown family {name!r}; known: {sorted(_REGISTRY)}") from None


def all_families() -> List[Family]:
    _ensure_loaded()
    return sorted(_REGISTRY.values(), key=lambda f: (f.cost_rank, f.name))


_LOADED = False


def _ensure_loaded() -> None:
    # A flag, not "registry non-empty": importing one family module directly
    # registers only that family.
    global _LOADED
    if _LOADED:
        return
    _LOADED = True
    from . import baseline, knn, linear, naive_bayes, tree  # noqa: F401  (register on import)
    # Optional families register themselves only when their dependency imports.
    for mod in ("mlp", "boosting"):
        try:
            __import__(f"{__package__}.{mod}")
        except ImportError:
            pass


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def sigmoid(z: np.ndarray) -> np.ndarray:
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out
