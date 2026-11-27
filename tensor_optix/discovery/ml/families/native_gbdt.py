"""
tensor-optix's native gradient-boosted trees (``tensor_optix.gbdt``) as a
discovery family.

It is a *default* candidate only when neither LightGBM nor XGBoost is
installed (then discovery still tries boosted trees, the strongest tabular
family on many problems). With either library installed it is opt-in
(``families=["gbdt"]``): accuracy is on par with LightGBM on discovery's
benchmark (``benchmarks/validate_gbdt.py``), but it is pure numpy and an order
of magnitude slower.
"""
from __future__ import annotations

import importlib.util
from typing import Any, Dict, List

import numpy as np

from .base import CLASSIFICATION, REGRESSION, Estimator, Family, register
from ..preprocess import PipelineOptions


def _no_library() -> bool:
    return importlib.util.find_spec("lightgbm") is None and importlib.util.find_spec("xgboost") is None


class NativeGBDT(Estimator):
    def __init__(self, cfg: Dict[str, Any], seed: int, task: str, k: int):
        self.cfg, self.seed, self.task, self.k = dict(cfg), int(seed) % (2 ** 31), task, k
        self.stop_reason = "not_fitted"

    def fit(self, X, y, X_val=None, y_val=None):
        from ....gbdt import GBDTClassifier, GBDTRegressor
        kw = dict(learning_rate=self.cfg.get("learning_rate", 0.05), max_depth=self.cfg.get("max_depth", 6),
                  min_samples_leaf=self.cfg.get("min_samples_leaf", 20),
                  reg_lambda=self.cfg.get("reg_lambda", 1.0), subsample=self.cfg.get("subsample", 1.0),
                  colsample=self.cfg.get("colsample", 1.0), seed=self.seed)
        ev = None if X_val is None else (X_val, y_val)
        if self.task in CLASSIFICATION:
            self.m = GBDTClassifier(**kw).fit(X, np.asarray(y).astype(int), eval_set=ev)
        else:
            self.m = GBDTRegressor(objective="poisson" if self.task == "count" else "l2", **kw).fit(X, y, eval_set=ev)
        self.stop_reason = self.m.stop_reason_
        return self

    def predict_proba(self, X):
        P = self.m.predict_proba(X)
        if P.shape[1] == self.k:
            return P
        out = np.zeros((len(X), self.k))                       # a class absent from this training fold
        out[:, self.m.classes_.astype(int)] = P
        return out

    def predict(self, X):
        return self.m.predict(X)


def _configs(n: int, p: int, task: str) -> List[Dict[str, Any]]:
    return [{"max_depth": 6, "min_samples_leaf": 20},
            {"max_depth": 4, "min_samples_leaf": 40, "reg_lambda": 5.0, "subsample": 0.8, "colsample": 0.8}]


register(Family(
    name="gbdt", cost_rank=5, tasks=CLASSIFICATION + REGRESSION,
    make=lambda cfg, seed, task, k: NativeGBDT(cfg, seed, task, k),
    configs=_configs,
    preprocessing=PipelineOptions(scale="none", categorical="ordinal", high_cardinality="target"),
    landmark=False, candidate=True, default=_no_library,
    description="tensor-optix native histogram gradient-boosted trees (numpy), early-stopped by the "
                "ConvergenceMonitor."))
