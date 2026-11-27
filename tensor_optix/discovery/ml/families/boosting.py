"""
Gradient-boosted tree families backed by LightGBM and XGBoost, used only when
the library is installed (``pip install tensor-optix[discovery-boost]``).

Each fit holds out an inner validation split of its training fold and stops
adding trees once that loss has not improved for ``EARLY_STOP`` rounds (the
test fold is never seen). Single-threaded and seeded, so results are
reproducible run to run.

Importing this module raises ImportError when neither library is installed;
the family registry then offers neither.
"""
from __future__ import annotations

import importlib.util
from typing import Any, Dict, List

import numpy as np

from .base import CLASSIFICATION, REGRESSION, Estimator, Family, register
from ..preprocess import PipelineOptions

HAS_LIGHTGBM = importlib.util.find_spec("lightgbm") is not None
HAS_XGBOOST = importlib.util.find_spec("xgboost") is not None
if not (HAS_LIGHTGBM or HAS_XGBOOST):
    raise ImportError("neither lightgbm nor xgboost is installed")

VAL_FRACTION = 0.15
EARLY_STOP = 50
MAX_TREES = 2000

_TREE_PREP = PipelineOptions(scale="none", categorical="ordinal", high_cardinality="target")


def _inner_split(n: int, y: np.ndarray, task: str, seed: int):
    rng = np.random.default_rng(seed)
    va = []
    if task in CLASSIFICATION:
        for c in np.unique(y):
            rows = np.flatnonzero(y == c)
            take = int(round(len(rows) * VAL_FRACTION))
            if take >= 1 and len(rows) - take >= 1:
                va.append(rng.choice(rows, take, replace=False))
    va = np.concatenate(va) if va else rng.choice(n, max(1, int(round(VAL_FRACTION * n))), replace=False)
    mask = np.zeros(n, dtype=bool)
    mask[va] = True
    return np.flatnonzero(~mask), np.flatnonzero(mask)


class _Boosted(Estimator):
    def __init__(self, cfg: Dict[str, Any], seed: int, task: str, k: int):
        self.cfg = dict(cfg)
        self.seed = int(seed) % (2 ** 31)
        self.task = task
        self.k = int(k)
        self.stop_reason = "not_fitted"
        self.n_trees = 0

    def _data(self, X, y, X_val, y_val):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        if X_val is None:
            tr, va = _inner_split(len(X), y, self.task, self.seed)
            return X[tr], y[tr], X[va], y[va]
        return X, y, np.asarray(X_val, dtype=np.float64), np.asarray(y_val)

    def _finish(self, best: int, total: int):
        self.n_trees = int(best)
        self.stop_reason = "early_stopping" if total < MAX_TREES else "max_trees"


class LightGBMEstimator(_Boosted):
    def fit(self, X, y, X_val=None, y_val=None):
        import lightgbm as lgb
        Xtr, ytr, Xva, yva = self._data(X, y, X_val, y_val)
        params = {"learning_rate": self.cfg.get("learning_rate", 0.05),
                  "num_leaves": self.cfg.get("num_leaves", 31),
                  "min_child_samples": self.cfg.get("min_child_samples", 20),
                  "reg_lambda": self.cfg.get("reg_lambda", 0.0),
                  "subsample": self.cfg.get("subsample", 1.0), "subsample_freq": 1 if self.cfg.get("subsample") else 0,
                  "colsample_bytree": self.cfg.get("colsample_bytree", 1.0),
                  "seed": self.seed, "deterministic": True, "force_col_wise": True, "num_threads": 1,
                  "verbose": -1}
        if self.task == "binary":
            params["objective"] = "binary"
        elif self.task == "multiclass":
            params.update(objective="multiclass", num_class=self.k)
        elif self.task == "count":
            params["objective"] = "poisson"
        else:
            params["objective"] = "regression"
        dtr = lgb.Dataset(Xtr, ytr, free_raw_data=True)
        dva = lgb.Dataset(Xva, yva, reference=dtr)
        self._b = lgb.train(params, dtr, num_boost_round=MAX_TREES, valid_sets=[dva],
                            callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)])
        self._finish(self._b.best_iteration or self._b.current_iteration(), self._b.current_iteration())
        return self

    def _raw(self, X):
        return self._b.predict(np.asarray(X, dtype=np.float64), num_iteration=self._b.best_iteration or None)

    def predict_proba(self, X):
        p = self._raw(X)
        if self.task == "binary":
            p = np.clip(p, 0.0, 1.0)
            return np.column_stack([1.0 - p, p])
        return p

    def predict(self, X):
        return self._raw(X)


class XGBoostEstimator(_Boosted):
    def fit(self, X, y, X_val=None, y_val=None):
        import xgboost as xgb
        Xtr, ytr, Xva, yva = self._data(X, y, X_val, y_val)
        params = {"eta": self.cfg.get("learning_rate", 0.05), "max_depth": self.cfg.get("max_depth", 6),
                  "min_child_weight": self.cfg.get("min_child_weight", 1.0),
                  "lambda": self.cfg.get("reg_lambda", 1.0), "subsample": self.cfg.get("subsample", 1.0),
                  "colsample_bytree": self.cfg.get("colsample_bytree", 1.0),
                  "tree_method": "hist", "seed": self.seed, "nthread": 1, "verbosity": 0}
        if self.task == "binary":
            params.update(objective="binary:logistic", eval_metric="logloss")
        elif self.task == "multiclass":
            params.update(objective="multi:softprob", num_class=self.k, eval_metric="mlogloss")
        elif self.task == "count":
            params.update(objective="count:poisson", eval_metric="poisson-nloglik")
        else:
            params.update(objective="reg:squarederror", eval_metric="rmse")
        dtr = xgb.DMatrix(Xtr, label=ytr)
        dva = xgb.DMatrix(Xva, label=yva)
        self._b = xgb.train(params, dtr, num_boost_round=MAX_TREES, evals=[(dva, "val")],
                            early_stopping_rounds=EARLY_STOP, verbose_eval=False)
        self._best = int(self._b.best_iteration) + 1
        self._finish(self._best, self._b.num_boosted_rounds())
        return self

    def _raw(self, X):
        import xgboost as xgb
        return self._b.predict(xgb.DMatrix(np.asarray(X, dtype=np.float64)), iteration_range=(0, self._best))

    def predict_proba(self, X):
        p = self._raw(X)
        if self.task == "binary":
            return np.column_stack([1.0 - p, p])
        return p

    def predict(self, X):
        return self._raw(X)


def _lgb_configs(n: int, p: int, task: str) -> List[Dict[str, Any]]:
    return [{"num_leaves": 31, "min_child_samples": 20},
            {"num_leaves": 15, "min_child_samples": 40, "reg_lambda": 1.0, "subsample": 0.8,
             "colsample_bytree": 0.8}]


def _xgb_configs(n: int, p: int, task: str) -> List[Dict[str, Any]]:
    return [{"max_depth": 6, "min_child_weight": 1.0},
            {"max_depth": 4, "min_child_weight": 5.0, "reg_lambda": 5.0, "subsample": 0.8,
             "colsample_bytree": 0.8}]


if HAS_LIGHTGBM:
    register(Family(
        name="lightgbm", cost_rank=5, tasks=CLASSIFICATION + REGRESSION,
        make=lambda cfg, seed, task, k: LightGBMEstimator(cfg, seed, task, k),
        configs=_lgb_configs, preprocessing=_TREE_PREP, landmark=False, candidate=True,
        description="Gradient-boosted trees (LightGBM), early-stopped on an inner validation split."))

if HAS_XGBOOST:
    register(Family(
        name="xgboost", cost_rank=5, tasks=CLASSIFICATION + REGRESSION,
        make=lambda cfg, seed, task, k: XGBoostEstimator(cfg, seed, task, k),
        configs=_xgb_configs, preprocessing=_TREE_PREP, landmark=False, candidate=True,
        description="Gradient-boosted trees (XGBoost), early-stopped on an inner validation split."))
