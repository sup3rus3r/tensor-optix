#!/usr/bin/env python3
"""
Parity benchmark: tensor_optix.gbdt (native, numpy) vs LightGBM.

Six synthetic problems x 3 seeds, each with a fresh test set. Both libraries
get the same rows, the same inner validation rows for early stopping, the same
learning rate (0.05) and comparable capacity (native max_depth 6 vs LightGBM
num_leaves 31). The test-set metric is compared per seed (paired).

Parity (exit code 0) when, on every problem, the native model is not worse
than LightGBM by more than the metric's practical threshold (ROPE): mean
paired difference >= -ROPE. Time ratios are reported, not judged.

Usage:
    python benchmarks/validate_gbdt.py
"""
from __future__ import annotations

import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np  # noqa: E402

from tensor_optix.discovery.stats.compare import default_rope  # noqa: E402
from tensor_optix.discovery.stats.metrics import score  # noqa: E402
from tensor_optix.gbdt import GBDTClassifier, GBDTRegressor  # noqa: E402

N_TRAIN, N_TEST, VAL = 8000, 4000, 800
SEEDS = (0, 1, 2)


def _sig(z):
    return 1.0 / (1.0 + np.exp(-z))


def friedman1(r):
    X = r.uniform(size=(N_TRAIN + N_TEST, 10))
    y = 10 * np.sin(np.pi * X[:, 0] * X[:, 1]) + 20 * (X[:, 2] - .5) ** 2 + 10 * X[:, 3] + 5 * X[:, 4] \
        + r.normal(size=len(X))
    return X, y, "regression", (), "rmse"


def interactions_with_missing(r):
    X = r.normal(size=(N_TRAIN + N_TEST, 8))
    z = 1.5 * X[:, 0] * X[:, 1] - X[:, 2] + 0.8 * np.abs(X[:, 3])
    y = (r.random(len(X)) < _sig(z)).astype(int)
    X[r.random(X.shape) < 0.1] = np.nan
    return X, y, "binary", (), "auc"


def high_cardinality_categorical(r):
    n = N_TRAIN + N_TEST
    cat = r.integers(0, 200, n).astype(float)
    eff = r.normal(size=200)
    x = r.normal(size=(n, 4))
    y = (r.random(n) < _sig(1.5 * eff[cat.astype(int)] + x[:, 0])).astype(int)
    return np.column_stack([cat, x]), y, "binary", (0,), "auc"


def multiclass(r):
    X = r.normal(size=(N_TRAIN + N_TEST, 6))
    s = X[:, 0] + 0.5 * X[:, 1] * X[:, 2]
    y = np.digitize(s + 0.5 * r.normal(size=len(X)), [-0.7, 0.0, 0.7])
    return X, y, "multiclass", (), "macro_auc"


def poisson_counts(r):
    X = r.uniform(-1, 1, size=(N_TRAIN + N_TEST, 5))
    y = r.poisson(np.exp(0.8 * X[:, 0] + 0.5 * np.sin(3 * X[:, 1]) + 0.3)).astype(float)
    return X, y, "count", (), "poisson_deviance"


def imbalanced(r):
    X = r.normal(size=(N_TRAIN + N_TEST, 6))
    y = (r.random(len(X)) < _sig(-4.0 + 1.2 * X[:, 0] + X[:, 1] * X[:, 2])).astype(int)
    return X, y, "binary", (), "auc"


PROBLEMS = [friedman1, interactions_with_missing, high_cardinality_categorical, multiclass, poisson_counts,
            imbalanced]


def _score(metric, task, y, pred, k):
    if task in ("binary", "multiclass"):
        return score(metric, y, proba=pred, n_classes=k)
    return score(metric, y, values=pred, n_classes=0)


def run(problem, seed):
    import lightgbm as lgb
    r = np.random.default_rng(seed)
    X, y, task, cats, metric = problem(r)
    Xtr, ytr, Xte, yte = X[:N_TRAIN], y[:N_TRAIN], X[N_TRAIN:], y[N_TRAIN:]
    fit_rows, val_rows = slice(0, N_TRAIN - VAL), slice(N_TRAIN - VAL, N_TRAIN)
    k = int(y.max()) + 1 if task in ("binary", "multiclass") else 0
    t0 = time.perf_counter()
    if task in ("binary", "multiclass"):
        m = GBDTClassifier(learning_rate=0.05, categorical_features=cats, seed=seed).fit(
            Xtr[fit_rows], ytr[fit_rows], eval_set=(Xtr[val_rows], ytr[val_rows]))
        pn = m.predict_proba(Xte)
    else:
        m = GBDTRegressor(learning_rate=0.05, objective="poisson" if task == "count" else "l2", seed=seed).fit(
            Xtr[fit_rows], ytr[fit_rows], eval_set=(Xtr[val_rows], ytr[val_rows]))
        pn = m.predict(Xte)
    tn = time.perf_counter() - t0
    obj = {"binary": "binary", "multiclass": "multiclass", "regression": "regression", "count": "poisson"}[task]
    params = {"objective": obj, "learning_rate": 0.05, "num_leaves": 31, "verbose": -1, "seed": seed,
              "deterministic": True, "num_threads": 1}
    if task == "multiclass":
        params["num_class"] = k
    cf = list(cats) or "auto"
    dtr = lgb.Dataset(Xtr[fit_rows], ytr[fit_rows], categorical_feature=cf)
    dva = lgb.Dataset(Xtr[val_rows], ytr[val_rows], reference=dtr, categorical_feature=cf)
    t0 = time.perf_counter()
    b = lgb.train(params, dtr, 4000, valid_sets=[dva], callbacks=[lgb.early_stopping(50, verbose=False)])
    tl = time.perf_counter() - t0
    pl = b.predict(Xte)
    if task == "binary":
        pl = np.column_stack([1 - pl, pl])
    sn, sl = _score(metric, task, yte, pn, k), _score(metric, task, yte, pl, k)
    return metric, sn, sl, tn, tl, m.best_iteration_, np.std(y)


def main():
    from tensor_optix.discovery.stats.metrics import get_metric
    ok_all = True
    print(f"{'problem':32s} {'metric':18s} {'native':>10s} {'lightgbm':>10s} {'diff':>9s} {'rope':>7s} {'time x':>7s}  parity")
    for prob in PROBLEMS:
        rows = [run(prob, s) for s in SEEDS]
        metric = rows[0][0]
        hib = get_metric(metric).higher_is_better
        diffs = [(sn - sl) if hib else (sl - sn) for _, sn, sl, *_ in rows]     # positive = native better
        if metric in ("rmse", "mae"):
            rope = default_rope(metric, target_std=float(np.mean([r[6] for r in rows])))
        elif metric in ("log_loss", "brier", "poisson_deviance"):
            rope = default_rope(metric, scale=float(np.mean([r[2] for r in rows])))
        else:
            rope = default_rope(metric)
        mean_d = float(np.mean(diffs))
        parity = mean_d >= -rope
        ok_all &= parity
        tr = float(np.mean([r[3] for r in rows]) / max(np.mean([r[4] for r in rows]), 1e-9))
        print(f"{prob.__name__:32s} {metric:18s} {np.mean([r[1] for r in rows]):10.4f} "
              f"{np.mean([r[2] for r in rows]):10.4f} {mean_d:+9.4f} {rope:7.4f} {tr:7.1f}  "
              f"{'PASS' if parity else 'FAIL'}", flush=True)
    print("\nParity:", "PASS" if ok_all else "FAIL")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
