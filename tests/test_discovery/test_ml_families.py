"""
Landmark / candidate families: correctness against closed forms and optimality
conditions, and the structural behaviour landmarks rely on (linear truth ->
linear wins; XOR -> trees / kNN beat linear).
"""
import numpy as np
import pytest

from tensor_optix.discovery.ml.data import to_table
from tensor_optix.discovery.ml.evaluate import evaluate_family
from tensor_optix.discovery.ml.families.base import Family, all_families, get_family, sigmoid
from tensor_optix.discovery.ml.families.knn import KNN
from tensor_optix.discovery.ml.families.linear import Logistic, Ridge, _logistic_binary
from tensor_optix.discovery.ml.families.naive_bayes import LDA, GaussianNB
from tensor_optix.discovery.ml.families.tree import Tree
from tensor_optix.discovery.ml.splits import KFold, RepeatedSplitter
from tensor_optix.discovery.ml.types import infer_types
from tensor_optix.discovery.stats.metrics import accuracy, roc_auc_binary


def test_registry():
    names = {f.name for f in all_families()}
    assert {"baseline", "linear", "tree", "knn", "naive_bayes", "lda"} <= names
    with pytest.raises(ValueError):
        get_family("magic")
    ranks = [f.cost_rank for f in all_families()]
    assert ranks == sorted(ranks)


def test_ridge_matches_least_squares_and_shrinks():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 5))
    y = X @ np.array([1.0, -2.0, 0.5, 0.0, 3.0]) + 4.0 + rng.normal(scale=0.1, size=300)
    r = Ridge(alpha=1e-12).fit(X, y)
    A = np.hstack([X, np.ones((300, 1))])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    assert np.allclose(r.w, coef[:5], atol=1e-6)
    assert np.allclose(r.predict(X), A @ coef, atol=1e-6)
    big = Ridge(alpha=10.0).fit(X, y)
    assert np.linalg.norm(big.w) < np.linalg.norm(r.w)


@pytest.mark.parametrize("p", [4, 700])          # Newton path and gradient-descent path
def test_logistic_satisfies_the_optimality_condition(p):
    rng = np.random.default_rng(1)
    n = 400
    X = rng.normal(size=(n, p))
    w_true = np.zeros(p)
    w_true[:3] = [2.0, -1.0, 0.5]
    y = (rng.random(n) < sigmoid(X @ w_true)).astype(float)
    lam = 1e-1 * n
    w = _logistic_binary(X, y, lam)
    Xb = np.hstack([X, np.ones((n, 1))])
    grad = Xb.T @ (sigmoid(Xb @ w) - y)
    grad[:-1] += lam * w[:-1]
    assert np.max(np.abs(grad)) < (1e-4 if p < 600 else 5e-2) * n


def test_logistic_multiclass_probabilities():
    rng = np.random.default_rng(2)
    centers = np.array([[0, 0], [4, 0], [0, 4]])
    y = rng.integers(0, 3, 600)
    X = centers[y] + rng.normal(size=(600, 2))
    P = Logistic(1e-3, 3).fit(X, y).predict_proba(X)
    assert np.allclose(P.sum(axis=1), 1.0)
    assert accuracy(y, P.argmax(axis=1)) > 0.9


def test_tree_finds_a_step_and_xor():
    rng = np.random.default_rng(3)
    x = rng.uniform(-1, 1, size=(800, 2))
    y_step = (x[:, 0] > 0.3).astype(int)
    stump = Tree(1, "binary", 2).fit(x, y_step)
    assert accuracy(y_step, stump.predict_proba(x).argmax(axis=1)) > 0.97
    y_xor = ((x[:, 0] > 0) ^ (x[:, 1] > 0)).astype(int)
    assert accuracy(y_xor, Tree(1, "binary", 2).fit(x, y_xor).predict_proba(x).argmax(1)) < 0.7
    # greedy CART has no expected gain on XOR's first split, so it is not perfect -- but clearly non-linear
    assert accuracy(y_xor, Tree(3, "binary", 2).fit(x, y_xor).predict_proba(x).argmax(1)) > 0.9
    yr = np.where(x[:, 1] > -0.2, 5.0, -5.0) + rng.normal(scale=0.1, size=800)
    reg = Tree(1, "regression", 0).fit(x, yr)
    assert np.mean(np.abs(reg.predict(x) - yr)) < 0.3


def test_tree_respects_min_leaf_and_constant_input():
    X = np.zeros((50, 3))
    y = np.arange(50) % 2
    t = Tree(3, "binary", 2).fit(X, y)
    assert t.root[0] == "leaf"
    assert np.allclose(t.predict_proba(X).sum(axis=1), 1.0)


def test_knn_and_bayes_family():
    rng = np.random.default_rng(4)
    y = rng.integers(0, 2, 600)
    X = np.column_stack([y * 3.0, -y * 2.0]) + rng.normal(size=(600, 2))
    for est in (KNN(15, "binary", 2, seed=0), GaussianNB(2), LDA(2)):
        P = est.fit(X[:400], y[:400]).predict_proba(X[400:])
        assert np.allclose(P.sum(axis=1), 1.0) and (P > 0).all() and (P < 1).all()
        assert roc_auc_binary(y[400:], P[:, 1]) > 0.97
    knn_r = KNN(10, "regression", 0, seed=0).fit(X[:400], X[:400, 0])
    assert np.corrcoef(knn_r.predict(X[400:]), X[400:, 0])[0, 1] > 0.9


def test_knn_subsamples_deterministically():
    X = np.random.default_rng(5).normal(size=(8000, 3))
    y = (X[:, 0] > 0).astype(int)
    a = KNN(5, "binary", 2, seed=1, max_train=1000).fit(X, y)
    b = KNN(5, "binary", 2, seed=1, max_train=1000).fit(X, y)
    assert len(a.X) == 1000 and np.array_equal(a.X, b.X)


# ---------------------------------------------------------------- evaluation harness

def _setup(X, y, task, k, n_splits=5, repeats=2):
    table = to_table(X, target=y)
    types = infer_types(table, np.random.default_rng(0))
    folds = RepeatedSplitter(KFold(n_splits, seed=0), repeats).folds(len(y))
    return table, types, folds


def _eval(table, types, folds, y, name, cfg, task, k, metric):
    return evaluate_family(table, types, y, folds, get_family(name), cfg, task=task, n_classes=k,
                           metric=metric, secondary=("log_loss",) if task == "binary" else ())


def test_linear_truth_linear_beats_shallow_trees():
    rng = np.random.default_rng(6)
    X = rng.normal(size=(800, 6))
    y = (rng.random(800) < sigmoid(X @ np.array([1.5, -1, 0.8, 0.5, -0.5, 0.3]))).astype(int)
    table, types, folds = _setup(X, y, "binary", 2)
    lin = _eval(table, types, folds, y, "linear", {"alpha": 1e-3}, "binary", 2, "auc")
    tree = _eval(table, types, folds, y, "tree", {"depth": 3}, "binary", 2, "auc")
    base = _eval(table, types, folds, y, "baseline", {}, "binary", 2, "auc")
    assert lin.status == "ok" and len(lin.scores) == 10 and lin.interval is not None
    assert lin.mean > tree.mean > base.mean
    assert base.mean == pytest.approx(0.5)
    assert len(lin.secondary["log_loss"]) == 10


def test_xor_truth_nonlinear_families_beat_linear():
    rng = np.random.default_rng(7)
    X = rng.uniform(-1, 1, size=(900, 4))
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)
    table, types, folds = _setup(X, y, "binary", 2, repeats=1)
    lin = _eval(table, types, folds, y, "linear", {"alpha": 1e-3}, "binary", 2, "auc")
    tree = _eval(table, types, folds, y, "tree", {"depth": 3}, "binary", 2, "auc")
    knn = _eval(table, types, folds, y, "knn", {"k": 15}, "binary", 2, "auc")
    assert lin.mean < 0.6 and tree.mean > 0.9 and knn.mean > 0.9


def test_regression_families_and_failure_capture():
    rng = np.random.default_rng(8)
    X = rng.normal(size=(500, 3))
    y = 2 * X[:, 0] - X[:, 1] + rng.normal(scale=0.3, size=500)
    table, types, folds = _setup(X, y, "regression", 0, repeats=1)
    lin = _eval(table, types, folds, y, "linear", {"alpha": 1e-4}, "regression", 0, "rmse")
    base = _eval(table, types, folds, y, "baseline", {}, "regression", 0, "rmse")
    assert not lin.higher_is_better and lin.mean < 0.5 < base.mean

    class Boom:
        stop_reason = "x"
        def fit(self, *a, **k):
            raise RuntimeError("boom")
    bad = Family(name="boom", cost_rank=9, tasks=("regression",), make=lambda c, s, t, k: Boom(),
                 configs=lambda n, p, t: [{}])
    res = evaluate_family(table, types, y, folds, bad, {}, task="regression", n_classes=0, metric="rmse")
    assert res.status == "failed" and "boom" in res.error and all(s is None for s in res.scores)


def test_evaluation_is_reproducible():
    rng = np.random.default_rng(9)
    X = rng.normal(size=(300, 3))
    y = (X[:, 0] + rng.normal(size=300) > 0).astype(int)
    table, types, folds = _setup(X, y, "binary", 2, repeats=1)
    a = _eval(table, types, folds, y, "knn", {"k": 7}, "binary", 2, "auc")
    b = _eval(table, types, folds, y, "knn", {"k": 7}, "binary", 2, "auc")
    assert a.scores == b.scores
