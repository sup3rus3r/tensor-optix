"""tensor_optix.gbdt: binning, splits, missing values, categoricals, objectives, early stopping."""
import numpy as np
import pytest

from tensor_optix.gbdt import GBDTClassifier, GBDTRegressor
from tensor_optix.gbdt.core import BinMapper, TreeParams, build_tree


def test_bin_mapper_numeric_categorical_missing():
    X = np.array([[0.0, 3.0], [1.0, 5.0], [2.0, np.nan], [np.nan, 3.0], [1.0, 99.0]])
    m = BinMapper(max_bins=255, categorical=(1,)).fit(X)
    B = m.transform(X)
    assert B[3, 0] == m.missing_bin(0) and B[2, 1] == m.missing_bin(1)
    assert B[0, 0] < B[1, 0] < B[2, 0]
    assert B[0, 1] == B[3, 1]                                   # same category, same code
    new = m.transform(np.array([[0.5, 7.0]]))                   # unseen category -> "other"
    assert new[0, 1] == len(m.categories[1])
    many = np.random.default_rng(0).normal(size=(5000, 1))
    mm = BinMapper(max_bins=64).fit(many)
    assert mm.n_bins[0] <= 65 and len(np.unique(mm.transform(many))) <= 64


def test_single_split_is_the_newton_optimum():
    # a stump on one feature: the best threshold is at the jump, leaves are -G/(H+lambda)
    x = np.linspace(0, 1, 200)[:, None]
    y = (x[:, 0] > 0.6).astype(float) * 2.0
    m = BinMapper().fit(x)
    Xb = m.transform(x)
    g, h = -y, np.ones(len(y))                                  # residuals from a zero prediction
    t = build_tree(Xb, g, h, m.n_bins, np.zeros(1, dtype=bool), np.array([0]),
                   TreeParams(max_depth=1, min_samples_leaf=1, reg_lambda=0.0, learning_rate=1.0))
    pred = t.predict_binned(Xb, np.array(m.n_bins) - 1)
    assert np.allclose(pred, y)


def test_missing_values_learn_their_direction():
    r = np.random.default_rng(1)
    x = r.normal(size=4000)
    miss = r.random(4000) < 0.3
    y = np.where(miss, 1, (x > 0).astype(int))                 # missing behaves like the high side
    X = np.where(miss, np.nan, x)[:, None]
    m = GBDTClassifier(n_estimators=30, learning_rate=0.3, early_stopping=False).fit(X, y)
    p = m.predict_proba(np.array([[np.nan], [-2.0], [2.0]]))[:, 1]
    assert p[0] > 0.9 and p[1] < 0.1 and p[2] > 0.9


def test_native_categorical_partition_beats_ordinal_codes():
    r = np.random.default_rng(2)
    n = 6000
    cat = r.integers(0, 40, n)
    good = r.permutation(40)[:20]
    y = (np.isin(cat, good) ^ (r.random(n) < 0.05)).astype(int)
    X = cat.astype(float)[:, None]
    kw = dict(n_estimators=1, learning_rate=1.0, max_depth=1, early_stopping=False, min_samples_leaf=1)
    native = GBDTClassifier(categorical_features=(0,), **kw).fit(X, y)
    ordinal = GBDTClassifier(**kw).fit(X, y)
    acc = lambda m: np.mean(m.predict(X) == y)                 # noqa: E731
    assert acc(native) > 0.9 > acc(ordinal) + 0.1              # one categorical split separates the sets


def test_regression_and_early_stopping_with_convergence_monitor():
    r = np.random.default_rng(3)
    X = r.uniform(size=(5000, 5))
    y = 10 * np.sin(np.pi * X[:, 0] * X[:, 1]) + 5 * X[:, 2] + r.normal(size=5000)
    m = GBDTRegressor(learning_rate=0.1).fit(X[:4000], y[:4000])
    rmse = np.sqrt(np.mean((m.predict(X[4000:]) - y[4000:]) ** 2))
    assert rmse < 1.4
    assert m.stop_reason_ == "converged" and 20 < m.best_iteration_ < 2000
    assert len(m.booster_.trees) == m.best_iteration_
    h = m.booster_.history
    assert min(h) == h[m.best_iteration_ - 1]


def test_multiclass_poisson_and_determinism():
    r = np.random.default_rng(4)
    X = r.normal(size=(3000, 4))
    y = np.digitize(X[:, 0] + 0.3 * r.normal(size=3000), [-0.5, 0.5])
    a = GBDTClassifier(seed=7, subsample=0.8, colsample=0.75).fit(X, y)
    b = GBDTClassifier(seed=7, subsample=0.8, colsample=0.75).fit(X, y)
    P = a.predict_proba(X)
    assert P.shape == (3000, 3) and np.allclose(P.sum(1), 1.0) and np.array_equal(P, b.predict_proba(X))
    assert np.mean(a.predict(X) == y) > 0.8
    labels = np.array(["lo", "mid", "hi"], dtype=object)[y]
    c = GBDTClassifier(n_estimators=20, early_stopping=False).fit(X, labels)
    assert set(c.predict(X)) <= {"lo", "mid", "hi"}
    yc = r.poisson(np.exp(0.7 * X[:, 0])).astype(float)
    pm = GBDTRegressor(objective="poisson").fit(X, yc)
    mu = pm.predict(X)
    assert np.all(mu > 0) and np.corrcoef(np.log(mu), 0.7 * X[:, 0])[0, 1] > 0.9


def test_single_class_is_refused():
    with pytest.raises(ValueError, match="two classes"):
        GBDTClassifier().fit(np.zeros((10, 1)), np.zeros(10))


def test_discovery_family_opt_in_and_default_without_libraries(monkeypatch):
    from tensor_optix.discovery.ml.families import native_gbdt
    from tensor_optix.discovery.ml.families.base import get_family
    from tensor_optix.discovery.ml.trials import candidate_families
    fam = get_family("gbdt")
    monkeypatch.setattr(native_gbdt.importlib.util, "find_spec", lambda name: None)
    assert fam.is_default() and "gbdt" in [f.name for f in candidate_families("binary")]
    monkeypatch.setattr(native_gbdt.importlib.util, "find_spec", lambda name: object())
    assert not fam.is_default() and "gbdt" not in [f.name for f in candidate_families("binary")]
    assert "gbdt" in [f.name for f in candidate_families("binary", ["gbdt"])]
