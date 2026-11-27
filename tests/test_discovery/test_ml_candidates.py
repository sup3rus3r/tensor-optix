"""Candidate families with optional dependencies: MLP (torch), LightGBM, XGBoost."""
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from tensor_optix.discovery.ml.families.base import all_families, get_family
from tensor_optix.discovery.stats.metrics import roc_auc_binary

NAMES = {f.name for f in all_families()}


def _xor(n=1200, seed=0):
    r = np.random.default_rng(seed)
    X = r.normal(size=(n, 4))
    return X, ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)


def _check_family(name, task="binary"):
    fam = get_family(name)
    X, y = _xor()
    tr, te = slice(0, 900), slice(900, None)
    a = fam.make(fam.configs(900, 4, task)[0], 7, task, 2).fit(X[tr], y[tr])
    b = fam.make(fam.configs(900, 4, task)[0], 7, task, 2).fit(X[tr], y[tr])
    pa, pb = a.predict_proba(X[te]), b.predict_proba(X[te])
    assert pa.shape == (300, 2) and np.allclose(pa.sum(axis=1), 1.0)
    assert np.array_equal(pa, pb), "same seed must give identical predictions"
    assert roc_auc_binary(y[te], pa[:, 1]) > 0.93
    assert a.stop_reason in ("converged", "early_stopping", "max_epochs", "max_trees")


@pytest.mark.skipif("mlp" not in NAMES, reason="torch not installed")
def test_mlp_learns_xor_deterministically():
    _check_family("mlp")


@pytest.mark.skipif("mlp" not in NAMES, reason="torch not installed")
def test_mlp_regression_inverts_target_scaling():
    r = np.random.default_rng(1)
    X = r.normal(size=(800, 3))
    y = 1000.0 + 50.0 * X[:, 0] + r.normal(scale=5.0, size=800)
    fam = get_family("mlp")
    est = fam.make(fam.configs(600, 3, "regression")[0], 0, "regression", 0).fit(X[:600], y[:600])
    pred = est.predict(X[600:])
    assert abs(pred.mean() - y[600:].mean()) < 10.0
    assert np.sqrt(np.mean((pred - y[600:]) ** 2)) < 15.0


@pytest.mark.parametrize("name", ["lightgbm", "xgboost"])
def test_boosting_learns_xor_deterministically(name):
    if name not in NAMES:
        pytest.skip(f"{name} not installed")
    _check_family(name)


@pytest.mark.parametrize("name", ["lightgbm", "xgboost"])
def test_boosting_multiclass_and_count(name):
    if name not in NAMES:
        pytest.skip(f"{name} not installed")
    fam = get_family(name)
    r = np.random.default_rng(2)
    X = r.normal(size=(900, 3))
    y3 = np.digitize(X[:, 0] + 0.3 * r.normal(size=900), [-0.5, 0.5])
    est = fam.make(fam.configs(900, 3, "multiclass")[0], 0, "multiclass", 3).fit(X, y3)
    P = est.predict_proba(X)
    assert P.shape == (900, 3) and np.allclose(P.sum(axis=1), 1.0, atol=1e-6)
    yc = r.poisson(np.exp(0.5 * X[:, 0]))
    est = fam.make(fam.configs(900, 3, "count")[0], 0, "count", 0).fit(X, yc)
    assert np.all(est.predict(X) >= 0)


def test_registry_without_optional_dependencies():
    """With torch / lightgbm / xgboost unimportable, the registry offers only the numpy families."""
    code = textwrap.dedent("""
        import sys
        class Block:
            def find_spec(self, name, path=None, target=None):
                if name.split('.')[0] in ('torch', 'lightgbm', 'xgboost'):
                    raise ModuleNotFoundError(name)
                return None
        sys.meta_path.insert(0, Block())
        from tensor_optix.discovery.ml.families.base import all_families
        print(sorted(f.name for f in all_families()))
    """)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    names = eval(out.stdout.strip().splitlines()[-1])
    assert names == ["baseline", "knn", "lda", "linear", "naive_bayes", "tree"]
