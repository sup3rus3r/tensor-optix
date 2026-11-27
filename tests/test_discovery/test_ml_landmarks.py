"""Landmark stage: signal / non-linearity / linear-sufficient hints on known problems."""
import numpy as np

from tensor_optix.discovery.findings import Severity
from tensor_optix.discovery.ml.data import to_table
from tensor_optix.discovery.ml.families.base import sigmoid
from tensor_optix.discovery.ml.landmarks import run_landmarks
from tensor_optix.discovery.ml.splits import KFold, RepeatedSplitter, StratifiedKFold
from tensor_optix.discovery.ml.target import profile_target
from tensor_optix.discovery.ml.types import infer_types


def _lm(cols, y, repeats=2):
    table = to_table(cols, target=y)
    types = infer_types(table, np.random.default_rng(0))
    prof, yenc, keep, _ = profile_target(table.target)
    base = StratifiedKFold(5, 0) if prof.task in ("binary", "multiclass") else KFold(5, 0)
    folds = RepeatedSplitter(base, repeats).folds(len(yenc), y=yenc)
    return run_landmarks(table, types, yenc, folds, task=prof.task, n_classes=prof.n_classes,
                         metric=prof.metric, seed=0)


def test_linear_truth_gives_linear_sufficient():
    r = np.random.default_rng(0)
    X = r.normal(size=(1500, 5))
    y = (r.random(1500) < sigmoid(X @ np.array([1.0, -0.8, 0.6, 0.0, 0.0]))).astype(int)
    sig, results, comps, fs = _lm({f"f{i}": X[:, i] for i in range(5)}, y)
    assert sig["signal"] and not sig["nonlinear"] and sig["linear_sufficient"]
    assert [f.id for f in fs] == ["family.linear_sufficient"]
    assert {r.family for r in results} == {"baseline", "linear", "tree", "knn", "naive_bayes", "lda"}
    assert all(c.a != c.b for c in comps)


def test_xor_gives_nonlinear():
    r = np.random.default_rng(1)
    X = r.normal(size=(1500, 4))
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)
    y = np.where(r.random(1500) < 0.05, 1 - y, y)
    sig, results, comps, fs = _lm({f"f{i}": X[:, i] for i in range(4)}, y)
    # pure XOR has no marginal split gain, so greedy trees miss it; kNN must not
    assert sig["nonlinear"] and sig["local_structure"]
    f = {f.id: f for f in fs}["family.nonlinear_signal"]
    assert f.severity == Severity.INFO and len(f.evidence) >= 1


def test_threshold_interaction_beats_stump():
    r = np.random.default_rng(3)
    X = r.normal(size=(1500, 4))
    y = ((X[:, 0] > 0) & (X[:, 1] > 0.3)).astype(int)
    y = np.where(r.random(1500) < 0.05, 1 - y, y)
    sig, _, _, fs = _lm({f"f{i}": X[:, i] for i in range(4)}, y)
    assert sig["interactions"] and sig["verdicts"]["tree_d3_vs_linear"] == "better"


def test_pure_noise_gives_no_signal():
    hits = 0
    for seed in range(3):
        r = np.random.default_rng(10 + seed)
        X = r.normal(size=(1000, 5))
        y = r.integers(0, 2, 1000)
        sig, _, _, fs = _lm({f"f{i}": X[:, i] for i in range(5)}, y)
        hits += any(f.id == "family.no_signal" for f in fs)
        assert not sig["nonlinear"]
    assert hits == 3


def test_regression_landmarks():
    r = np.random.default_rng(2)
    X = r.uniform(-2, 2, size=(1200, 3))
    y = np.sin(2 * X[:, 0]) * 2 + X[:, 1] + r.normal(scale=0.2, size=1200)
    sig, results, _, fs = _lm({f"f{i}": X[:, i] for i in range(3)}, y)
    assert sig["metric"] == "rmse" and sig["signal"] and sig["nonlinear"]
    assert "naive_bayes" not in sig["scores"]
