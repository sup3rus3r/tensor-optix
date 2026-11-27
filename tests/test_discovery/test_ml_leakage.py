"""Leakage detection against planted leaks and legitimate strong features."""
import numpy as np

from tensor_optix.discovery.findings import Severity
from tensor_optix.discovery.ml.data import to_table
from tensor_optix.discovery.ml.evaluate import evaluate_family
from tensor_optix.discovery.ml.families.base import get_family, sigmoid
from tensor_optix.discovery.ml.leakage import detect_leakage, skill
from tensor_optix.discovery.ml.splits import KFold, RepeatedSplitter, StratifiedKFold
from tensor_optix.discovery.ml.target import profile_target
from tensor_optix.discovery.ml.types import infer_types

import synth


def _run(cols, y, target_name=None, time_order=None, folds=None):
    table = to_table(cols, target=y)
    if target_name:
        table.target_name = target_name
    types = infer_types(table, np.random.default_rng(0))
    prof, yenc, keep, _ = profile_target(table.target)
    if folds is None:
        folds = RepeatedSplitter(StratifiedKFold(5, 0) if prof.task in ("binary", "multiclass")
                                 else KFold(5, 0), 1).folds(len(yenc), y=yenc)
    dummy = evaluate_family(table, types, yenc, folds, get_family("baseline"), {}, task=prof.task,
                            n_classes=prof.n_classes, metric=prof.metric).mean
    full = evaluate_family(table, types, yenc, folds, get_family("linear"), {"alpha": 1e-3}, task=prof.task,
                           n_classes=prof.n_classes, metric=prof.metric).mean
    return detect_leakage(table, types, yenc, folds, task=prof.task, n_classes=prof.n_classes,
                          metric=prof.metric, seed=0, dummy_score=dummy, full_score=full,
                          time_order=time_order)


def test_skill_normalisation():
    assert skill(0.75, 0.5, "auc") == 0.5
    assert skill(0.0, 2.0, "rmse") == 1.0
    assert skill(1.0, 2.0, "rmse") == 0.5
    assert skill(None, 0.5, "auc") is None


def test_planted_leak_is_flagged_and_legit_strong_feature_is_not():
    cols, y = synth.leaky()
    summary, fs = _run(cols, y)
    by = {f.id: f for f in fs}
    leak = by.get("leakage.single_feature:col=refund_flag")
    assert leak is not None and leak.severity == Severity.WARNING and leak.default_applied
    assert leak.suggested_action.kind == "drop_column"
    assert "full_model_without_column" in leak.evidence[0].details
    assert not any("col=income" in k and v.severity.rank >= Severity.WARNING.rank for k, v in by.items())


def test_target_named_leak_is_critical():
    cols, y = synth.leaky()
    cols["churned_final"] = cols.pop("refund_flag")
    summary, fs = _run(cols, y, target_name="churned")
    f = {f.id: f for f in fs}["leakage.single_feature:col=churned_final"]
    assert f.severity == Severity.CRITICAL


def test_identifier_predicting_target_is_warned():
    rng = np.random.default_rng(1)
    n = 1200
    x = rng.normal(size=n)
    y = (np.arange(n) > n // 2).astype(int) ^ (rng.random(n) < 0.1)       # outcome depends on id order
    cols = {"x": x, "record_id": np.arange(n)}
    summary, fs = _run(cols, y.astype(int))
    assert any(f.id == "leakage.identifier_predicts:col=record_id" for f in fs)
    cols2, y2 = {"x": x, "record_id": np.arange(n)}, (rng.random(n) < sigmoid(x)).astype(int)
    summary, fs2 = _run(cols2, y2)
    assert not any(f.id.startswith("leakage.identifier_predicts") for f in fs2)


def test_missingness_signal():
    rng = np.random.default_rng(2)
    n = 1500
    x = rng.normal(size=n)
    y = (rng.random(n) < sigmoid(x)).astype(int)
    filled = np.where(y == 1, np.where(rng.random(n) < 0.7, np.nan, 1.0), 1.0) + rng.normal(size=n) * 0.01
    summary, fs = _run({"x": x, "followup_score": filled}, y)
    assert any(f.id == "leakage.missingness_predicts:col=followup_score" for f in fs)
    random_missing = np.where(rng.random(n) < 0.3, np.nan, rng.normal(size=n))
    summary, fs = _run({"x": x, "noise": random_missing}, y)
    assert not any(f.id.startswith("leakage.missingness_predicts") for f in fs)


def test_future_target_feature_in_time_ordered_data():
    rng = np.random.default_rng(3)
    n = 800
    y = np.cumsum(rng.normal(size=n))
    nxt = np.append(y[1:], np.nan)
    cols = {"x": rng.normal(size=n), "next_value": nxt}
    folds = RepeatedSplitter(KFold(5, 0), 1).folds(n)
    summary, fs = _run(cols, y, time_order=np.arange(n), folds=folds)
    f = {f.id: f for f in fs}.get("leakage.future_target:col=next_value")
    assert f is not None and f.severity == Severity.CRITICAL


def test_no_leaks_on_ordinary_data():
    for seed in range(3):
        r = np.random.default_rng(seed)
        X = r.normal(size=(1000, 6))
        y = (r.random(1000) < sigmoid(X @ np.array([0.8, -0.6, 0.4, 0.3, 0.0, 0.0]))).astype(int)
        summary, fs = _run({f"f{i}": X[:, i] for i in range(6)}, y)
        assert not any(f.severity.rank >= Severity.ADVISORY.rank for f in fs)
