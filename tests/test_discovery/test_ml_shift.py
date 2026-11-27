"""Shift detection against planted covariate shift, null data, overlap and drift."""
import math

import numpy as np

from tensor_optix.discovery.findings import Severity
from tensor_optix.discovery.ml.data import to_table
from tensor_optix.discovery.ml.shift import _chi2_categorical, detect_shift
from tensor_optix.discovery.ml.types import infer_types
from tensor_optix.discovery.stats import chi2_sf

import synth


def _shift(train_cols, test_cols, y=None, time_order=None):
    tr = to_table(train_cols, target=y)
    te = to_table(test_cols) if test_cols is not None else None
    types = infer_types(tr, np.random.default_rng(0))
    return detect_shift(tr, types, test=te, y=y, time_order=time_order, seed=0)


def test_chi2_sf_tabulated():
    for x, df, p in [(3.841458820694124, 1, 0.05), (5.991464547107979, 2, 0.05),
                     (18.307038053275146, 10, 0.05), (6.634896601021214, 1, 0.01),
                     (2.0, 4, math.exp(-1.0) * 2.0),             # df=4: sf = e^{-x/2}(1 + x/2)
                     (100.0, 80, 0.064570368921133), (0.5, 3, 0.9188914116546758)]:   # scipy.stats.chi2.sf
        assert abs(chi2_sf(x, df) - p) < 1e-12
    for x in (0.3, 1.0, 7.5, 40.0):                              # df=2: sf = e^{-x/2}
        assert abs(chi2_sf(x, 2) - math.exp(-x / 2)) < 1e-12
    assert chi2_sf(0.0, 3) == 1.0


def test_planted_covariate_shift_found_and_named():
    cols, y, test = synth.shifted()
    summary, fs = _shift(cols, test, y)
    ids = {f.id: f for f in fs}
    assert "shift.feature:col=f2" in ids and "shift.feature:col=f5" in ids
    flagged = {i for i in ids if i.startswith("shift.feature:")}
    assert flagged == {"shift.feature:col=f2", "shift.feature:col=f5"}
    adv = ids["shift.adversarial"]
    assert adv.severity.rank >= Severity.ADVISORY.rank
    assert set(adv.evidence[0].details["top_shifted"]) == {"f2", "f5"}


def test_no_shift_on_same_distribution():
    for seed in range(4):
        r = np.random.default_rng(seed)
        X, T = r.normal(size=(1500, 8)), r.normal(size=(1500, 8))
        cat = np.array(list("abcde"), dtype=object)
        cols = {**{f"f{i}": X[:, i] for i in range(8)}, "c": cat[r.integers(0, 5, 1500)]}
        test = {**{f"f{i}": T[:, i] for i in range(8)}, "c": cat[r.integers(0, 5, 1500)]}
        summary, fs = _shift(cols, test)
        assert not any(f.severity.rank >= Severity.ADVISORY.rank for f in fs), [f.id for f in fs]
        assert 0.4 < summary["adversarial"]["auc"] < 0.6


def test_categorical_shift_and_rare_pooling():
    r = np.random.default_rng(5)
    cat = np.array(list("abcdefghij"), dtype=object)
    a = cat[r.choice(10, 2000, p=[0.3, 0.2, 0.2, 0.1, 0.1, 0.05, 0.03, 0.01, 0.005, 0.005])]
    b = cat[r.choice(10, 2000, p=[0.1, 0.1, 0.2, 0.3, 0.2, 0.05, 0.03, 0.01, 0.005, 0.005])]
    tv, p, df = _chi2_categorical(a, b)
    assert tv > 0.3 and p < 1e-20 and df >= 1
    tv0, p0, _ = _chi2_categorical(a[:1000], a[1000:])
    assert tv0 < 0.08 and p0 > 1e-3
    summary, fs = _shift({"c": a, "x": r.normal(size=2000)}, {"c": b, "x": r.normal(size=2000)})
    f = {f.id: f for f in fs}["shift.feature:col=c"]
    assert f.severity == Severity.WARNING


def test_missing_column_overlap_and_time_range():
    r = np.random.default_rng(6)
    n = 800
    x = r.normal(size=n)
    when = np.datetime64("2023-01-01") + np.arange(n).astype("timedelta64[D]")
    train = {"x": x, "z": r.normal(size=n), "when": when}
    later = np.datetime64("2026-01-01") + np.arange(400).astype("timedelta64[D]")
    xt = np.concatenate([x[:100], r.normal(size=300)])
    zt = np.concatenate([train["z"][:100], r.normal(size=300)])
    test = {"x": xt, "when": later}
    summary, fs = _shift(train, test)
    ids = {f.id: f for f in fs}
    assert ids["shift.missing_in_test:col=z"].severity == Severity.WARNING
    assert ids["shift.time_range:col=when"].severity == Severity.INFO
    assert not any(i.startswith("shift.feature:col=when") for i in ids)
    test2 = {"x": xt, "z": zt, "when": later}
    summary, fs = _shift(train, test2)
    ov = {f.id: f for f in fs}["shift.train_test_overlap"]
    assert abs(summary["test_rows_in_train"] - 0.25) < 1e-9 and ov.severity == Severity.WARNING


def test_drift_over_time_and_target_drift():
    r = np.random.default_rng(7)
    n = 1500
    t = np.arange(n)
    trend = r.normal(size=n) + 1.5 * t / n
    stable = r.normal(size=n)
    y = stable + 2.0 * t / n + r.normal(scale=0.3, size=n)
    perm = r.permutation(n)
    order = np.argsort(t[perm], kind="stable")
    summary, fs = _shift({"trend": trend[perm], "stable": stable[perm]}, None, y=y[perm], time_order=order)
    ids = {f.id for f in fs}
    assert "shift.drift_over_time:col=trend" in ids
    assert "shift.drift_over_time:col=stable" not in ids
    assert "shift.target_drift" in ids
    y2 = stable + r.normal(scale=0.3, size=n)
    summary, fs = _shift({"stable": stable}, None, y=y2, time_order=np.arange(n))
    assert not fs
