"""
Calibration of discovery's detectors on null datasets (no planted effect).

Each detector must raise a non-INFO finding on a null dataset rarely: the
exact (Clopper-Pearson) upper 95% bound of the false-alarm rate must be
<= 0.08. Seeded, so every run is identical. Slow: deselect with -m "not slow".
``DISCOVERY_CALIBRATION_N`` overrides the number of datasets (default 200).
"""
import os

import numpy as np
import pytest

from tensor_optix.discovery.findings import Severity
from tensor_optix.discovery.ml.data import to_table
from tensor_optix.discovery.ml.evaluate import evaluate_family
from tensor_optix.discovery.ml.families.base import get_family, sigmoid
from tensor_optix.discovery.ml.landmarks import run_landmarks
from tensor_optix.discovery.ml.leakage import detect_leakage
from tensor_optix.discovery.ml.shift import detect_shift
from tensor_optix.discovery.ml.splits import RepeatedSplitter, StratifiedKFold
from tensor_optix.discovery.ml.structure import detect_structure
from tensor_optix.discovery.ml.target import profile_target
from tensor_optix.discovery.ml.types import infer_types
from tensor_optix.discovery.stats.distributions import betainc

import synth

N = int(os.environ.get("DISCOVERY_CALIBRATION_N", "200"))
MAX_UPPER = 0.08
pytestmark = pytest.mark.slow


def clopper_pearson_upper(k: int, n: int, level: float = 0.95) -> float:
    """Exact one-sided upper bound: the p with P(X <= k | n, p) = 1 - level."""
    if k >= n:
        return 1.0
    alpha = 1.0 - level
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        # P(X <= k) = 1 - I_mid(k + 1, n - k)
        if 1.0 - betainc(k + 1, n - k, mid) > alpha:
            lo = mid
        else:
            hi = mid
    return hi


def test_clopper_pearson_known_values():
    assert abs(clopper_pearson_upper(0, 200) - (1 - 0.05 ** (1 / 200))) < 1e-9
    assert clopper_pearson_upper(5, 200) < 0.06 < clopper_pearson_upper(7, 200)


def _alarms(findings, prefix):
    return [f.id for f in findings if f.id.startswith(prefix) and f.severity.rank >= Severity.ADVISORY.rank]


def _null_classification(seed, n=600):
    r = np.random.default_rng(seed)
    X = r.normal(size=(n, 5))
    y = (r.random(n) < sigmoid(X @ np.array([0.9, -0.6, 0.4, 0.0, 0.0]))).astype(int)
    cols = {f"f{i}": X[:, i] for i in range(5)}
    cols["customer_id"] = synth._obj([f"c{v:03d}" for v in r.integers(0, 60, n)])
    cols["when"] = np.datetime64("2023-01-01") + r.integers(0, 900, n).astype("timedelta64[D]")
    return cols, y


def _setup(cols, y, seed):
    table = to_table(cols, target=y)
    types = infer_types(table, np.random.default_rng(seed))
    prof, yenc, _, _ = profile_target(table.target)
    folds = RepeatedSplitter(StratifiedKFold(5, seed), 1).folds(len(yenc), y=yenc)
    return table, types, prof, yenc, folds


def _report(name, hits, n, examples):
    upper = clopper_pearson_upper(hits, n)
    assert upper <= MAX_UPPER, f"{name}: {hits}/{n} false alarms (upper 95% bound {upper:.3f}); e.g. {examples[:5]}"


def test_structure_false_alarm_rate():
    hits, ex = 0, []
    for s in range(N):
        cols, y = _null_classification(1000 + s)
        table, types, prof, yenc, _ = _setup(cols, y, s)
        sp, fs = detect_structure(table, types, yenc, task=prof.task, n_classes=prof.n_classes,
                                  metric=prof.metric, seed=s)
        a = _alarms(fs, "structure.")
        if a:
            hits += 1
            ex.append((s, a))
    _report("structure", hits, N, ex)


def test_leakage_false_alarm_rate():
    hits, ex = 0, []
    for s in range(N):
        cols, y = _null_classification(2000 + s)
        cols.pop("when")
        table, types, prof, yenc, folds = _setup(cols, y, s)
        kw = dict(task=prof.task, n_classes=prof.n_classes, metric=prof.metric, seed=s)
        dummy = evaluate_family(table, types, yenc, folds, get_family("baseline"), {}, **kw).mean
        full = evaluate_family(table, types, yenc, folds, get_family("linear"), {"alpha": 1e-3}, **kw).mean
        _, fs = detect_leakage(table, types, yenc, folds, dummy_score=dummy, full_score=full, **kw)
        a = _alarms(fs, "leakage.")
        if a:
            hits += 1
            ex.append((s, a))
    _report("leakage", hits, N, ex)


def test_shift_false_alarm_rate():
    hits, ex = 0, []
    for s in range(N):
        cols, y = _null_classification(3000 + s)
        test, _ = _null_classification(4000 + s)
        cols.pop("when"), test.pop("when")
        table = to_table(cols, target=y)
        types = infer_types(table, np.random.default_rng(s))
        _, fs = detect_shift(table, types, test=to_table(test), seed=s)
        a = _alarms(fs, "shift.")
        if a:
            hits += 1
            ex.append((s, a))
    _report("shift", hits, N, ex)


def test_drift_false_alarm_rate():
    hits, ex = 0, []
    for s in range(N):
        cols, y = _null_classification(5000 + s)
        cols.pop("when")
        table = to_table(cols, target=y)
        types = infer_types(table, np.random.default_rng(s))
        _, fs = detect_shift(table, types, y=y, time_order=np.arange(len(y)), seed=s)
        a = _alarms(fs, "shift.")
        if a:
            hits += 1
            ex.append((s, a))
    _report("drift", hits, N, ex)


def test_landmark_nonlinear_false_alarm_rate():
    """Linear truth: the 'non-linear' hint must rarely fire."""
    hits, ex = 0, []
    for s in range(N):
        cols, y = _null_classification(6000 + s)
        cols.pop("when"), cols.pop("customer_id")
        table, types, prof, yenc, folds = _setup(cols, y, s)
        sig, _, _, fs = run_landmarks(table, types, yenc, folds, task=prof.task, n_classes=prof.n_classes,
                                      metric=prof.metric, seed=s)
        if sig["nonlinear"]:
            hits += 1
            ex.append((s, sig["verdicts"]))
    _report("landmarks.nonlinear", hits, N, ex)


def test_pooled_dm_comparison_calibrated_on_persistent_series():
    """Two exchangeable families on strongly autocorrelated data: the pooled
    Diebold-Mariano comparison must rarely declare a winner (Andrews bandwidth)."""
    from dataclasses import replace
    from tensor_optix.discovery.ml.recommend import PooledLosses, _dm_compare
    from tensor_optix.discovery.ml.splits import ForwardChaining
    from tensor_optix.discovery.stats.compare import default_rope
    hits, ex = 0, []
    for s in range(N):
        r = np.random.default_rng(7000 + s)
        n, phi = 1000, 0.95
        X = r.normal(size=(n, 2))
        e = np.zeros(n)
        for t in range(1, n):
            X[t] = phi * X[t - 1] + np.sqrt(1 - phi ** 2) * X[t]
            e[t] = phi * e[t - 1] + r.normal()
        table = to_table({"a": X[:, 0], "b": X[:, 1]}, target=e)
        types = infer_types(table, np.random.default_rng(s))
        folds = RepeatedSplitter(ForwardChaining(7, gap=5), 1).folds(n)
        res, rows = {}, {}
        for name, drop in (("A", ("b",)), ("B", ("a",))):
            arr = np.full(n, np.nan)
            x = evaluate_family(table, types, e, folds, get_family("linear"), {"alpha": 1e-3},
                                task="regression", n_classes=0, metric="rmse", row_losses=arr, drop=drop)
            res[name] = replace(x, family=name)
            rows[res[name].key] = arr
        c = _dm_compare(res["A"], res["B"], PooledLosses(rows, np.arange(n), 5, "rmse"),
                        default_rope("rmse", target_std=float(np.std(e))))
        if c.verdict in ("better", "worse"):
            hits += 1
            ex.append((s, c.delta_interval))
    _report("pooled_dm", hits, N, ex)
