"""Feature opportunities: tests, leakage-safe generators, planted opportunities, build round-trip."""
import warnings

import numpy as np
import pytest

from tensor_optix.discovery import discover
from tensor_optix.discovery.findings import Action, Severity
from tensor_optix.discovery.ml.data import to_table
from tensor_optix.discovery.ml.feature_gen import FeatureSpec, generate
from tensor_optix.discovery.ml.features import ljung_box, mcleod_li, skewness
from tensor_optix.discovery.ml.families.base import sigmoid

FAST = ["linear", "naive_bayes", "knn"]


def _ar(n, phi, r):
    e = np.zeros(n)
    for t in range(1, n):
        e[t] = phi * e[t - 1] + r.normal()
    return e


def _feature_ids(rep):
    return sorted(f.id for f in rep.findings if f.category.value == "features")


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

def test_ljung_box_and_mcleod_li():
    r = np.random.default_rng(0)
    assert ljung_box(r.normal(size=2000))[1] > 0.01
    assert ljung_box(_ar(2000, 0.5, r))[1] < 1e-10
    # GARCH(1,1)-like volatility clustering, no autocorrelation in levels
    n = 3000
    eps, s2 = np.zeros(n), np.ones(n)
    for t in range(1, n):
        s2[t] = 0.1 + 0.2 * eps[t - 1] ** 2 + 0.7 * s2[t - 1]
        eps[t] = np.sqrt(s2[t]) * r.normal()
    pos = [np.arange(n)]
    assert mcleod_li(eps, pos)[1] < 1e-6
    assert mcleod_li(r.normal(size=n), pos)[1] > 0.01
    assert abs(skewness(r.lognormal(0, 1, 5000))) > 2


# ---------------------------------------------------------------------------
# generators are leakage-safe
# ---------------------------------------------------------------------------

def test_target_lags_never_see_current_or_future_target():
    r = np.random.default_rng(1)
    n = 200
    y = r.normal(size=n)
    order = r.permutation(n)                           # time order != row order
    t = to_table({"x": r.normal(size=n)})
    spec = FeatureSpec("target_lags", {"lags": [1, 2], "windows": [3], "time": "<row order>"})
    base = generate(spec, t, y, order)
    pos_in_time = np.empty(n, dtype=int)
    pos_in_time[order] = np.arange(n)
    for i in (5, 50, 150):
        y2 = y.copy()
        y2[i] += 100.0
        new = generate(spec, t, y2, order)
        unaffected = pos_in_time <= pos_in_time[i]     # the row itself and everything before it
        for k in base:
            a, b = base[k][unaffected], new[k][unaffected]
            assert np.array_equal(np.isnan(a), np.isnan(b)) and np.allclose(a[~np.isnan(a)], b[~np.isnan(b)]), k


def test_series_are_separate_per_group():
    y = np.arange(10, dtype=float)
    groups = np.array([0, 1] * 5)
    t = to_table({"x": np.zeros(10)})
    out = generate(FeatureSpec("target_lags", {"lags": [1], "windows": [2]}), t, y, np.arange(10), groups)
    lag = out["fx__y_lag1"]
    assert np.isnan(lag[0]) and np.isnan(lag[1])
    assert lag[2] == 0.0 and lag[3] == 1.0              # previous row of the *same* group


def test_rolling_features_use_current_and_past_rows_only():
    r = np.random.default_rng(2)
    x = r.normal(size=50)
    t = to_table({"x": x})
    spec = FeatureSpec("rolling_features", {"columns": ["x"], "windows": [5], "time": "<row order>"})
    base = generate(spec, t, None, np.arange(50))
    x2 = x.copy()
    x2[30] += 50
    new = generate(spec, to_table({"x": x2}), None, np.arange(50))
    for k in base:
        assert np.allclose(base[k][:30], new[k][:30], equal_nan=True)
    assert abs(base["fx__x_rmean5"][10] - x[6:11].mean()) < 1e-12


def test_row_wise_families():
    when = np.datetime64("2024-01-01") + np.arange(14).astype("timedelta64[D]")
    t = to_table({"when": when, "a": np.array([1.0, -10.0] * 7), "b": np.arange(14.0)})
    cyc = generate(FeatureSpec("calendar_cycles", {"columns": ["when"]}), t)
    assert np.allclose(cyc["fx__when_dow_sin"][:7], cyc["fx__when_dow_sin"][7:])   # weekly period
    assert np.all(np.abs(cyc["fx__when_month_cos"]) <= 1.0)
    lg = generate(FeatureSpec("log_transform", {"columns": ["a"]}), t)["fx__a_slog"]
    assert lg[1] == -np.log1p(10.0)
    ix = generate(FeatureSpec("interactions", {"columns": ["a", "b"]}), t)["fx__a_x_b"]
    assert np.array_equal(ix, t.columns["a"] * t.columns["b"])
    ga = generate(FeatureSpec("group_aggregates", {"columns": ["b"]}), t, groups=np.repeat([0, 1], 7))
    assert ga["fx__b_gmean"][0] == 3.0 and ga["fx__b_gmean"][13] == 10.0


def test_action_grammar_for_feature_families():
    a = Action("add_feature_family", FeatureSpec("log_transform", {"columns": ["a"]}).to_dict())
    assert a.grammar_version == 2
    assert Action("drop_column", {"column": "a"}).grammar_version == 1
    with pytest.raises(ValueError):
        FeatureSpec("not_a_family")


# ---------------------------------------------------------------------------
# planted opportunities (end to end)
# ---------------------------------------------------------------------------

def test_persistent_target_recommends_lags_and_build_uses_history():
    r = np.random.default_rng(3)
    n = 1500
    x = r.normal(size=n)
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = 0.8 * y[t - 1] + 0.5 * x[t] + r.normal(scale=0.5)
    when = np.datetime64("2020-01-01") + np.arange(n).astype("timedelta64[D]")
    perm = r.permutation(n)
    cols = {"when": when[perm], "x": x[perm]}
    rep = discover(cols, y[perm], families=FAST, seed=0)
    assert "features.target_lags" in _feature_ids(rep)
    f = rep.finding("features.target_lags")
    assert f.severity == Severity.ADVISORY and not f.default_applied
    assert "one-step-ahead" in f.statement
    rep.accept("features.target_lags")
    for fid in rep.profile.sections["applied_findings"]:
        rep.accept(fid)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sol = rep.build()
    assert sol.features and sol.features[0]["family"] == "target_lags"
    sol.fit(cols, y[perm])
    with pytest.raises(ValueError, match="past values of the target"):
        sol.predict(cols)
    hist = y[perm].copy()
    pred = sol.predict(cols, target=hist)
    assert np.sqrt(np.mean((pred - y[perm]) ** 2)) < 0.65    # lags explain most of the AR variance


def test_weekly_seasonality_recommends_calendar_cycles_not_lags_on_tied_dates():
    r = np.random.default_rng(4)
    n = 1500
    d = r.integers(0, 700, n)
    when = np.datetime64("2020-01-01") + d.astype("timedelta64[D]")
    y = 3 * np.sin(2 * np.pi * ((d + 2) % 7) / 7) + r.normal(size=n)
    rep = discover({"when": when, "z": r.normal(size=n)}, y, families=["linear"], seed=0)
    ids = _feature_ids(rep)
    assert "features.calendar_cycles" in ids and "features.target_lags" not in ids
    assert rep.profile.sections["features"]["not_tried"]
    assert "Feature opportunities" in rep.to_html()


def test_skewed_feature_recommends_log_transform():
    r = np.random.default_rng(5)
    n = 1500
    a = r.lognormal(0, 1.5, n)
    y = np.log(a) + r.normal(scale=0.3, size=n)
    rep = discover({"a": a, "b": r.normal(size=n)}, y, families=["linear"], seed=0)
    ids = _feature_ids(rep)
    # for a log-normal column the quantile (rank-to-normal) transform *is* the log, so both may be recommended
    assert "features.log_transform" in ids and set(ids) <= {"features.log_transform", "preprocess.scaler:method=quantile"}


def test_interactions_for_additive_model():
    r = np.random.default_rng(6)
    n = 2000
    X = r.normal(size=(n, 3))
    y = (r.random(n) < sigmoid(2.0 * X[:, 0] * X[:, 1] + 0.5 * X[:, 2])).astype(int)
    rep = discover({f"x{i}": X[:, i] for i in range(3)}, y, families=["linear"], seed=0)
    assert "features.interactions" in _feature_ids(rep)


def test_entity_aggregates_for_grouped_data():
    r = np.random.default_rng(7)
    n_g, per = 200, 8
    g = np.repeat(np.arange(n_g), per)
    level = r.normal(size=n_g)[g]
    x = level + r.normal(scale=1.0, size=len(g))          # noisy reading of a per-entity level
    y = 2.0 * level + r.normal(scale=0.3, size=len(g))
    cols = {"entity": np.array([f"e{v}" for v in g], dtype=object), "x": x}
    rep = discover(cols, y, families=["linear"], seed=0, groups="entity")
    assert "features.group_aggregates" in _feature_ids(rep)


def test_ordinary_iid_data_gets_no_feature_recommendation():
    for seed in range(3):
        r = np.random.default_rng(10 + seed)
        X = r.normal(size=(1200, 4))
        y = X @ np.array([1.0, -0.5, 0.3, 0.0]) + r.normal(size=1200)
        rep = discover({f"x{i}": X[:, i] for i in range(4)}, y, families=FAST, seed=0)
        assert _feature_ids(rep) == []


@pytest.mark.slow
def test_feature_recommendation_false_alarm_rate():
    """Null datasets that trigger detection (skewed noise features, a datetime column, a
    persistent but irrelevant feature) must rarely get a feature family recommended."""
    from test_calibration import N, _report
    hits, ex = 0, []
    for s in range(N):
        r = np.random.default_rng(9000 + s)
        n = 600
        x = r.normal(size=n)
        y = x + r.normal(size=n)
        cols = {"x": x, "skewed_noise": r.lognormal(0, 1.5, n),
                "when": np.datetime64("2021-01-01") + r.permutation(n).astype("timedelta64[D]"),
                "persistent_noise": _ar(n, 0.9, r)}
        rep = discover(cols, y, families=["linear"], seed=s, holdout=0)
        ids = _feature_ids(rep)
        if ids:
            hits += 1
            ex.append((s, ids))
    _report("features", hits, N, ex)


# ---------------------------------------------------------------------------
# scaler recommendation
# ---------------------------------------------------------------------------

def test_quantile_scaling_is_fold_safe_and_rank_based():
    from tensor_optix.discovery.ml.preprocess import ColumnPipeline, PipelineOptions
    from tensor_optix.discovery.ml.types import ColumnType
    r = np.random.default_rng(20)
    x = r.standard_cauchy(500)
    t = to_table({"x": x})
    types = {"x": ColumnType(name="x", type="numeric_continuous", confidence="high", declared=True)}
    pipe = ColumnPipeline(types, PipelineOptions(quantile_columns=("x",)), task="regression", n_classes=0, seed=0)
    tr = np.arange(400)
    Xtr = pipe.fit_transform(t, tr, x[tr])
    Xte = pipe.transform(t, np.arange(400, 500))
    assert np.all(np.abs(Xtr) < 3.2) and np.all(np.abs(Xte) < 3.2)          # bounded normal scores
    assert np.all(np.diff(Xtr[np.argsort(x[tr]), 0]) >= -1e-12)               # monotone in x
    assert abs(np.median(Xtr)) < 0.1


def test_heavy_tailed_feature_gets_a_scaler_for_knn():
    r = np.random.default_rng(21)
    n = 2000
    a = r.standard_cauchy(n)
    b = r.normal(size=n)
    y = np.tanh(a) * 2 + b + r.normal(scale=0.3, size=n)
    rep = discover({"a": a, "b": b}, y, families=["knn"], seed=0)
    ids = _feature_ids(rep)
    assert any(i.startswith("preprocess.scaler:") for i in ids), ids
    fid = next(i for i in ids if i.startswith("preprocess.scaler:"))
    rep.accept(fid)
    for x in rep.profile.sections["applied_findings"]:
        rep.accept(x)
    sol = rep.build()
    assert sol.scaling.get("a") in ("robust", "quantile")
    cv = sol.cross_validate({"a": a, "b": b}, y)
    base = next(x for x in rep.experiments if x.stage == "trial" and x.family == "knn"
                and "scaler" not in x.config)
    assert cv.mean < base.mean                                               # rmse: lower is better


def test_trees_get_no_scaler_and_say_why():
    pytest.importorskip("lightgbm")
    r = np.random.default_rng(22)
    n = 1200
    a = r.standard_cauchy(n)
    y = np.tanh(a) + r.normal(scale=0.3, size=n)
    rep = discover({"a": a, "b": r.normal(size=n)}, y, families=["lightgbm"], seed=0, holdout=0)
    assert not any(i.startswith("preprocess.scaler:") for i in _feature_ids(rep))
    assert "invariant" in rep.profile.sections["features"]["scaling"]["note"]
