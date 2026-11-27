"""
Discovery ML phase 2: data container, feature-type inference, target
characterisation and quality checks -- each against planted ground truth.
"""
import numpy as np
import pytest

from tensor_optix.discovery.ml.data import is_dataframe, missing_mask, to_table
from tensor_optix.discovery.ml.quality import quality_profile, row_group_ids
from tensor_optix.discovery.ml.target import decide_task, encode_classes, profile_target
from tensor_optix.discovery.ml.types import infer_column_type, infer_types, parse_embedded_number, type_findings

RNG = np.random.default_rng(0)


def _t(col, declared=None):
    return infer_column_type("c", np.asarray(col) if not isinstance(col, np.ndarray) else col,
                             np.random.default_rng(0), declared)


def _obj(values):
    a = np.empty(len(values), dtype=object)
    a[:] = values
    return a


# ---------------------------------------------------------------- data container

def test_numpy_input_with_names_and_target_by_name():
    X = np.column_stack([np.arange(20.0), np.arange(20.0) * 2, np.arange(20) % 2])
    t = to_table(X, target="y", column_names=["a", "b", "y"])
    assert t.names == ["a", "b"] and t.target_name == "y" and len(t.target) == 20


def test_xy_tuple_dict_and_structured_inputs():
    X, y = np.ones((10, 3)), np.arange(10)
    t = to_table((X, y))
    assert t.names == ["x0", "x1", "x2"] and list(t.target) == list(y)
    t = to_table({"u": np.arange(5), "v": ["a"] * 5}, target=np.arange(5))
    assert t.columns["v"].dtype == object
    s = np.zeros(4, dtype=[("p", float), ("q", "U3")])
    t = to_table(s)
    assert t.names == ["p", "q"] and t.columns["q"].dtype == object


def test_groups_and_time_are_removed_from_features():
    t = to_table({"a": np.arange(6.0), "g": [1, 1, 2, 2, 3, 3], "ts": np.arange(6)},
                 target=np.arange(6), groups="g", time="ts")
    assert t.names == ["a"] and t.groups_name == "g" and t.time_name == "ts"


def test_input_errors():
    with pytest.raises(ValueError):
        to_table(np.ones((5, 2)), column_names=["a"])
    with pytest.raises(ValueError):
        to_table(np.ones((5, 2)), column_names=["a", "a"])
    with pytest.raises(KeyError):
        to_table({"a": np.arange(3)}, target="nope")
    with pytest.raises(ValueError):
        to_table({"a": np.arange(3)}, target=np.arange(4))
    with pytest.raises(KeyError):
        to_table({"a": np.arange(3)}, column_types={"zz": "numeric_continuous"})


def test_missing_values_are_normalised():
    t = to_table({"o": ["x", None, float("nan"), "y"], "f": [1.0, np.nan, 2.0, 3.0]})
    assert list(missing_mask(t.columns["o"])) == [False, True, True, False]
    assert list(missing_mask(t.columns["f"])) == [False, True, False, False]


def test_fingerprint_is_stable_and_sensitive():
    d = {"a": np.arange(50.0), "b": ["x"] * 50}
    f1 = to_table(d, target=np.arange(50)).fingerprint()
    f2 = to_table(d, target=np.arange(50)).fingerprint()
    d2 = {"a": np.arange(50.0) + 1, "b": ["x"] * 50}
    assert f1 == f2 and f1 != to_table(d2, target=np.arange(50)).fingerprint()


def test_is_dataframe_never_imports_pandas_and_is_duck_typed():
    class Fake:
        columns = ["a"]
        iloc = None
    Fake.__module__ = "pandas.core.frame"
    assert is_dataframe(Fake()) and not is_dataframe(np.ones(3)) and not is_dataframe({"a": 1})


def test_pandas_dtypes():
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({
        "cat": pd.Categorical(["a", "b", None, "a"]),
        "nint": pd.array([1, None, 3, 4], dtype="Int64"),
        "ts": pd.to_datetime(["2024-01-01", None, "2024-03-01", "2024-04-01"]).tz_localize("UTC"),
        "s": pd.array(["x", None, "z", "x"], dtype="string"),
        "b": pd.array([True, None, False, True], dtype="boolean"),
        "y": [0, 1, 0, 1],
    })
    t = to_table(df, target="y")
    assert t.columns["cat"].dtype == object and t.columns["cat"][2] is None
    assert t.columns["nint"].dtype.kind == "f" and np.isnan(t.columns["nint"][1])
    assert t.columns["ts"].dtype.kind == "M" and np.isnat(t.columns["ts"][1])
    assert t.columns["s"][1] is None and t.columns["b"][1] is None
    with pytest.raises(ValueError):
        to_table(df, column_names=["x"])


# ---------------------------------------------------------------- feature types

def test_numeric_types():
    assert _t(RNG.normal(size=500)).type == "numeric_continuous"
    assert _t(RNG.integers(0, 5, size=500)).type == "numeric_discrete"
    assert _t(RNG.integers(0, 2, size=500).astype(float)).type == "boolean"
    assert _t(np.arange(1000)).type == "identifier"                         # row counter
    assert _t(RNG.permutation(1000) + 5000).type == "identifier"            # shuffled consecutive ids
    assert _t(RNG.integers(0, 100_000, size=1000)).type == "numeric_continuous"  # not consecutive


def test_constant_and_near_constant():
    assert _t(np.ones(100)).type == "constant"
    x = np.zeros(1000)
    x[0] = 1.0
    assert _t(x).type == "constant"
    x = np.zeros(1000)
    x[:30] = 1.0
    assert _t(x).type == "boolean"


def test_string_types():
    n = 400
    assert _t(_obj(["yes", "no"] * (n // 2))).type == "boolean"
    assert _t(_obj([str(v) for v in RNG.normal(size=n)])).type == "numeric_continuous"
    assert _t(_obj([f"${v:,.2f}" for v in RNG.uniform(10, 5000, n)])).type == "embedded_number"
    assert _t(_obj([f"{v:.1f}kg" for v in RNG.uniform(1, 90, n)])).type == "embedded_number"
    assert _t(_obj([f"2024-{1 + i % 12:02d}-{1 + i % 28:02d}" for i in range(n)])).type == "datetime"
    import uuid
    assert _t(_obj([str(uuid.UUID(int=int(RNG.integers(0, 2 ** 62)))) for _ in range(n)])).type == "identifier"
    assert _t(_obj([f"https://example.com/p/{i}" for i in range(n)])).type == "url"
    words = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]
    text = [" ".join(RNG.choice(words, 6)) + f" {i}" for i in range(n)]
    assert _t(_obj(text)).type == "text"
    assert _t(_obj(list(RNG.choice(["red", "green", "blue"], n)))).type == "categorical_low"
    assert _t(_obj([f"city{v}" for v in RNG.integers(0, 120, n)])).type == "categorical_high"
    assert _t(_obj([f"user_{i}" for i in range(n)])).type == "identifier"


def test_datetime_dtype_and_all_missing_and_declared():
    assert _t(np.arange("2024-01-01", "2024-03-01", dtype="datetime64[D]")).type == "datetime"
    assert _t(_obj([None] * 10)).type == "unknown"
    ct = _t(np.arange(1000), declared="categorical_high")
    assert ct.type == "categorical_high" and ct.declared
    with pytest.raises(ValueError):
        _t(np.arange(10), declared="banana")


def test_parse_embedded_number():
    assert parse_embedded_number("$1,200") == 1200.0
    assert parse_embedded_number("-$5.5") == -5.5
    assert parse_embedded_number("12kg") == 12.0
    assert parse_embedded_number("15 %") == 15.0
    assert np.isnan(parse_embedded_number("n/a")) and np.isnan(parse_embedded_number(None))


def test_type_findings_are_reversible_and_skip_declared():
    t = to_table({"id": np.arange(500), "c": np.ones(500), "price": _obj([f"${i},000" for i in range(500)]),
                  "z": RNG.normal(size=500)}, column_types={"z": "numeric_continuous"})
    types = infer_types(t, np.random.default_rng(0))
    fs = {f.id: f for f in type_findings(types)}
    assert "data_types.identifier:col=id" in fs and fs["data_types.identifier:col=id"].suggested_action.kind == "drop_column"
    assert "data_types.constant:col=c" in fs
    assert "data_types.embedded_number:col=price" in fs
    assert not any("col=z" in k for k in fs)


# ---------------------------------------------------------------- target

def test_task_detection():
    assert decide_task(_obj(["a", "b"] * 10))[0] == "binary"
    assert decide_task(_obj(["a", "b", "c"] * 10))[0] == "multiclass"
    assert decide_task(np.array([0, 1] * 50))[0] == "binary"
    assert decide_task(RNG.integers(0, 4, 400))[0] == "multiclass"
    assert decide_task(RNG.normal(size=400))[0] == "regression"
    assert decide_task(RNG.poisson(30, 400))[0] == "count"
    assert decide_task(RNG.integers(-200, 200, 400))[0] == "regression"
    assert decide_task(RNG.integers(0, 4, 400), "regression") == ("regression", "declared by the user")
    with pytest.raises(ValueError):
        decide_task(np.ones(20))
    with pytest.raises(ValueError):
        decide_task(np.arange(20), "banana")


def test_encode_classes_is_deterministic():
    enc, classes = encode_classes(_obj(["b", "a", "c", "a"]))
    assert classes == ("a", "b", "c") and list(enc) == [1, 0, 2, 0]


def test_target_profile_findings():
    y = np.array([0] * 960 + [1] * 30 + [2] * 10)
    prof, enc, keep, fs = profile_target(y.astype(object).astype(str))
    ids = {f.id for f in fs}
    assert prof.task == "multiclass" and prof.metric == "macro_auc"
    assert {"target.imbalance", "target.rare_classes", "target.metric"} <= ids
    yb = np.array(["yes", "no"] * 50, dtype=object)
    yb[3] = None
    prof, enc, keep, fs = profile_target(yb)
    assert prof.task == "binary" and prof.metric == "auc" and prof.n_missing == 1 and keep.sum() == 99
    assert "target.missing" in {f.id for f in fs}
    prof, enc, keep, fs = profile_target(RNG.lognormal(0, 1.5, 500))
    assert prof.task == "regression" and "target.skewed" in {f.id for f in fs}
    with pytest.raises(ValueError):
        profile_target(np.array([1.0, 2.0, np.nan]))


# ---------------------------------------------------------------- quality

def test_duplicates_conflicts_missing_collinear_outliers():
    n = 300
    a = RNG.normal(size=n)
    cols = {"a": a.copy(), "b": a * 2 + 1e-6 * RNG.normal(size=n), "c": RNG.normal(size=n),
            "m": np.where(RNG.random(n) < 0.6, np.nan, 1.0) + RNG.normal(size=n)}
    cols["c"][:3] = 1e6
    for k in cols:
        cols[k][10:20] = cols[k][0:10]                # 10 duplicated rows
    y = np.zeros(n)
    y[10:15] = 1                                     # 5 of the duplicates conflict
    t = to_table(cols, target=y)
    types = infer_types(t, np.random.default_rng(0))
    summary, fs, dup = quality_profile(t, types, y)
    ids = {f.id for f in fs}
    assert summary["duplicate_rows"] == 20 and summary["conflicting_duplicate_rows"] == 10
    assert "quality.duplicate_rows" in ids
    assert "quality.high_missing:col=m" in ids
    assert "quality.collinear:cols=a|b" in ids
    assert "quality.outliers:col=c" in ids
    assert dup[0] == dup[10] and dup[1] != dup[0]


def test_row_group_ids_handles_missing_and_objects():
    t = to_table({"o": ["x", None, "x", None], "f": [1.0, np.nan, 1.0, np.nan]})
    g = row_group_ids(t, ["o", "f"])
    assert g[0] == g[2] and g[1] == g[3] and g[0] != g[1]
