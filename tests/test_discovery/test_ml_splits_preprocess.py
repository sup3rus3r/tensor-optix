"""
Splitters (property tests) and fold-safe preprocessing (no information from
test rows may reach the fitted state).
"""
import numpy as np
import pytest

from tensor_optix.discovery.ml.data import to_table
from tensor_optix.discovery.ml.preprocess import ColumnPipeline, PipelineOptions, expm1_target, log1p_target
from tensor_optix.discovery.ml.splits import (
    ForwardChaining, GroupKFold, KFold, RepeatedSplitter, StratifiedGroupKFold, StratifiedKFold,
    carve_holdout, make_splitter, splitter_from_dict, splitter_to_dict,
)
from tensor_optix.discovery.ml.types import infer_types


def _check_partition(folds, n):
    tests = np.concatenate([te for _, te in folds])
    assert len(tests) == n and len(np.unique(tests)) == n
    for tr, te in folds:
        assert len(np.intersect1d(tr, te)) == 0
        assert len(tr) + len(te) == n


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("n", [37, 200, 1001])
def test_kfold_and_stratified_partition(seed, n):
    rng = np.random.default_rng(seed)
    _check_partition(KFold(5, seed).split(n), n)
    y = rng.choice(3, n, p=[0.7, 0.2, 0.1])
    folds = StratifiedKFold(5, seed).split(n, y=y)
    _check_partition(folds, n)
    for c in range(3):
        per_fold = [int(np.sum(y[te] == c)) for _, te in folds]
        assert max(per_fold) - min(per_fold) <= 1
    sizes = [len(te) for _, te in folds]
    assert max(sizes) - min(sizes) <= 3


@pytest.mark.parametrize("seed", range(5))
def test_group_splitters_never_split_a_group(seed):
    rng = np.random.default_rng(seed)
    n = 600
    groups = rng.integers(0, 70, n)
    y = rng.integers(0, 2, n)
    for sp in (GroupKFold(5, seed), StratifiedGroupKFold(5, seed)):
        folds = sp.split(n, y=y, groups=groups)
        _check_partition(folds, n)
        for tr, te in folds:
            assert len(te) > 0 and len(tr) > 0
            assert len(np.intersect1d(groups[tr], groups[te])) == 0
    sizes = [len(te) for _, te in GroupKFold(5, seed).split(n, groups=groups)]
    assert max(sizes) / min(sizes) < 1.5


def test_stratified_group_keeps_class_balance_better_than_plain_group():
    rng = np.random.default_rng(3)
    groups = np.repeat(np.arange(100), 10)
    y = np.repeat(rng.random(100) < 0.2, 10).astype(int)       # label constant within group
    def spread(sp):
        return np.std([y[te].mean() for _, te in sp.split(1000, y=y, groups=groups)])
    assert spread(StratifiedGroupKFold(5, 3)) <= spread(GroupKFold(5, 3)) + 1e-9


@pytest.mark.parametrize("gap", [0, 7])
def test_forward_chaining_respects_time_and_gap(gap):
    n = 500
    rng = np.random.default_rng(1)
    time = rng.permutation(n).astype(float)                    # time order != row order
    folds = ForwardChaining(4, gap=gap, min_train_frac=0.3).split(n, time=time)
    assert len(folds) == 4
    for tr, te in folds:
        assert time[tr].max() < time[te].min()
        assert time[te].min() - time[tr].max() >= gap + 1 - 1e-9
    tests = np.concatenate([te for _, te in folds])
    assert len(np.unique(tests)) == len(tests)


def test_grouped_forward_chaining():
    n = 400
    groups = np.repeat(np.arange(40), 10)
    time = np.arange(n, dtype=float)
    for tr, te in ForwardChaining(4).split(n, groups=groups, time=time):
        assert len(np.intersect1d(groups[tr], groups[te])) == 0
        assert time[tr].max() < time[te].min()


def test_repeated_splitter_ids_and_independence():
    rs = RepeatedSplitter(KFold(3, seed=0), n_repeats=2)
    folds = rs.folds(30)
    assert [f[0] for f in folds] == ["r0f0", "r0f1", "r0f2", "r1f0", "r1f1", "r1f2"]
    assert not np.array_equal(folds[0][2], folds[3][2])
    assert [f[0] for f in RepeatedSplitter(ForwardChaining(3), 4).folds(60)] == ["r0f0", "r0f1", "r0f2"]
    again = RepeatedSplitter(KFold(3, seed=0), n_repeats=2).folds(30)
    assert all(np.array_equal(a[2], b[2]) for a, b in zip(folds, again))


def test_splitter_serialisation_and_errors():
    s = make_splitter("stratified_group", n_splits=4, seed=9)
    assert splitter_from_dict(splitter_to_dict(s)) == s
    with pytest.raises(ValueError):
        make_splitter("magic")
    with pytest.raises(ValueError):
        KFold(1).split(10)
    with pytest.raises(ValueError):
        GroupKFold(5).split(10, groups=np.arange(10) % 3)
    with pytest.raises(ValueError):
        StratifiedKFold(5).split(10)


def test_holdout_structures():
    n = 1000
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, n)
    dev, hold = carve_holdout(n, structure="iid", frac=0.15, seed=1, y=y)
    assert len(np.intersect1d(dev, hold)) == 0 and len(dev) + len(hold) == n
    assert abs(y[hold].mean() - y.mean()) < 0.03
    time = rng.permutation(n).astype(float)
    dev, hold = carve_holdout(n, structure="temporal", frac=0.2, seed=1, time=time)
    assert time[hold].min() > time[dev].max()
    groups = np.repeat(np.arange(100), 10)
    dev, hold = carve_holdout(n, structure="grouped", frac=0.15, seed=1, groups=groups)
    assert len(np.intersect1d(groups[dev], groups[hold])) == 0
    dev, hold = carve_holdout(n, structure="grouped_temporal", frac=0.15, seed=1, groups=groups,
                              time=np.arange(n, dtype=float))
    assert groups[hold].min() > groups[dev].max()
    with pytest.raises(ValueError):
        carve_holdout(n, structure="iid", frac=0.7, seed=0)


# ---------------------------------------------------------------- preprocessing

def _table(n=400, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    city = np.empty(n, dtype=object)
    city[:] = [f"c{v}" for v in rng.integers(0, 80, n)]
    color = np.empty(n, dtype=object)
    color[:] = list(rng.choice(["red", "green", "blue"], n))
    color[::17] = None
    x = rng.normal(size=n)
    x[::13] = np.nan
    when = np.array("2024-01-01", dtype="datetime64[D]") + rng.integers(0, 400, n)
    price = np.empty(n, dtype=object)
    price[:] = [f"${v:,.0f}" for v in rng.uniform(100, 9000, n)]
    t = to_table({"x": x, "city": city, "color": color, "when": when, "price": price,
                  "flag": np.array(["yes", "no"] * (n // 2), dtype=object), "id": np.arange(n)}, target=y)
    return t, y


def test_pipeline_fits_only_on_training_rows():
    t, y = _table()
    types = infer_types(t, np.random.default_rng(0))
    train, test = np.arange(0, 300), np.arange(300, 400)
    p = ColumnPipeline(types, task="binary", n_classes=2)
    Xtr = p.fit_transform(t, train, y[train])
    Xte = p.transform(t, test)
    assert np.array_equal(p.fitted_rows, train)
    assert Xtr.shape[1] == Xte.shape[1] == len(p.feature_names)
    assert np.isfinite(Xtr).all() and np.isfinite(Xte).all()
    assert "id" not in " ".join(p.feature_names)                          # identifiers dropped
    # Changing test rows must not change the fitted state (fit the same, transform differs only on changed rows)
    t2, _ = _table()
    t2.columns["x"][300:] = 1e9
    p2 = ColumnPipeline(types, task="binary", n_classes=2)
    Xtr2 = p2.fit_transform(t2, train, y[train])
    assert np.allclose(Xtr, Xtr2)


def test_numeric_scaling_imputation_and_indicator():
    t, y = _table()
    types = infer_types(t, np.random.default_rng(0))
    p = ColumnPipeline(types, task="binary", n_classes=2)
    X = p.fit_transform(t, np.arange(400), y)
    j = p.feature_names.index("x")
    assert abs(X[:, j].mean()) < 1e-9 and abs(X[:, j].std() - 1) < 1e-9
    assert "x__missing" in p.feature_names
    k = p.feature_names.index("x__missing")
    assert X[:, k].sum() == np.isnan(t.columns["x"]).sum()


def test_onehot_unseen_and_missing_levels():
    t, y = _table()
    types = infer_types(t, np.random.default_rng(0))
    p = ColumnPipeline(types, task="binary", n_classes=2)
    p.fit_transform(t, np.arange(400), y)
    cols = [i for i, nm in enumerate(p.feature_names) if nm.startswith("color=")]
    assert any(nm == "color=<missing>" for nm in p.feature_names)
    t.columns["color"][0] = "purple"                                    # unseen level
    row = p.transform(t, np.array([0]))[0, cols]
    assert row.sum() <= 1.0


def test_target_encoding_is_out_of_fold_for_training_rows():
    t, y = _table()
    types = infer_types(t, np.random.default_rng(0))
    assert types["city"].type == "categorical_high"
    rows = np.arange(400)
    p1 = ColumnPipeline(types, task="binary", n_classes=2, seed=5)
    X1 = p1.fit_transform(t, rows, y)
    y2 = y.copy()
    y2[0] = 1 - y2[0]                                                   # flip one row's own target
    p2 = ColumnPipeline(types, task="binary", n_classes=2, seed=5)
    X2 = p2.fit_transform(t, rows, y2)
    j = p1.feature_names.index("city__te")
    assert X1[0, j] == X2[0, j]                                         # own target never used for own encoding


def test_multiclass_target_encoding_width_and_ordinal_option():
    t, _ = _table()
    y = np.random.default_rng(1).integers(0, 3, 400)
    types = infer_types(t, np.random.default_rng(0))
    p = ColumnPipeline(types, task="multiclass", n_classes=3)
    p.fit_transform(t, np.arange(400), y)
    assert sum(nm.startswith("city__te") for nm in p.feature_names) == 3
    p = ColumnPipeline(types, PipelineOptions(categorical="ordinal", high_cardinality="ordinal", scale="none"),
                       task="multiclass", n_classes=3)
    X = p.fit_transform(t, np.arange(400), y)
    assert "city" in p.feature_names and "color" in p.feature_names and X.shape[1] < 15


def test_datetime_embedded_boolean_and_drop_option():
    t, y = _table()
    types = infer_types(t, np.random.default_rng(0))
    p = ColumnPipeline(types, PipelineOptions(drop=("x",)), task="binary", n_classes=2)
    X = p.fit_transform(t, np.arange(400), y)
    names = p.feature_names
    assert "x" not in names and any(nm.startswith("when__") for nm in names)
    assert "price" in names and "flag" in names
    raw = np.array([float(s.replace("$", "").replace(",", "")) for s in t.columns["price"]])
    assert np.corrcoef(X[:, names.index("price")], raw)[0, 1] > 0.9999
    flag = X[:, names.index("flag")]
    assert set(np.unique(flag)) <= {0.0, 1.0}


def test_transform_before_fit_and_bad_options():
    t, _ = _table()
    types = infer_types(t, np.random.default_rng(0))
    with pytest.raises(RuntimeError):
        ColumnPipeline(types).transform(t, np.arange(3))
    with pytest.raises(ValueError):
        PipelineOptions(scale="magic")


def test_log1p_target_roundtrip():
    y = np.array([0.0, 1.0, 10.0])
    assert np.allclose(expm1_target(log1p_target(y)), y)
    with pytest.raises(ValueError):
        log1p_target(np.array([-1.0]))
