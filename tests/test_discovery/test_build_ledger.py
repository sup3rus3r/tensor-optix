"""build() from accepted findings, the ledger bridge, meta-features and guardrails."""
import json
import os
import warnings

import numpy as np
import pytest

from tensor_optix.discovery import discover
from tensor_optix.discovery.build import PendingCriticalFindings
from tensor_optix.discovery.ledger_bridge import RESERVED_CONFIG_KEYS, _check
from tensor_optix.discovery.meta_features import META_FEATURES_VERSION
from tensor_optix.discovery.ml.families.base import sigmoid

import synth

FAST = ["linear", "naive_bayes", "knn"]


@pytest.fixture(scope="module")
def grouped_report():
    cols, y = synth.grouped(n_groups=150, per_group=10)
    return cols, y, discover(cols, y, families=FAST, seed=0)


@pytest.fixture(scope="module")
def leaky_report():
    cols, y = synth.leaky()
    labels = np.where(y == 1, "churned", "stayed").astype(object)
    return cols, labels, discover(cols, labels, families=FAST, seed=0)


def test_build_refuses_pending_critical(grouped_report):
    cols, y, rep = grouped_report
    fid = "structure.grouped:col=customer_id"
    assert rep.finding(fid).severity.value == "critical"
    with pytest.raises(PendingCriticalFindings, match="customer_id"):
        rep.build()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sol = rep.build(acknowledge_pending=True)
    assert sol.split.kind == "random" and fid in sol.evidence_mismatch


def test_build_applies_accepted_group_split_and_reproduces_cv(grouped_report):
    cols, y, rep = grouped_report
    rep2 = discover(cols, y, families=FAST, seed=0)
    rep2.accept("structure.grouped:col=customer_id")
    with warnings.catch_warnings():
        warnings.simplefilter("error")                     # nothing pending that experiments applied
        sol = rep2.build()
    assert sol.split.kind == "group" and sol.split.groups == "customer_id"
    assert sol.evidence_mismatch == []
    assert "customer_id" not in sol.column_types
    cv = sol.cross_validate(cols, y)
    best = next(x for x in rep2.experiments if x.stage == "trial" and x.family == sol.family
                and x.config == sol.config)
    assert abs(cv.mean - best.mean) < 0.03                 # same family / config / split kind
    sol.fit(cols, y)
    pred = sol.predict({k: v[:20] for k, v in cols.items()})
    assert set(np.unique(pred)) <= {0, 1}


def test_build_drops_accepted_leak_and_returns_original_labels(leaky_report):
    cols, labels, rep = leaky_report
    fid = "leakage.single_feature:col=refund_flag"
    rep.accept(fid)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        sol = rep.build()
    assert "refund_flag" in sol.drop and fid in sol.applied
    assert any("different findings" in str(x.message) for x in w) == bool(sol.evidence_mismatch)
    sol.fit(cols, labels)
    pred = sol.predict(cols)
    assert set(pred.tolist()) <= {"churned", "stayed"}
    acc = float(np.mean(pred == labels))
    assert 0.6 < acc < 0.95                                 # no leak: realistic accuracy
    assert sol.predict_proba(cols).shape == (len(labels), 2)
    with pytest.raises(KeyError, match="income"):
        sol.predict({"other": cols["other"]})


def test_rejected_leak_is_kept_and_flagged_as_mismatch(leaky_report):
    cols, labels, _ = leaky_report
    rep = discover(cols, labels, families=FAST, seed=0)
    fid = "leakage.single_feature:col=refund_flag"
    rep.reject(fid, note="known at prediction time")
    with pytest.warns(UserWarning, match="different findings"):
        sol = rep.build()
    assert "refund_flag" not in sol.drop and fid in sol.evidence_mismatch


def test_abstain_needs_explicit_family():
    r = np.random.default_rng(2)
    X = r.normal(size=(1000, 4))
    y = r.integers(0, 2, 1000)
    rep = discover({f"f{i}": X[:, i] for i in range(4)}, y, families=FAST, seed=0)
    assert rep.recommendation.verdict == "abstain"
    with pytest.raises(ValueError, match="abstained"):
        rep.build()
    sol = rep.build(family="linear")
    assert sol.family == "linear" and sol.config


def test_temporal_solution_uses_forward_chaining():
    cols, y = synth.drifting()
    rep = discover(cols, y, families=FAST, seed=0)
    fid = next(f.id for f in rep.findings if f.id.startswith("structure.temporal"))
    rep.accept(fid)
    sol = rep.build(family="linear")
    n_splits = rep.profile.sections["structure"]["splitter"]["n_splits"]
    assert n_splits > 5                                     # forward chaining gets more test blocks
    assert sol.split.kind == "forward" and sol.splitter().kind == "forward"
    cv = sol.cross_validate(cols, y)
    assert cv.status == "ok" and len(cv.scores) == n_splits


def test_decisions_roundtrip_through_discover(leaky_report):
    cols, labels, rep = leaky_report
    rep.reject("leakage.single_feature:col=refund_flag")
    rep2 = discover(cols, labels, families=FAST, seed=0, decisions=rep.decisions())
    assert "refund_flag" not in rep2.profile.sections["excluded_columns"]
    rep3 = discover(cols, labels, families=FAST, seed=0, decisions={"no.such:finding": "accept"})
    assert any("no.such:finding" in s for s in rep3.limitations)


# ---------------------------------------------------------------------------
# ledger
# ---------------------------------------------------------------------------

def _linear(n=800, seed=0):
    r = np.random.default_rng(seed)
    X = r.normal(size=(n, 4))
    y = (r.random(n) < sigmoid(X @ np.array([1.0, -0.7, 0.4, 0.0]))).astype(int)
    return {f"f{i}": X[:, i] for i in range(4)}, y


def test_ledger_records_parent_children_and_events(tmp_path):
    from tensor_optix.ledger import ExperimentStore
    from tensor_optix.selection.selector import record_matches_method
    cols, y = _linear()
    path = str(tmp_path / "ledger.db")
    rep = discover(cols, y, families=FAST, seed=0, store=path)
    refs = rep.ledger_refs
    st = ExperimentStore(path)
    try:
        parent = st.get_experiment(refs["experiment_id"])
        assert parent.algorithm == "discovery" and parent.config["role"] == "discovery"
        assert parent.task_env.startswith("discovery:")
        kids = [st.get_experiment(i) for i in refs["trials"].values()]
        assert kids and all(k.parent_experiment_id == parent.experiment_id for k in kids)
        assert all(k.config["role"] == "discovery_trial" for k in kids)
        types = [e.event_type for e in st.get_events(parent.experiment_id)]
        assert types[0] == "discovery.profile" and "discovery.recommendation" in types
        assert types.count("discovery.finding") == len(rep.findings)
        prof = st.get_events(parent.experiment_id)[0].payload
        assert prof["meta_features"]["version"] == META_FEATURES_VERSION
        for rec in [parent] + kids:
            for m in ("TPESampler", "RandomSampler", "linear", "discovery", "discovery_trial"):
                assert not record_matches_method(rec, m)
            assert not any(k in rec.config for k in RESERVED_CONFIG_KEYS)
            assert rec.best_primary_score is None
        assert st.query_experiments(task_env="CartPole-v1", limit=100) == []
    finally:
        st.close()


def test_reserved_keys_are_refused():
    for k in RESERVED_CONFIG_KEYS:
        with pytest.raises(ValueError):
            _check({"role": "discovery", k: "x"})


def test_store_with_other_history_does_not_change_the_recommendation(tmp_path):
    from tensor_optix.ledger import ExperimentStore
    path = str(tmp_path / "busy.db")
    st = ExperimentStore(path)
    for i in range(5):
        st.create_experiment(task_env="other-task", algorithm="ppo", framework="torch", seed=i,
                             checkpoint_dir="", config={"sampler": "TPESampler", "lr": 1e-3})
    st.close()
    cols, y = _linear(seed=1)
    a = discover(cols, y, families=FAST, seed=4)
    b = discover(cols, y, families=FAST, seed=4, store=path)

    def core(r):
        d = r.to_dict()
        return json.dumps({k: d[k] for k in ("recommendation", "findings", "comparisons", "plan")}, sort_keys=True)

    assert core(a) == core(b)


# ---------------------------------------------------------------------------
# guardrails
# ---------------------------------------------------------------------------

def test_script_column_name_renders_inert():
    r = np.random.default_rng(3)
    n = 600
    evil = "<script>alert('x')</script>"
    x = r.normal(size=n)
    y = (r.random(n) < sigmoid(2 * x)).astype(int)
    leak = np.where(r.random(n) < 0.99, y, 1 - y).astype(float)
    rep = discover({"x": x, evil: leak}, y, families=["linear"], seed=0)
    html = rep.to_html()
    assert evil not in html
    assert "&lt;script&gt;" in html
    assert rep.finding(f"leakage.single_feature:col={evil}") is not None


def test_meta_features_in_profile_and_unused_for_decisions(monkeypatch):
    from tensor_optix.discovery.ml import pipeline
    cols, y = _linear(seed=2)
    a = discover(cols, y, families=FAST, seed=0)
    mf = a.profile.sections["meta_features"]
    assert mf["version"] == META_FEATURES_VERSION and mf["n_features"] == 4 and mf["task"] == "binary"
    monkeypatch.setattr(pipeline, "ml_meta_features", lambda s: {"version": 999, "poison": True})
    b = discover(cols, y, families=FAST, seed=0)
    assert a.recommendation.to_dict() == b.recommendation.to_dict()
    assert [f.id for f in a.findings] == [f.id for f in b.findings]


def test_discover_writes_nothing_without_store(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cols, y = _linear(seed=3)
    discover(cols, y, families=FAST, seed=0)
    assert os.listdir(tmp_path) == []
