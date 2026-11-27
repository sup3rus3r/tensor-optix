"""End-to-end discover(): known-best families, decisions, reproducibility, guardrails."""
import json

import numpy as np
import pytest

from tensor_optix.discovery import Budget, DiscoveryReport, discover
from tensor_optix.discovery.findings import Severity
from tensor_optix.discovery.ml.families.base import sigmoid

import synth

FAST = ["linear", "naive_bayes", "knn"]


def _linear_data(n=1500, seed=0):
    r = np.random.default_rng(seed)
    X = r.normal(size=(n, 5))
    y = (r.random(n) < sigmoid(X @ np.array([1.2, -0.9, 0.6, 0.0, 0.0]))).astype(int)
    return {f"f{i}": X[:, i] for i in range(5)}, y


def _strip_timing(d):
    d = json.loads(json.dumps(d))
    d.pop("created_at", None)
    d.pop("budget_used", None)
    for x in d.get("experiments", []):
        x.pop("fit_time_s", None)
    return d


def test_linear_truth_recommends_linear():
    cols, y = _linear_data()
    rep = discover(cols, y, families=FAST, seed=0)
    rec = rep.recommendation
    assert rec.family == "linear"
    assert rec.verdict != "abstain"          # NB is often inseparable from linear on such data
    assert rec.expected_score is not None and rec.expected_interval is not None
    lo, hi = rec.expected_interval
    assert lo <= rec.expected_score <= hi
    assert rep.profile.sections["structure"]["evaluation_split"] == "iid"
    assert rep.profile.sections["structure"]["holdout_rows"] > 0


def test_xor_does_not_recommend_linear():
    r = np.random.default_rng(1)
    X = r.normal(size=(1500, 4))
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)
    rep = discover({f"f{i}": X[:, i] for i in range(4)}, y, families=FAST, seed=0)
    assert rep.recommendation.family == "knn"
    assert rep.recommendation.verdict == "clear_winner"
    assert any(f.id == "family.nonlinear_signal" for f in rep.findings)


def test_pure_noise_abstains():
    r = np.random.default_rng(2)
    X = r.normal(size=(1200, 5))
    y = r.integers(0, 2, 1200)
    rep = discover({f"f{i}": X[:, i] for i in range(5)}, y, families=FAST, seed=0)
    assert rep.recommendation.verdict == "abstain" and rep.recommendation.family is None
    assert any(f.id == "family.abstain" for f in rep.findings)
    assert any(s.kind == "data" for s in rep.plan)


def test_regression_end_to_end():
    r = np.random.default_rng(3)
    X = r.normal(size=(1200, 4))
    y = 2 * X[:, 0] - X[:, 1] + r.normal(scale=0.5, size=1200)
    rep = discover({f"f{i}": X[:, i] for i in range(4)}, y, families=FAST, seed=0)
    assert rep.input_summary["metric"] == "rmse"
    assert rep.recommendation.family == "linear"
    assert rep.recommendation.expected_score < np.std(y)


def test_leak_excluded_and_rejection_respected():
    cols, y = synth.leaky()
    rep = discover(cols, y, families=FAST, seed=0)
    fid = "leakage.single_feature:col=refund_flag"
    assert rep.finding(fid).severity == Severity.WARNING
    assert "refund_flag" in rep.profile.sections["excluded_columns"]
    assert any("refund_flag" in c for c in rep.recommendation.conditions)
    assert rep.recommendation.expected_score < 0.95
    rep2 = discover(cols, y, families=FAST, seed=0, decisions={fid: "reject"})
    assert "refund_flag" not in rep2.profile.sections["excluded_columns"]
    assert rep2.recommendation.expected_score > 0.97
    assert rep2.decision(fid).state.value == "rejected"


def test_grouped_data_uses_group_split_and_holdout():
    cols, y = synth.grouped(n_groups=150, per_group=10)
    rep = discover(cols, y, families=FAST, seed=0)
    s = rep.profile.sections["structure"]
    assert s["evaluation_split"] == "grouped" and s["splitter"]["kind"] in ("stratified_group", "group")
    assert any(f.id == "structure.grouped:col=customer_id" for f in rep.findings)
    rep2 = discover(cols, y, families=FAST, seed=0, decisions={"structure.grouped:col=customer_id": "reject"})
    assert rep2.profile.sections["structure"]["evaluation_split"] == "iid"


def test_temporal_data_uses_forward_chaining():
    cols, y = synth.drifting()
    rep = discover(cols, y, families=FAST, seed=0)
    s = rep.profile.sections["structure"]
    assert s["evaluation_split"] == "temporal" and s["splitter"]["kind"] == "forward"
    assert "temporal" in rep.plan[0].description.lower() or "forward" in rep.plan[0].description.lower()


def test_same_seed_same_report_and_roundtrip():
    cols, y = _linear_data(n=900, seed=5)
    a = discover(cols, y, families=FAST, seed=3)
    b = discover(cols, y, families=FAST, seed=3)
    assert _strip_timing(a.to_dict()) == _strip_timing(b.to_dict())
    assert DiscoveryReport.from_json(a.to_json()).to_dict() == a.to_dict()
    assert DiscoveryReport.from_html(a.to_html()).to_dict() == a.to_dict()


def test_no_global_rng_side_effects():
    torch = pytest.importorskip("torch")
    cols, y = _linear_data(n=600, seed=6)
    np.random.seed(123)
    torch.manual_seed(123)
    before_np = np.random.get_state()[1].copy()
    before_t = torch.random.get_rng_state().clone()
    discover(cols, y, families=FAST + (["mlp"] if _has("mlp") else []), seed=0, budget=Budget(time_s=60))
    assert np.array_equal(np.random.get_state()[1], before_np)
    assert torch.equal(torch.random.get_rng_state(), before_t)


def _has(name):
    from tensor_optix.discovery.ml.families.base import all_families
    return any(f.name == name for f in all_families())


def test_tiny_budget_records_cuts_instead_of_overrunning():
    cols, y = _linear_data(n=1500, seed=7)
    rep = discover(cols, y, families=FAST, seed=0, budget=Budget(time_s=60, max_fits=40))
    assert rep.budget_used["fits"] <= 40 + 10          # one experiment may finish its current fold set
    assert rep.budget_used["cuts"]
    assert any(f.category.value == "budget" for f in rep.findings)


def test_missing_boosting_is_a_limitation(monkeypatch):
    from tensor_optix.discovery.ml import pipeline
    from tensor_optix.discovery.ml.families import base
    real = base.all_families
    monkeypatch.setattr(pipeline, "all_families",
                        lambda: [f for f in real() if f.name not in ("lightgbm", "xgboost", "mlp")])
    from tensor_optix.discovery.ml import trials
    monkeypatch.setattr(trials, "all_families",
                        lambda: [f for f in real() if f.name not in ("lightgbm", "xgboost", "mlp")])
    r = np.random.default_rng(8)
    X = r.normal(size=(1000, 4))
    y = ((X[:, 0] > 0) & (X[:, 1] > 0.3)).astype(int)
    rep = discover({f"f{i}": X[:, i] for i in range(4)}, y, seed=0)
    f = rep.finding("limitation.no_boosting")
    assert f.severity == Severity.ADVISORY
    assert all(x.family not in ("lightgbm", "xgboost", "mlp") for x in rep.experiments)


def test_rl_input_is_refused_clearly():
    class Env:
        action_space = None

        def reset(self):
            pass

        def step(self, a):
            pass

    with pytest.raises(NotImplementedError, match="RL discovery"):
        discover(Env())


def test_target_required():
    with pytest.raises(ValueError, match="needs a target"):
        discover({"a": np.arange(50.0)})


@pytest.mark.slow
def test_full_portfolio_on_nonlinear_regression():
    r = np.random.default_rng(9)
    X = r.uniform(-2, 2, size=(2000, 4))
    y = np.sin(2 * X[:, 0]) * 2 + X[:, 1] * X[:, 2] + r.normal(scale=0.2, size=2000)
    rep = discover({f"f{i}": X[:, i] for i in range(4)}, y, seed=0, budget=Budget(time_s=300))
    assert rep.recommendation.family in ("lightgbm", "xgboost", "mlp", "knn")
    assert rep.recommendation.family != "linear"
