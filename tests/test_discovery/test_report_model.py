"""
Discovery report data model: findings, actions, decisions, JSON / HTML
round-trips, schema versioning, HTML safety, budget and seeding.
"""
import json

import numpy as np
import pytest

from tensor_optix.discovery import (
    Action, Budget, Category, Comparison, Decision, DecisionState, DiscoveryReport, Evidence,
    ExperimentResult, Finding, PlanStep, Profile, Recommendation, Severity,
    DISCOVERY_SCHEMA_VERSION,
)
from tensor_optix.discovery.budget import BudgetLedger, predict_cost
from tensor_optix.discovery.findings import confidence_from_p, jsonable
from tensor_optix.discovery.report import extract_json
from tensor_optix.discovery.rng import int_seed_for, rng_for


def _finding(fid="leakage.single_feature:col=x", sev=Severity.WARNING, cat=Category.LEAKAGE, **kw):
    return Finding(
        id=fid, category=cat, severity=sev, statement=kw.pop("statement", "x predicts the target too well"),
        evidence=kw.pop("evidence", (Evidence(test="single_feature_auc", statistic=0.99, n=500,
                                              null_value=0.5, p_value=0.001, interval=(0.98, 1.0),
                                              details={"next_best": 0.71}, seed=3),)),
        confidence=kw.pop("confidence", "high"),
        suggested_action=kw.pop("action", Action("drop_column", {"column": "x"})),
        **kw,
    )


def _report(**kw):
    exps = [
        ExperimentResult(family="linear", config={"l2": 1.0}, metric="auc", fold_ids=("r0f0", "r0f1", "r0f2"),
                         scores=(0.81, 0.79, 0.80), interval=(0.78, 0.82), mean=0.80, fit_time_s=0.1),
        ExperimentResult(family="mlp", config={"hidden": [64]}, metric="auc", fold_ids=("r0f0", "r0f1", "r0f2"),
                         scores=(0.84, 0.83, None), interval=(0.81, 0.86), mean=0.835,
                         stop_reasons=("converged", "converged"), fit_time_s=2.5),
        ExperimentResult(family="lightgbm", metric="auc", status="unavailable_dependency",
                         error="lightgbm not installed"),
    ]
    findings = kw.pop("findings", [
        _finding(),
        _finding("structure.grouped:col=customer_id", Severity.CRITICAL, Category.STRUCTURE,
                 statement="rows repeat per customer; a random split leaks",
                 action=Action("use_split", {"kind": "group", "groups": "customer_id"}), default_applied=True),
        _finding("data_types.identifier:col=row_id", Severity.INFO, Category.DATA_TYPES,
                 statement="row_id looks like an identifier", action=Action("drop_column", {"column": "row_id"})),
    ])
    return DiscoveryReport(
        domain="ml",
        profile=Profile(domain="ml", n_total=10_000, n_sampled=2_000,
                        sampling={"method": "stratified", "seed": 0},
                        sections={"columns": [{"name": "x", "type": "numeric_continuous", "missing": 0.0},
                                              {"name": "<script>alert(1)</script>", "type": "text"}],
                                  "target": {"task": "binary", "positive_rate": 0.2}},
                        fingerprint="abc123"),
        findings=findings, experiments=exps,
        comparisons=[Comparison(a=exps[1].key, b=exps[0].key, n_pairs=2, delta_mean=0.035,
                                delta_interval=(0.01, 0.06), rope=0.005, verdict="better",
                                p_value=0.01, p_holm=0.02, method="corrected_t")],
        recommendation=Recommendation(family="mlp", config={"hidden": [64]}, verdict="clear_winner",
                                      rationale=("beats linear by 0.035 AUC [0.01, 0.06]",),
                                      expected_score=0.83, expected_interval=(0.80, 0.86), metric="auc",
                                      conditions=("valid only if column x is dropped",)),
        plan=[PlanStep(kind="split", description="use GroupKFold on customer_id",
                       finding_refs=("structure.grouped:col=customer_id",), code="splitter = ...")],
        limitations=["tree ensembles were not evaluated (lightgbm not installed)"],
        budget_requested=Budget(time_s=60).to_dict(), budget_used={"elapsed_s": 12.3},
        seed=7, **kw,
    )


# ---------------------------------------------------------------- findings

def test_action_grammar_is_closed():
    with pytest.raises(ValueError, match="unknown action kind"):
        Action("delete_everything", {})
    with pytest.raises(ValueError, match="missing parameter"):
        Action("drop_column", {})
    with pytest.raises(ValueError, match="newer"):
        Action("drop_column", {"column": "x"}, grammar_version=99)


def test_finding_validation_and_roundtrip():
    f = _finding()
    assert Finding.from_dict(json.loads(json.dumps(f.to_dict()))) == f
    with pytest.raises(ValueError):
        _finding(fid="")
    with pytest.raises(ValueError):
        _finding(confidence="certain")
    assert Severity.CRITICAL.rank > Severity.WARNING.rank > Severity.ADVISORY.rank > Severity.INFO.rank


def test_jsonable_handles_numpy_and_non_finite():
    d = jsonable({"a": np.float32(1.5), "b": np.int64(3), "c": np.array([1, 2]), "d": float("nan"),
                  "e": (1, float("inf")), "f": Severity.INFO})
    assert d == {"a": 1.5, "b": 3, "c": [1, 2], "d": None, "e": [1, None], "f": "info"}
    json.dumps(d, allow_nan=False)


def test_confidence_rule():
    assert confidence_from_p(0.0001) == "high"
    assert confidence_from_p(0.005) == "moderate"
    assert confidence_from_p(0.2) == "low"
    assert confidence_from_p(0.0001, effect_ok=False) == "low"
    assert confidence_from_p(None) == "low"


# ---------------------------------------------------------------- decisions

def test_decisions_start_pending_and_are_per_finding():
    r = _report()
    assert all(r.decision(f.id).state == DecisionState.PENDING for f in r.findings)
    r.reject("leakage.single_feature:col=x", note="known at prediction time")
    r.accept("structure.grouped:col=customer_id")
    assert r.decision("leakage.single_feature:col=x").state == DecisionState.REJECTED
    assert r.decision("leakage.single_feature:col=x").note == "known at prediction time"
    assert r.decision("structure.grouped:col=customer_id").state == DecisionState.ACCEPTED
    assert [f.id for f in r.pending()] == ["data_types.identifier:col=row_id"]
    with pytest.raises(KeyError):
        r.accept("no.such.finding")


def test_accept_where_requires_a_filter_and_skips_decided():
    r = _report()
    with pytest.raises(ValueError):
        r.accept_where()
    r.reject("structure.grouped:col=customer_id")
    done = r.accept_where(min_severity="warning")
    assert done == ["leakage.single_feature:col=x"]                 # the CRITICAL one was already rejected
    assert r.decision("structure.grouped:col=customer_id").state == DecisionState.REJECTED


def test_duplicate_ids_and_stale_decisions():
    with pytest.raises(ValueError, match="duplicate"):
        _report(findings=[_finding(), _finding()])
    r = _report()
    stale = r.apply_decisions({"leakage.single_feature:col=x": Decision(DecisionState.ACCEPTED),
                               "gone:col=y": Decision(DecisionState.REJECTED)})
    assert stale == ["gone:col=y"]
    assert r.decision("leakage.single_feature:col=x").state == DecisionState.ACCEPTED


# ---------------------------------------------------------------- serialisation

def test_json_roundtrip_is_exact():
    r = _report()
    r.reject("leakage.single_feature:col=x", note="fine")
    d = r.to_dict()
    r2 = DiscoveryReport.from_json(r.to_json())
    assert r2.to_dict() == d
    json.dumps(d, allow_nan=False)


def test_html_roundtrip_is_exact_and_self_contained(tmp_path):
    r = _report()
    r.accept("structure.grouped:col=customer_id")
    path = tmp_path / "report.html"
    html = r.to_html(str(path))
    assert DiscoveryReport.from_html(str(path)).to_dict() == r.to_dict()
    assert DiscoveryReport.from_html(html).to_dict() == r.to_dict()
    low = html.lower()
    assert "<script src" not in low and "http://" not in low and "https://" not in low
    assert "<style>" in low


def test_html_escapes_untrusted_strings():
    html = _report().to_html()
    body = html.split('<script type="application/json"')[0]
    assert "<script>alert(1)</script>" not in body
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in body
    # the embedded JSON cannot be terminated early by a "</script>" inside the data
    data_block = html.split('<script type="application/json" id="tensor-optix-discovery">')[1]
    assert data_block.count("</script>") == 1
    assert extract_json(html)["profile"]["sections"]["columns"][1]["name"] == "<script>alert(1)</script>"


def test_html_contains_every_section_and_finding():
    r = _report()
    html = r.to_html()
    for anchor in ("summary", "understanding", "risks", "families", "recommendation", "plan", "limits"):
        assert f'id="{anchor}"' in html
    for f in r.findings:
        assert f.id in html
    assert "Clear winner" in html and "<svg" in html


def test_text_summary_lists_findings():
    s = _report().summary()
    assert "mlp" in s and "CRITICAL" in s and "structure.grouped:col=customer_id" in s


def test_schema_versioning():
    d = _report().to_dict()
    newer = dict(d, schema_version=DISCOVERY_SCHEMA_VERSION + 1)
    with pytest.raises(ValueError, match="upgrade tensor-optix"):
        DiscoveryReport.from_dict(newer)
    with pytest.raises(ValueError, match="not a discovery report"):
        DiscoveryReport.from_dict({k: v for k, v in d.items() if k != "schema_version"})


def test_experiment_and_comparison_validation():
    with pytest.raises(ValueError):
        ExperimentResult(family="x", status="weird")
    with pytest.raises(ValueError):
        ExperimentResult(family="x", fold_ids=("a",), scores=())
    with pytest.raises(ValueError):
        Comparison(a="a", b="b", n_pairs=1, delta_mean=0.0, delta_interval=None, rope=0.0, verdict="maybe")
    with pytest.raises(ValueError):
        Recommendation(family=None, verdict="certain")


def test_render_handles_an_empty_report():
    r = DiscoveryReport(domain="ml", profile=Profile(domain="ml"))
    html = r.to_html()
    assert "No experiments were run" in html and DiscoveryReport.from_html(html).to_dict() == r.to_dict()


def test_build_without_a_registered_builder_says_so():
    with pytest.raises(NotImplementedError):
        DiscoveryReport(domain="other", profile=Profile(domain="other")).build()


def test_rl_build_without_a_recommendation_asks_for_a_family():
    with pytest.raises(ValueError, match="family"):
        DiscoveryReport(domain="rl", profile=Profile(domain="rl")).build()


# ---------------------------------------------------------------- budget and rng

def test_budget_validation_and_defaults():
    with pytest.raises(ValueError):
        Budget(time_s=0)
    with pytest.raises(ValueError):
        Budget(max_rows=-5)
    assert Budget.default_for_rows(10_000).time_s == pytest.approx(60.0)
    sizes = [100, 10_000, 10 ** 5, 10 ** 6, 10 ** 9, 10 ** 13]
    times = [Budget.default_for_rows(n).time_s for n in sizes]
    assert times == sorted(times)                                  # grows with size
    assert times[0] >= 30.0 and times[-1] == 900.0                 # floor and 15-minute cap


def test_budget_ledger():
    led = BudgetLedger(Budget(time_s=100, max_fits=2))
    assert not led.exhausted()
    led.record_fit(500)
    led.record_fit(500)
    assert led.exhausted() and led.fits_left() == 0 and led.rows_fitted == 1000
    led.record_cut("trials", "repeats 4 -> 2", "time")
    assert led.to_dict()["cuts"][0]["what"] == "repeats 4 -> 2"


def test_predict_cost_clamps_exponent():
    assert predict_cost(1.0, 1000, 2000, exponent=0.2) == pytest.approx(2.0)   # clamped to 1
    assert predict_cost(1.0, 1000, 2000, exponent=5.0) == pytest.approx(4.0)   # clamped to 2


def test_rng_streams_are_reproducible_and_independent():
    a1 = rng_for(7, "folds", 0).integers(0, 10 ** 9, size=5)
    a2 = rng_for(7, "folds", 0).integers(0, 10 ** 9, size=5)
    b = rng_for(7, "folds", 1).integers(0, 10 ** 9, size=5)
    c = rng_for(8, "folds", 0).integers(0, 10 ** 9, size=5)
    assert list(a1) == list(a2)
    assert list(a1) != list(b) and list(a1) != list(c)
    assert int_seed_for(7, "torch") == int_seed_for(7, "torch") < 2 ** 31
    with pytest.raises(ValueError):
        rng_for(-1, "x")
    before = np.random.get_state()[1][:5].copy()
    rng_for(1, "x").random(10)
    assert list(np.random.get_state()[1][:5]) == list(before)       # global RNG untouched
