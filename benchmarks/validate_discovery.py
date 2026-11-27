#!/usr/bin/env python3
"""
Validation of tensor_optix.discovery on an offline suite of realistic synthetic
datasets, each with a known planted issue and a fresh, structure-correct test
set drawn from the same generator (new customers / new patients / the future).

For every dataset:
  * discover() runs with the default budget on the training part;
  * the planted issue must be reported, at (at least) the expected severity;
  * the findings discovery applied are accepted, the recommendation is built,
    fit on the training part and scored on the fresh test set;
  * every other candidate family's best configuration is built and scored the
    same way (the oracle): regret = best family's test score - recommended
    family's test score, oriented so positive = the recommendation is worse;
  * the report must render in < 1 s and be < 2 MB.

Acceptance (printed at the end; exit code 1 if any fails):
  * all planted issues detected at the expected severity;
  * regret <= the metric's practical threshold (ROPE) on >= 80% of datasets;
  * the recommendation is never worse than the dummy baseline on the test set;
  * wall time within 110% of the budget.

Usage:
    python benchmarks/validate_discovery.py            # full suite (~10-15 min)
    python benchmarks/validate_discovery.py --quick    # numpy families only
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np  # noqa: E402

from tensor_optix.discovery import discover  # noqa: E402
from tensor_optix.discovery.findings import Severity  # noqa: E402
from tensor_optix.discovery.ml.families.base import CLASSIFICATION, sigmoid  # noqa: E402
from tensor_optix.discovery.stats.metrics import get_metric, score  # noqa: E402


def _obj(values):
    a = np.empty(len(values), dtype=object)
    a[:] = list(values)
    return a


def _take(cols, idx):
    return {k: v[idx] for k, v in cols.items()}


# ---------------------------------------------------------------------------
# datasets: each returns (train_cols, train_y, test_cols, test_y, expected)
# expected = list of (finding id, minimum severity[, "full"]); "full" = only checked when
# the full family portfolio runs (not --quick)
# ---------------------------------------------------------------------------

def churn_with_leak(seed=0):
    r = np.random.default_rng(seed)
    n = 6000
    tenure = r.gamma(2.0, 12.0, n)
    income = r.normal(size=n)
    plan = _obj(r.choice(["basic", "plus", "pro"], n, p=[0.6, 0.3, 0.1]))
    support = r.poisson(1.5, n)
    z = -1.0 - 0.04 * tenure + 0.9 * income + 0.4 * support + np.where(plan == "basic", 0.5, 0.0)
    y = (r.random(n) < sigmoid(z)).astype(int)
    refund = np.where(r.random(n) < 0.97, y, 1 - y).astype(float)          # filled in after churn
    cols = {"tenure_months": tenure, "income": income, "plan": plan, "support_calls": support,
            "refund_issued": refund, "signup_id": np.arange(n)}
    tr, te = np.arange(4000), np.arange(4000, n)
    return (_take(cols, tr), y[tr], _take(cols, te), y[te],
            [("leakage.single_feature:col=refund_issued", Severity.WARNING)])


def grouped_patients(seed=0):
    r = np.random.default_rng(seed)
    n_pat, visits = 500, 8
    g = np.repeat(np.arange(n_pat), visits)
    frailty = r.normal(scale=1.5, size=n_pat)[g]
    age = (r.normal(60, 12, n_pat)[g]).round()
    bp = r.normal(130, 15, len(g))
    dose = r.uniform(0, 10, len(g))
    z = -0.5 + 0.03 * (age - 60) + 0.02 * (bp - 130) - 0.15 * dose + frailty
    y = (r.random(len(g)) < sigmoid(z)).astype(int)
    cols = {"patient_id": _obj([f"P{v:05d}" for v in g]), "age": age, "systolic_bp": bp, "dose_mg": dose}
    tr, te = np.flatnonzero(g < 350), np.flatnonzero(g >= 350)            # new patients at test time
    return (_take(cols, tr), y[tr], _take(cols, te), y[te],
            [("structure.grouped:col=patient_id", Severity.CRITICAL)])


def temporal_sales(seed=0):
    r = np.random.default_rng(seed)
    n = 3000
    day = np.arange(n)
    level = 50 + np.cumsum(r.normal(scale=0.6, size=n))
    weekly = 6.0 * np.sin(2 * np.pi * day / 7)
    promo = (r.random(n) < 0.2).astype(float)
    price = r.uniform(8, 12, n)
    y = level + weekly + 9.0 * promo - 2.0 * (price - 10) + r.normal(scale=2.0, size=n)
    date = np.datetime64("2018-01-01") + day.astype("timedelta64[D]")
    cols = {"date": date, "promo": promo, "price": price}
    tr, te = np.arange(2500), np.arange(2500, n)                          # the future
    perm = r.permutation(len(tr))                                          # rows arrive unordered
    return (_take(cols, tr[perm]), y[tr][perm], _take(cols, te), y[te],
            [("structure.temporal:col=date", Severity.CRITICAL)])


def high_cardinality(seed=0):
    r = np.random.default_rng(seed)
    n, levels = 8000, 600
    effect = r.normal(scale=1.0, size=levels)
    store = r.integers(0, levels, n)
    x = r.normal(size=n)
    y = 2.0 * effect[store] + x + r.normal(scale=1.0, size=n)
    cols = {"store_code": _obj([f"S{v:04d}" for v in store]), "x": x}
    tr, te = np.arange(6000), np.arange(6000, n)
    return _take(cols, tr), y[tr], _take(cols, te), y[te], []


def imbalanced_fraud(seed=0):
    r = np.random.default_rng(seed)
    n = 12000
    amount = r.lognormal(3.5, 1.0, n)
    hour = r.integers(0, 24, n).astype(float)
    foreign = (r.random(n) < 0.1).astype(float)
    velocity = r.poisson(2.0, n).astype(float)
    z = -6.2 + 0.6 * np.log(amount) + 1.8 * foreign + 0.35 * velocity + 0.8 * ((hour < 5) | (hour > 22))
    y = (r.random(n) < sigmoid(z)).astype(int)
    cols = {"amount": amount, "hour": hour, "foreign": foreign, "velocity": velocity}
    tr, te = np.arange(8000), np.arange(8000, n)
    return (_take(cols, tr), y[tr], _take(cols, te), y[te],
            [("target.imbalance", Severity.ADVISORY)])


def friedman1(seed=0):
    r = np.random.default_rng(seed)
    n = 5000
    X = r.uniform(size=(n, 10))
    y = (10 * np.sin(np.pi * X[:, 0] * X[:, 1]) + 20 * (X[:, 2] - 0.5) ** 2 + 10 * X[:, 3] + 5 * X[:, 4]
         + r.normal(size=n))
    cols = {f"x{i}": X[:, i] for i in range(10)}
    tr, te = np.arange(3500), np.arange(3500, n)
    return (_take(cols, tr), y[tr], _take(cols, te), y[te],
            [("family.nonlinear_signal", Severity.INFO, "full")])        # needs a strong non-linear family


DATASETS = [churn_with_leak, grouped_patients, temporal_sales, high_cardinality, imbalanced_fraud, friedman1]


# ---------------------------------------------------------------------------

def _test_score(sol, test_cols, test_y, metric):
    if sol.task in CLASSIFICATION:
        P = sol.predict_proba(test_cols)
        idx = {c: i for i, c in enumerate(sol.classes_)}
        yt = np.array([idx[v] for v in test_y.tolist()])
        return score(metric, yt, proba=P, n_classes=len(sol.classes_))
    return score(metric, np.asarray(test_y, dtype=np.float64), values=sol.predict(test_cols), n_classes=0)


def run_one(gen, families, seed):
    tr_cols, tr_y, te_cols, te_y, expected = gen(seed)
    expected = [e[:2] for e in expected if not (families is not None and len(e) > 2 and e[2] == "full")]
    t0 = time.perf_counter()
    rep = discover(tr_cols, tr_y, families=families, seed=seed)
    wall = time.perf_counter() - t0
    t1 = time.perf_counter()
    html = rep.to_html()
    render_s = time.perf_counter() - t1
    sev = {f.id: f.severity for f in rep.findings}
    detected = [(fid, fid in sev and sev[fid].rank >= s.rank) for fid, s in expected]
    for fid in rep.profile.sections.get("applied_findings") or []:
        rep.accept(fid)
    metric = rep.input_summary["metric"]
    hib = get_metric(metric).higher_is_better
    rope = rep.profile.sections["landmarks"]["rope"]
    best_cfg = {}
    for x in rep.experiments:
        if x.stage == "trial" and x.mean is not None:
            cur = best_cfg.get(x.family)
            if cur is None or (x.mean > cur.mean if hib else x.mean < cur.mean):
                best_cfg[x.family] = x
    scores = {}
    for fam, x in sorted(best_cfg.items()):
        sol = rep.build(family=fam, config=x.config, acknowledge_pending=True)
        sol.fit(tr_cols, tr_y)
        scores[fam] = _test_score(sol, te_cols, te_y, metric)
    dsol = rep.build(family="baseline", config={}, acknowledge_pending=True).fit(tr_cols, tr_y)
    dummy = _test_score(dsol, te_cols, te_y, metric)
    rec = rep.recommendation
    rec_score = scores.get(rec.family) if rec.family else dummy
    best_fam = max(scores, key=lambda f: scores[f] if hib else -scores[f]) if scores else None
    best = scores[best_fam] if best_fam else dummy
    regret = (best - rec_score) if hib else (rec_score - best)
    beats_dummy = (rec_score >= dummy) if hib else (rec_score <= dummy)
    return {
        "dataset": gen.__name__, "rows": len(tr_y), "metric": metric, "rope": rope,
        "verdict": rec.verdict, "recommended": rec.family, "recommended_test": rec_score,
        "holdout_estimate": rec.expected_score, "best_family": best_fam, "best_test": best,
        "dummy_test": dummy, "regret": regret, "regret_ok": regret <= rope or rec.family == best_fam,
        "not_worse_than_dummy": bool(beats_dummy), "detected": detected,
        "all_detected": all(ok for _, ok in detected), "budget_s": rep.budget_requested["time_s"],
        "wall_s": wall, "within_budget": wall <= 1.1 * rep.budget_requested["time_s"],
        "render_s": render_s, "html_mb": len(html.encode("utf-8")) / 1e6,
        "family_test_scores": scores,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--quick", action="store_true", help="numpy families only (linear, naive_bayes, knn)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", type=str, default=None, help="write results to this file")
    args = ap.parse_args(argv)
    families = ["linear", "naive_bayes", "knn"] if args.quick else None
    rows = []
    for gen in DATASETS:
        print(f"== {gen.__name__}", flush=True)
        res = run_one(gen, families, args.seed)
        rows.append(res)
        print(f"   {res['metric']}: recommended {res['recommended']} ({res['verdict']}) test "
              f"{res['recommended_test']:.4f}; best {res['best_family']} {res['best_test']:.4f}; dummy "
              f"{res['dummy_test']:.4f}; regret {res['regret']:+.4f} (rope {res['rope']:.4f}); "
              f"detected {res['detected']}; {res['wall_s']:.0f}s of {res['budget_s']:.0f}s; "
              f"render {res['render_s'] * 1000:.0f} ms, {res['html_mb']:.2f} MB", flush=True)
    checks = {
        "planted issues detected": all(r["all_detected"] for r in rows),
        "regret <= ROPE on >= 80% of datasets": np.mean([r["regret_ok"] for r in rows]) >= 0.8,
        "never worse than dummy": all(r["not_worse_than_dummy"] for r in rows),
        "within 110% of budget": all(r["within_budget"] for r in rows),
        "report < 1 s and < 2 MB": all(r["render_s"] < 1.0 and r["html_mb"] < 2.0 for r in rows),
    }
    print("\nAcceptance:")
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    if args.json:
        Path(args.json).write_text(json.dumps({"results": rows, "checks": checks}, indent=2, default=str))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
