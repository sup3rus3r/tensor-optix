"""
The ML discovery pipeline: dataset in, ``DiscoveryReport`` out.

Order (each stage only uses what the earlier ones established):

  0. normalise input, profile the target, rows without a target set aside
  1. experiment sample (budget / ``max_rows``), feature types, data quality
  2. structure (grouped / temporal) -> the evaluation split
  3. confirmation holdout carved with that split, *before* any model selection
  4. leakage on the development rows (suspect columns are excluded from every
     later experiment unless the user rejected the finding)
  5. landmarks, distribution shift
  6. candidate-family trials, refinement of the leaders
  7. recommendation; the chosen family refit on all development rows and
     scored once on the holdout
  8. plan

Findings whose ``default_applied`` is true shape discovery's own experiments
unless ``decisions`` rejects them; findings with ``default_applied=False`` only
apply when ``decisions`` accepts them. Re-run with ``decisions=report.decisions()``
after reviewing a report to get experiments consistent with those choices.
"""
from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np

from ..budget import Budget, BudgetLedger
from ..findings import Category, Decision, DecisionState, Evidence, Finding, Severity, jsonable
from ..meta_features import ml_meta_features
from ..rng import rng_for
from ..schema import DiscoveryReport, PlanStep, Profile, Recommendation
from ..stats.intervals import bootstrap_interval
from ..stats.metrics import get_metric, score
from .data import Table, to_table
from .evaluate import ROW_LOSS_METRICS, evaluate_family, predict_and_score
from .families.base import CLASSIFICATION, all_families, get_family
from .landmarks import metric_rope, run_landmarks
from .features import detect_candidates, feature_opportunities, scaler_opportunities
from .leakage import detect_leakage
from .preprocess import ColumnPipeline
from .quality import quality_profile
from .recommend import PooledLosses, recommend, undetermined_vs_leader
from .shift import detect_shift
from .splits import (
    ForwardChaining, GroupKFold, KFold, RepeatedSplitter, StratifiedGroupKFold, StratifiedKFold,
    _time_order, carve_holdout, splitter_to_dict,
)
from .structure import _strat, _time_numeric, detect_structure
from .target import profile_target
from .trials import candidate_families, refine, run_trials
from .types import ColumnType, infer_types, type_findings

DEFAULT_MAX_ROWS = 50_000
MIN_HOLDOUT_ROWS = 100
BASE_REPEATS = 2
MAX_REPEATS = 5
BOOSTING = ("lightgbm", "xgboost")

DecisionsIn = Optional[Mapping[str, Union[Decision, str, Dict[str, Any]]]]


# ---------------------------------------------------------------------------
# decisions
# ---------------------------------------------------------------------------

def _normalise_decisions(decisions: DecisionsIn) -> Dict[str, Decision]:
    if callable(decisions):                                   # report.decisions passed without ()
        decisions = decisions()
    out: Dict[str, Decision] = {}
    for fid, d in (decisions or {}).items():
        if isinstance(d, Decision):
            out[fid] = d
        elif isinstance(d, str):
            s = d.lower()
            if s in ("accept", "accepted"):
                out[fid] = Decision(DecisionState.ACCEPTED)
            elif s in ("reject", "rejected"):
                out[fid] = Decision(DecisionState.REJECTED)
            elif s == "pending":
                out[fid] = Decision()
            else:
                raise ValueError(f"decision for {fid!r} must be 'accept', 'reject' or 'pending', got {d!r}")
        elif isinstance(d, dict):
            out[fid] = Decision.from_dict(d)
        else:
            raise TypeError(f"decision for {fid!r} must be a Decision, a string or a dict")
    return out


class _Applier:
    """Which findings shape discovery's own experiments."""

    def __init__(self, decisions: Dict[str, Decision]):
        self.decisions = decisions
        self.applied: List[str] = []

    def state(self, fid: str) -> DecisionState:
        d = self.decisions.get(fid)
        return d.state if d is not None else DecisionState.PENDING

    def applies(self, f: Finding) -> bool:
        st = self.state(f.id)
        ok = st == DecisionState.ACCEPTED or (f.default_applied and st != DecisionState.REJECTED)
        if ok and f.suggested_action is not None and f.id not in self.applied:
            self.applied.append(f.id)
        return ok


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _log(verbose: bool, msg: str) -> None:
    if verbose:
        print(f"[discovery] {msg}", flush=True)


def _sample_rows(n: int, cap: int, y: np.ndarray, task: str, groups: Optional[np.ndarray], seed: int
                 ) -> np.ndarray:
    rng = rng_for(seed, "experiment_sample")
    if n <= cap:
        return np.arange(n)
    if groups is not None:
        g = np.unique(groups, return_inverse=True)[1]
        sizes = np.bincount(g)
        order = rng.permutation(len(sizes))
        keep, total = [], 0
        for u in order:
            if total >= cap:
                break
            keep.append(u)
            total += sizes[u]
        return np.flatnonzero(np.isin(g, keep))
    strat = _strat(y, task)
    out = []
    for c in np.unique(strat):
        rows = np.flatnonzero(strat == c)
        k = max(min(len(rows), 20), int(round(len(rows) * cap / n)))     # per-class floor
        out.append(rng.choice(rows, min(k, len(rows)), replace=False))
    return np.sort(np.concatenate(out))


def _apply_type_decisions(types: Dict[str, ColumnType], table: Table, tfind: List[Finding],
                          ap: _Applier, rng: np.random.Generator) -> Dict[str, ColumnType]:
    types = dict(types)
    for f in tfind:
        act = f.suggested_action
        if act is None or act.kind != "set_column_type":
            continue
        col = act.params["column"]
        if ap.applies(f):
            if types[col].type != act.params["type"]:
                types[col] = replace(types[col], type=act.params["type"], declared=True)
        elif f.default_applied:
            # rejected inference: fall back to how the column is stored
            c = table.columns[col]
            fallback = "numeric_continuous" if c.dtype.kind in "biuf" else (
                "categorical_low" if types[col].n_unique <= 50 else "categorical_high")
            types[col] = replace(types[col], type=fallback, declared=True)
    return types


def forward_splits(n_rows: int) -> int:
    """Forward chaining cannot be repeated, so its power comes from the number of
    test blocks: as many as keep each block >= 100 rows, between 5 and 10."""
    return int(np.clip((0.7 * n_rows) // 100, 5, 10))


def _splitter(kind: str, task: str, gap: int, seed: int, use_groups: bool, n_rows: int):
    if kind in ("temporal", "grouped_temporal"):
        return ForwardChaining(forward_splits(n_rows), gap=gap, min_train_frac=0.3)
    if use_groups:
        return StratifiedGroupKFold(5, seed) if task in CLASSIFICATION else GroupKFold(5, seed)
    return StratifiedKFold(5, seed) if task in CLASSIFICATION else KFold(5, seed)


def _holdout_interval(est, X, y, task, k, metric, rng, *, groups=None, block=1):
    vals = predict_and_score(est, X, y, task, k, [metric])
    point = vals[metric]
    if task in CLASSIFICATION:
        proba = est.predict_proba(X)

        def stat(idx):
            idx = idx.astype(int)
            return score(metric, y[idx], proba=proba[idx], n_classes=k)
    else:
        pred = est.predict(X)
        if task == "count":
            pred = np.clip(pred, 0.0, None)

        def stat(idx):
            idx = idx.astype(int)
            return score(metric, y[idx], values=pred[idx], n_classes=k)

    def safe(idx):
        v = stat(idx)
        return v if v is not None and math.isfinite(v) else np.nan

    iv = bootstrap_interval(np.arange(len(y), dtype=np.float64), rng, stat=safe, n_boot=1000,
                            groups=groups, block=block)
    lo, hi = iv.lo, iv.hi
    return point, ((lo, hi) if math.isfinite(lo) and math.isfinite(hi) else None), iv.method


def _fit_on(table: Table, types, y, rows, family, cfg, task, k, seed, drop):
    from ..rng import int_seed_for
    opts = family.preprocessing
    if drop:
        opts = replace(opts, drop=tuple(sorted(set(opts.drop) | set(drop))))
    pipe = ColumnPipeline(types, opts, task=task, n_classes=k, seed=int_seed_for(seed, "final_pipeline", family.name))
    X = pipe.fit_transform(table, rows, y[rows])
    est = family.make(cfg, int_seed_for(seed, "final_estimator", family.name), task, k)
    est.fit(X, y[rows])
    return pipe, est


def _build_code(applied: Sequence[str], target: Optional[str]) -> str:
    lines = ["# review each finding discovery applied, then accept (or reject) it:"]
    lines += [f"report.accept({fid!r})" for fid in applied] or ["# (none)"]
    lines.append("solution = report.build()          # applies accepted findings only")
    lines.append(f"solution.fit(data, target={target!r})" if target else "solution.fit(X, y)")
    return "\n".join(lines)


def _plan(rec: Recommendation, split_desc: str, split_refs: List[str], leak_drops: List[str],
          leak_refs: List[str], undetermined: Sequence[str], target: Optional[str], metric: str,
          applied: Sequence[str] = ()) -> List[PlanStep]:
    steps = [PlanStep("split", split_desc, tuple(split_refs))]
    if leak_drops:
        steps.append(PlanStep("preprocess", f"Exclude {', '.join(leak_drops)} (suspected leakage) unless you can "
                                            "confirm each is known at prediction time.", tuple(leak_refs)))
    steps.append(PlanStep("preprocess", "Fit every preprocessing step (imputation, scaling, encodings) on "
                                        "training folds only -- discovery's pipeline does this by construction."))
    if rec.family:
        cfg = ", ".join(f"{k}={v!r}" for k, v in rec.config.items())
        steps.append(PlanStep("train", f"Train {rec.family}" + (f" ({cfg})" if cfg else "") +
                              f" and judge it by {metric}.",
                              code=_build_code(applied, target)))
        if undetermined:
            steps.append(PlanStep("train", f"If the budget allows, also tune {', '.join(undetermined)}: "
                                           "discovery could not separate them from the best family."))
    else:
        steps.append(PlanStep("data", "Before training anything, improve the signal: review the leakage and "
                                      "data-quality findings, add features that carry information about the "
                                      "target, or check that the target is what you intend to predict."))
    steps.append(PlanStep("monitor", "Compare the deployed model's score with the holdout estimate in this "
                                     "report; a large drop points to shift between training and use."))
    return steps


NONLINEAR_FAMILIES = ("knn", "mlp", "lightgbm", "xgboost")


def _confirm_nonlinearity(findings: List[Finding], best: Dict[str, Any], *, rope: float, n_train: float,
                          n_test: float, pooled, seed: int) -> List[Finding]:
    """
    The landmarks are cheap and can miss non-linear structure that a stronger
    family exposes. If a non-linear candidate beats linear in the trials, say so
    (and withdraw a contradicting "linear is enough" hint).
    """
    from .recommend import compare_results
    lin = best.get("linear")
    if lin is None or lin.mean is None:
        return findings
    wins = []
    for f in NONLINEAR_FAMILIES:
        r = best.get(f)
        if r is None or r.mean is None:
            continue
        c = compare_results(r, lin, rope=rope, n_train=n_train, n_test=n_test, pooled=pooled)
        if c is not None and c.verdict == "better":
            wins.append((f, c))
    if not wins:
        return findings
    out = [f for f in findings if f.id != "family.linear_sufficient"]
    if any(f.id == "family.nonlinear_signal" for f in out):
        return out
    out.append(Finding(
        id="family.nonlinear_signal", category=Category.FAMILY, severity=Severity.INFO,
        statement=f"{' and '.join(f for f, _ in wins)} beat the linear model by more than the practical "
                  "threshold in the trials: the relationship is non-linear (thresholds, interactions or local "
                  "structure).",
        evidence=tuple(Evidence(test="paired_comparison", statistic=c.delta_mean, interval=c.delta_interval,
                                p_value=c.p_value, details={"a": c.a, "b": c.b, "rope": rope, "method": c.method},
                                seed=seed) for _, c in wins),
        confidence="high", confidence_basis="interval of the difference vs linear clears the ROPE"))
    return out


# ---------------------------------------------------------------------------
# the pipeline
# ---------------------------------------------------------------------------

def discover_ml(data: Any, target: Any = None, *, task: Optional[str] = None, groups: Any = None,
                time: Any = None, test: Any = None, column_names: Optional[Sequence[str]] = None,
                column_types: Optional[Dict[str, str]] = None, metric: Optional[str] = None,
                families: Optional[Sequence[str]] = None, budget: Optional[Budget] = None,
                decisions: DecisionsIn = None, seed: int = 0, holdout: float = 0.15,
                store: Any = None, verbose: bool = False) -> DiscoveryReport:
    table = to_table(data, target, groups=groups, time=time, column_names=column_names,
                     column_types=column_types)
    if table.target is None:
        raise ValueError("discovery needs a target: pass target= (a column name, index or array)")
    budget = budget or Budget.default_for_rows(table.n_rows)
    ledger = BudgetLedger(budget)
    ap = _Applier(_normalise_decisions(decisions))
    findings: List[Finding] = []
    limitations: List[str] = []
    n_total = table.n_rows

    # ---- 0. target
    prof, yenc, keep, tf = profile_target(table.target, task, metric)
    findings += tf
    if not keep.all():
        table = table.take(np.flatnonzero(keep))
    y = np.asarray(yenc)
    task, k, metric = prof.task, prof.n_classes, prof.metric
    secondary = tuple(prof.secondary_metrics)
    _log(verbose, f"{table.n_rows} rows with a target, task={task}, metric={metric}")

    # ---- 1. sample, types, quality
    cap = budget.max_rows or DEFAULT_MAX_ROWS
    declared_groups = None if table.groups is None else np.unique(
        np.array([str(v) for v in table.groups], dtype=object), return_inverse=True)[1]
    rows = _sample_rows(table.n_rows, cap, y, task, declared_groups, seed)
    sampling: Dict[str, Any] = {"rows_with_target": int(table.n_rows), "experiment_rows": int(len(rows))}
    if len(rows) < table.n_rows:
        table, y = table.take(rows), y[rows]
        sampling["method"] = "whole groups" if declared_groups is not None else "stratified random"
        findings.append(Finding(
            id="budget.sampled", category=Category.BUDGET, severity=Severity.INFO,
            statement=f"Experiments use {len(rows):,} of {sampling['rows_with_target']:,} rows "
                      f"({sampling['method']}); raise Budget(max_rows=...) to use more.",
            evidence=(Evidence(test="sampling", n=len(rows), details=dict(sampling)),),
            confidence="high", confidence_basis="row counts"))
    rng = rng_for(seed, "types")
    types = infer_types(table, rng)
    tfind = type_findings(types)
    findings += tfind
    types = _apply_type_decisions(types, table, tfind, ap, rng)
    qsum, qf, dup_ids = quality_profile(table, types, y)
    findings += qf
    dedupe = next((f for f in qf if f.id == "quality.duplicate_rows"), None)
    use_dup_groups = dedupe is not None and ap.applies(dedupe)

    # ---- 2. structure
    _log(verbose, "structure")
    sp, sf = detect_structure(table, types, y, task=task, n_classes=k, metric=metric, seed=seed)
    findings += sf
    by_id = {f.id: f for f in sf}
    kind = sp.kind
    group_arr, group_col = sp.groups, sp.group_column
    split_refs = [f.id for f in sf if f.suggested_action is not None and f.suggested_action.kind == "use_split"]
    if sp.group_column and f"structure.grouped:col={sp.group_column}" in by_id and \
            not ap.applies(by_id[f"structure.grouped:col={sp.group_column}"]):
        kind = {"grouped": "iid", "grouped_temporal": "temporal"}.get(kind, kind)
        group_arr, group_col = None, None
    for f in sf:                                            # accepted "possible groups"
        if f.id.startswith("structure.possible_groups:") and ap.applies(f) and group_arr is None:
            group_col = f.suggested_action.params["groups"]
            group_arr = np.unique(np.array([str(v) for v in table.columns[group_col]], dtype=object),
                                  return_inverse=True)[1]
            kind = "grouped_temporal" if kind == "temporal" else "grouped"
    temporal_f = next((f for f in sf if f.id.startswith("structure.temporal")), None)
    if temporal_f is not None and not ap.applies(temporal_f):
        kind = {"temporal": "iid", "grouped_temporal": "grouped"}.get(kind, kind)
    time_arr = sp.time if kind in ("temporal", "grouped_temporal") else None
    split_groups = group_arr
    if split_groups is None and use_dup_groups and kind == "iid":
        split_groups = dup_ids
    if group_col and kind in ("grouped", "grouped_temporal") and group_col in types:
        types = {c: t for c, t in types.items() if c != group_col}     # the group id is not a feature
    time_col = sp.time_column if kind in ("temporal", "grouped_temporal") else None

    # ---- 3. holdout
    n = table.n_rows
    strat_all = _strat(y, task)
    hold = np.array([], dtype=np.int64)
    dev = np.arange(n)
    if holdout and holdout > 0:
        hk = {"iid": "iid", "grouped": "grouped", "temporal": "temporal",
              "grouped_temporal": "grouped_temporal"}[kind]
        if hk == "iid" and split_groups is not None:
            hk = "grouped"
        if int(round(holdout * n)) >= MIN_HOLDOUT_ROWS:
            try:
                dev, hold = carve_holdout(n, structure=hk, frac=holdout, seed=seed, y=strat_all,
                                          groups=split_groups, time=time_arr)
            except ValueError as exc:
                limitations.append(f"No confirmation holdout: {exc}.")
        else:
            limitations.append(f"No confirmation holdout: {holdout:.0%} of {n} rows is under "
                               f"{MIN_HOLDOUT_ROWS} rows, too few for a useful estimate. The expected score "
                               "below is the cross-validated one, which is optimistic after model selection.")
    dtab, ydev = table.take(dev), y[dev]
    gdev = None if split_groups is None else split_groups[dev]
    tdev = None if time_arr is None else time_arr[dev]
    base = _splitter(kind, task, sp.gap, seed, gdev is not None, len(dev))
    forward = isinstance(base, ForwardChaining)
    sdev = _strat(ydev, task)

    def folds_for(repeats):
        return RepeatedSplitter(base, max(repeats) + 1).folds(len(ydev), y=sdev, groups=gdev, time=tdev,
                                                              repeats=repeats)

    folds = folds_for(list(range(1 if forward else BASE_REPEATS)))
    n_tr = float(np.mean([len(tr) for _, tr, _ in folds]))
    n_te = float(np.mean([len(te) for _, _, te in folds]))
    time_order = _time_order(len(ydev), tdev) if kind in ("temporal", "grouped_temporal") else None
    _log(verbose, f"split: {kind}, {len(folds)} folds on {len(ydev)} development rows, holdout {len(hold)}")

    # ---- 4. leakage
    _log(verbose, "leakage")
    dummy0 = evaluate_family(dtab, types, ydev, folds, get_family("baseline"), {}, task=task, n_classes=k,
                             metric=metric, seed=seed, stage="landmark", budget=ledger)
    full0 = evaluate_family(dtab, types, ydev, folds, get_family("linear"), {"alpha": 1e-3}, task=task,
                            n_classes=k, metric=metric, seed=seed, stage="landmark", budget=ledger)
    lsum, lf = detect_leakage(dtab, types, ydev, folds, task=task, n_classes=k, metric=metric, seed=seed,
                              dummy_score=dummy0.mean, full_score=full0.mean, time_order=time_order)
    findings += lf
    drops, leak_drops, leak_refs = [], [], []
    for f in findings:
        act = f.suggested_action
        if act is not None and act.kind == "drop_column" and ap.applies(f):
            c = act.params["column"]
            if c in types and c not in drops:
                drops.append(c)
                if f.category == Category.LEAKAGE:
                    leak_drops.append(c)
                    leak_refs.append(f.id)

    # ---- 5. landmarks, shift
    _log(verbose, "landmarks")
    signals, lres, lcomps, lmf = run_landmarks(dtab, types, ydev, folds, task=task, n_classes=k, metric=metric,
                                               seed=seed, drop=drops, secondary=secondary, budget=ledger)
    findings += lmf
    dummy = next((r for r in lres if r.family == "baseline"), None)
    rope = metric_rope(metric, ydev, dummy)
    test_table = None
    if test is not None:
        test_table = to_table(test, column_names=column_names, column_types=column_types)
    _log(verbose, "shift")
    ssum, shf = detect_shift(table, types, test=test_table, y=y,
                             time_order=_time_order(n, time_arr) if time_arr is not None else None,
                             exclude=[c for c in (time_col, group_col) if c], seed=seed)
    findings += shf

    # ---- 6. trials
    fams = candidate_families(task, families)
    have = {f.name for f in all_families()}
    if families is None and not any(b in have for b in BOOSTING):
        native = any(f.name == "gbdt" for f in fams)
        tree_friendly = bool(signals.get("nonlinear") or signals.get("interactions"))
        findings.append(Finding(
            id="limitation.no_boosting", category=Category.LIMITATION,
            severity=Severity.ADVISORY if (tree_friendly and not native) else Severity.INFO,
            statement=("LightGBM / XGBoost are not installed, so boosted trees were tried with tensor-optix's "
                       "native implementation (same accuracy on discovery's benchmark, about 10x slower); "
                       "install tensor-optix[discovery-boost] for faster runs." if native else
                       "Gradient-boosted trees were not tried (LightGBM / XGBoost are not installed)"
                       + (". The landmarks show non-linear structure, where boosted trees are often the "
                          "strongest tabular family: install tensor-optix[discovery-boost] and re-run."
                          if tree_friendly else "; install tensor-optix[discovery-boost] to include them.")),
            evidence=(Evidence(test="dependency_check", details={"nonlinear": signals.get("nonlinear"),
                                                                  "interactions": signals.get("interactions")}),),
            confidence="high", confidence_basis="import check"))
    if "mlp" not in have and families is None:
        limitations.append("Neural networks were not tried (torch is not installed).")
    _log(verbose, f"trials: {', '.join(f.name for f in fams)}")
    reserve = 0.15 * ledger.remaining_s()
    use_pooled = forward and metric in ROW_LOSS_METRICS
    state = run_trials(dtab, types, ydev, folds, task=task, n_classes=k, metric=metric, seed=seed,
                       ledger=ledger, families=fams, drop=drops, secondary=secondary, reserve_s=reserve,
                       collect_rows=use_pooled)
    pooled = None
    if use_pooled and dummy is not None:
        drows = np.full(len(ydev), np.nan)
        evaluate_family(dtab, types, ydev, folds, get_family("baseline"), {}, task=task, n_classes=k,
                        metric=metric, seed=seed, stage="landmark", drop=drops, row_losses=drows)
        pooled = PooledLosses(losses={**state.row_losses, dummy.key: drows}, order=time_order,
                              lag=int(sp.gap), metric=metric)
    for fam, cfg, why in state.skipped:
        findings_id = f"budget.skipped:family={fam}"
        if not any(f.id == findings_id for f in findings):
            findings.append(Finding(
                id=findings_id, category=Category.BUDGET, severity=Severity.ADVISORY,
                statement=f"Some {fam} configurations were not run ({why}). Increase Budget(time_s=...) to "
                          "include them.",
                evidence=(Evidence(test="budget", details={"config": cfg, "reason": why}),),
                confidence="high", confidence_basis="budget ledger"))
    if not forward:
        _log(verbose, "refinement")
        state = refine(dtab, types, ydev, state, splitter_folds=folds_for, next_repeat=BASE_REPEATS,
                       max_repeats=MAX_REPEATS,
                       undetermined=lambda s: undetermined_vs_leader(s.best, rope=rope, n_train=n_tr, n_test=n_te),
                       task=task, n_classes=k, metric=metric, seed=seed, ledger=ledger, drop=drops,
                       secondary=secondary, reserve_s=0.05 * budget.time_s)

    # ---- 7. recommendation + holdout
    rec, rcomps, rf = recommend(state.best, dummy, rope=rope, n_train=n_tr, n_test=n_te, metric=metric,
                                pooled=pooled)
    findings += rf
    findings = _confirm_nonlinearity(findings, state.best, rope=rope, n_train=n_tr, n_test=n_te,
                                     pooled=pooled, seed=seed)

    # ---- 7b. feature opportunities (development folds only; the holdout stays untouched)
    fsum, ff = {"tested": []}, []
    probe_name = rec.family if rec.family and rec.family != "mlp" else None
    if probe_name is None:
        cheap = [f for f, r in state.best.items() if f != "mlp" and r.mean is not None]
        hib_ = get_metric(metric).higher_is_better
        probe_name = (max(cheap, key=lambda f: state.best[f].mean if hib_ else -state.best[f].mean)
                      if cheap else None)
    if probe_name is not None and probe_name in state.best:
        _log(verbose, f"feature opportunities (model: {probe_name})")
        base_r = state.best[probe_name]
        ftypes = {c: t for c, t in types.items() if c not in drops}
        # persistence is a property of the series, not of drift: temporal families are tried whenever
        # the rows have a time order (detected temporal structure, or a datetime column)
        f_order, f_time = time_order, time_col
        if f_order is None:
            dts = [c for c, t in ftypes.items() if t.type == "datetime"]
            if dts:
                f_time = dts[0]
                f_order = np.argsort(np.nan_to_num(_time_numeric(dtab.columns[f_time]), nan=-np.inf),
                                     kind="stable")
        entity_groups = None if group_arr is None or not group_col else group_arr[dev]
        series_groups = entity_groups if f_order is not None else None
        cands = detect_candidates(dtab, ftypes, ydev, task=task, probe=probe_name,
                                  order=f_order, series_groups=series_groups, entity_groups=entity_groups,
                                  time_column=f_time, group_column=group_col, signals=signals,
                                  time_values=(tdev if f_time == time_col and tdev is not None else
                                               (_time_numeric(dtab.columns[f_time]) if f_time in dtab.columns
                                                else np.arange(len(ydev), dtype=np.float64))),
                                  ranked=lsum.get("single_feature") or [])
        from .recommend import compare_results

        def _cmp(a, b, pl):
            return compare_results(a, b, rope=rope, n_train=n_tr, n_test=n_te, pooled=pl)

        fsum, ff = feature_opportunities(
            dtab, ftypes, ydev, folds, task=task, n_classes=k, metric=metric, seed=seed,
            probe=(probe_name, dict(base_r.config)), base=base_r, candidates=cands, order=f_order,
            series_groups=series_groups, entity_groups=entity_groups, compare=_cmp, ledger=ledger, drop=drops,
            base_rows=state.row_losses.get(base_r.key) if use_pooled else None,
            pooled_factory=(lambda rows: PooledLosses(rows, time_order, int(sp.gap), metric)) if use_pooled
            else None)
        findings += ff
        ssum, sf2 = scaler_opportunities(
            dtab, ftypes, ydev, folds, task=task, n_classes=k, metric=metric, seed=seed,
            probe=(probe_name, dict(base_r.config)), base=base_r, compare=_cmp, ledger=ledger, drop=drops,
            base_rows=state.row_losses.get(base_r.key) if use_pooled else None,
            pooled_factory=(lambda rows: PooledLosses(rows, time_order, int(sp.gap), metric)) if use_pooled
            else None)
        fsum["scaling"] = ssum
        findings += sf2
        ff = ff + sf2
    if rec.family is not None and len(hold):
        _log(verbose, f"confirming {rec.family} on the holdout")
        fam = get_family(rec.family)
        try:
            pipe, est = _fit_on(table, types, y, dev, fam, rec.config, task, k, seed, drops)
            Xh = pipe.transform(table, hold)
            yh = y[hold]
            hg = None if split_groups is None else split_groups[hold]
            block = 1
            if time_arr is not None:
                order = _time_order(len(hold), time_arr[hold])
                Xh, yh = Xh[order], yh[order]
                hg = None if hg is None else hg[order]
                block = max(1, sp.gap, int(round(len(hold) ** (1 / 3))))
            point, iv, method = _holdout_interval(est, Xh, yh, task, k, metric, rng_for(seed, "holdout_boot"),
                                                  groups=hg if block == 1 else None, block=block)
            dpipe, dest = _fit_on(table, types, y, dev, get_family("baseline"), {}, task, k, seed, drops)
            dpoint = predict_and_score(dest, dpipe.transform(table, hold), y[hold], task, k, [metric])[metric]
            rec = replace(rec, expected_score=point, expected_interval=iv,
                          rationale=rec.rationale + (
                              f"On the untouched holdout ({len(hold)} rows) it scores {metric} {point:.4f}"
                              + (f" (95% {method.replace('_', ' ')} CI {iv[0]:.4f} to {iv[1]:.4f})" if iv else "")
                              + f"; the dummy baseline scores {dpoint:.4f} there.",))
        except Exception as exc:                               # report, never crash the whole run
            limitations.append(f"Holdout confirmation failed: {type(exc).__name__}: {exc}")
    elif rec.family is not None:
        cv = state.best[rec.family]
        rec = replace(rec, expected_score=cv.mean, expected_interval=cv.interval,
                      conditions=rec.conditions + ("Expected score is the cross-validated estimate (no holdout); "
                                                   "it is optimistic after choosing the best family.",))
    if leak_drops:
        rec = replace(rec, conditions=rec.conditions + (
            f"Assumes {', '.join(leak_drops)} {'is' if len(leak_drops) == 1 else 'are'} excluded "
            "(suspected leakage).",))

    # ---- 8. plan + report
    split_desc = {
        "iid": "Random (stratified) K-fold cross-validation" + (
            ", keeping duplicate rows in one fold" if gdev is not None and group_col is None else ""),
        "grouped": f"Group K-fold on '{group_col}': every {group_col} in exactly one fold",
        "temporal": f"Forward-chaining (time-series) CV on '{time_col or 'row order'}', gap {sp.gap} rows",
        "grouped_temporal": f"Forward-chaining CV on '{time_col or 'row order'}' moving whole '{group_col}' "
                            "groups",
    }[kind]
    und = [c.a.split(":")[1] for c in rcomps
           if c.verdict == "undetermined" and c.a.startswith("trial:") and c.a.split(":")[1] != rec.family]
    plan = _plan(rec, split_desc + ".", split_refs, leak_drops, leak_refs, und, table.target_name, metric,
                 applied=list(ap.applied))
    if ff:
        plan.insert(len(plan) - 1, PlanStep(
            "features", "Consider engineering: " + "; ".join(f.statement.split(" improves ")[0].replace(
                "Engineering ", "") for f in ff) + ". Accept the findings you want and build() adds them.",
            tuple(f.id for f in ff)))
    if forward:
        limitations.append(
            f"Scores describe forecasting about {int(round(n_te))} rows ahead (the length of each forward-chaining "
            "test block). Further ahead, families that cannot extrapolate beyond the training range -- boosted "
            "trees, k-NN -- tend to degrade faster than linear models or neural networks; validate at the horizon "
            "you will actually forecast.")
    if ledger.cuts:
        limitations.append(f"The budget forced {len(ledger.cuts)} cut(s); see the budget findings.")
    seen, uniq = set(), []
    for f in findings:
        if f.id not in seen:
            seen.add(f.id)
            uniq.append(f)
    experiments = [dummy0, full0] + [r for r in lres if r.key not in {dummy0.key, full0.key}] + state.results
    exp_seen, exps = set(), []
    for x in experiments:
        if x.key not in exp_seen:
            exp_seen.add(x.key)
            exps.append(x)
    stale = sorted(fid for fid in ap.decisions if fid not in seen)
    if stale:
        limitations.append(f"Decisions for findings this run did not produce were ignored: {', '.join(stale)}.")
    profile = Profile(
        domain="ml", n_total=n_total, n_sampled=int(table.n_rows), sampling=sampling,
        sections={
            "columns": [ct.to_dict() for ct in types.values()],
            "target": prof.to_dict(),
            "structure": {**sp.to_dict(), "evaluation_split": kind, "splitter": splitter_to_dict(base),
                          "development_rows": int(len(dev)), "holdout_rows": int(len(hold))},
            "quality": qsum,
            "leakage": lsum,
            "landmarks": signals,
            "shift": ssum,
            "applied_findings": list(ap.applied),
            "excluded_columns": drops,
            "features": fsum,
        },
        fingerprint=table.fingerprint())
    profile.sections["meta_features"] = ml_meta_features(jsonable(profile.sections))
    report = DiscoveryReport(
        domain="ml", profile=profile, findings=uniq, experiments=exps, comparisons=list(lcomps) + rcomps,
        recommendation=rec, plan=plan, limitations=limitations,
        input_summary={"rows": n_total, "columns": len(table.columns), "task": task, "metric": metric,
                       "target": table.target_name, "test_rows": None if test_table is None else test_table.n_rows},
        budget_requested=budget.to_dict(), budget_used=ledger.to_dict(), seed=seed)
    for fid, d in ap.decisions.items():
        if fid in seen:
            report._decisions[fid] = d
    if store is not None:
        from ..ledger_bridge import record_report
        report.ledger_refs = record_report(report, store)
    _log(verbose, f"done in {ledger.elapsed():.1f}s: {rec.verdict} -> {rec.family}")
    return report
