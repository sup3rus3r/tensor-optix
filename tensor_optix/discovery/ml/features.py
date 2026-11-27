"""
Feature opportunities: which *kinds* of features are worth engineering.

Two steps, so a recommendation never rests on a statistical curiosity alone:

1. **Detect** properties of the data that a feature family could exploit
   (the tests only decide which families are worth trying):

   =====================  =====================================================  ==================
   property               test                                                   family tried
   =====================  =====================================================  ==================
   target persistence     Ljung-Box on the target in time order (per series)     target_lags
   feature persistence    Ljung-Box on numeric features in time order            rolling_features
   volatility clustering  McLeod-Li (Ljung-Box on squared AR(1) residuals)       volatility
   calendar structure     a datetime feature is present                          calendar_cycles
   skewed features        |skewness| > 2 (models sensitive to scale only)        log_transform
   interactions           landmark depth-3 tree beats a stump / non-linear       interactions
                          signal, and the model is additive (linear, NB)
   entity structure       rows grouped by an entity                              group_aggregates
   =====================  =====================================================  ==================

2. **Validate**: build the family's representative features (leakage-safe,
   see ``feature_gen``), evaluate the model family the user will train -- the
   recommendation, or the best cheaper family when that is a neural network --
   on the *same folds* with and without them, and compare (paired corrected-t,
   or the pooled-row Diebold-Mariano test under forward chaining). Holm across
   the families tried. A family is recommended only when its gain clears the
   ROPE *and* its Holm-adjusted p < 0.05; families that do not are reported as
   tested without gain.

Recommended families become ADVISORY findings with an ``add_feature_family``
action (never applied by default); accepted ones flow into ``report.build()``.
"""
from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..findings import Action, Category, Evidence, Finding, Severity
from ..stats.compare import holm
from ..stats.distributions import chi2_sf
from .data import Table
from .evaluate import evaluate_family
from .families.base import get_family
from .feature_gen import FeatureSpec, _num, augment, series_positions
from .types import ColumnType

DETECT_ALPHA = 0.01
GAIN_ALPHA = 0.05
SKEW_LIMIT = 2.0
MAX_COLUMNS = 5
_NUMERIC = ("numeric_continuous", "numeric_discrete")
_SCALE_SENSITIVE = ("linear", "knn", "naive_bayes", "mlp", "lda")
_ADDITIVE = ("linear", "naive_bayes")


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

def ljung_box(x: np.ndarray, h: int = 10) -> Tuple[float, float]:
    """Ljung-Box Q over lags 1..h and its chi-square(h) p-value. NaNs dropped."""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = len(x)
    h = min(h, n // 4)
    if n < 20 or h < 1:
        return 0.0, 1.0
    xc = x - x.mean()
    den = float(xc @ xc)
    if den <= 0:
        return 0.0, 1.0
    q = 0.0
    for k in range(1, h + 1):
        r = float(xc[k:] @ xc[:-k]) / den
        q += r * r / (n - k)
    q *= n * (n + 2)
    return q, chi2_sf(q, h)


def series_ljung_box(v: np.ndarray, positions: List[np.ndarray], h: int = 10) -> Tuple[float, float, int]:
    """
    Ljung-Box across several series: the per-series Q statistics are summed
    (independent series -> chi-square with the summed degrees of freedom).
    Returns (Q, p, number of series used).
    """
    q_tot, df_tot, used = 0.0, 0, 0
    for pos in positions:
        s = np.asarray(v[pos], dtype=np.float64)
        s = s[np.isfinite(s)]
        hh = min(h, len(s) // 4)
        if len(s) < 20 or hh < 1:
            continue
        q, _ = ljung_box(s, hh)
        q_tot += q
        df_tot += hh
        used += 1
    if used == 0:
        return 0.0, 1.0, 0
    return q_tot, chi2_sf(q_tot, df_tot), used


def mcleod_li(v: np.ndarray, positions: List[np.ndarray], h: int = 10) -> Tuple[float, float, int]:
    """Ljung-Box on squared residuals of a per-series AR(1) fit (ARCH effects)."""
    resid_sq = np.full(len(v), np.nan)
    for pos in positions:
        s = np.asarray(v[pos], dtype=np.float64)
        if len(s) < 20:
            continue
        a, b = s[:-1], s[1:]
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < 10:
            continue
        A = np.column_stack([np.ones(ok.sum()), a[ok]])
        coef, *_ = np.linalg.lstsq(A, b[ok], rcond=None)
        e = np.full(len(s), np.nan)
        e[1:] = b - (coef[0] + coef[1] * a)
        resid_sq[pos] = e * e
    return series_ljung_box(resid_sq, positions, h)


def skewness(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) < 20:
        return 0.0
    sd = x.std()
    return float(np.mean(((x - x.mean()) / sd) ** 3)) if sd > 0 else 0.0


# ---------------------------------------------------------------------------
# detection
# ---------------------------------------------------------------------------

def _top_numeric(types: Dict[str, ColumnType], ranked: Sequence[Dict[str, Any]], k: int) -> List[str]:
    order = [r["column"] for r in ranked if r.get("column") in types and types[r["column"]].type in _NUMERIC]
    rest = [c for c, t in types.items() if t.type in _NUMERIC and c not in order]
    return (order + rest)[:k]


def detect_candidates(table: Table, types: Dict[str, ColumnType], y: np.ndarray, *, task: str,
                      probe: str, order: Optional[np.ndarray], series_groups: Optional[np.ndarray],
                      entity_groups: Optional[np.ndarray], time_column: Optional[str],
                      group_column: Optional[str] = None, time_values: Optional[np.ndarray] = None,
                      signals: Dict[str, Any], ranked: Sequence[Dict[str, Any]]
                      ) -> List[Tuple[FeatureSpec, Dict[str, Any]]]:
    """(spec, trigger evidence) for every family worth validating."""
    out: List[Tuple[FeatureSpec, Dict[str, Any]]] = []
    n = table.n_rows
    numeric = [c for c, t in types.items() if t.type in _NUMERIC]
    series_meta = {"time": time_column or "<row order>",
                   "groups": group_column if series_groups is not None else None}
    skipped: List[Dict[str, Any]] = []
    if order is not None and time_values is not None:
        positions = series_positions(n, order, series_groups)
        tied = sum(int(np.sum(np.diff(np.asarray(time_values, dtype=np.float64)[pos]) == 0)) for pos in positions)
        if tied:
            # rows sharing a timestamp are concurrent: "earlier rows" is undefined, so past-only
            # features (and target lags in particular) could use same-time information
            skipped.append({"families": ["target_lags", "volatility", "rolling_features"],
                            "reason": f"{tied} rows share a timestamp with the previous row of their series; "
                                      "past-only features are undefined (declare groups= if each entity is "
                                      "its own series)"})
            order = None
    if order is not None:
        positions = series_positions(n, order, series_groups)
        yv = np.asarray(y, dtype=np.float64)
        q, p, used = series_ljung_box(yv, positions)
        if p < DETECT_ALPHA:
            out.append((FeatureSpec("target_lags", {"lags": [1, 2, 3], "windows": [5, 20], **series_meta}),
                        {"test": "ljung_box", "statistic": q, "p_value": p, "series": used, "on": "target"}))
        if task in ("regression", "count"):
            q2, p2, used2 = mcleod_li(yv, positions)
            if p2 < DETECT_ALPHA:
                out.append((FeatureSpec("volatility", {"windows": [10, 30], **series_meta}),
                            {"test": "mcleod_li", "statistic": q2, "p_value": p2, "series": used2, "on": "target"}))
        persistent = []
        for c in _top_numeric(types, ranked, 2 * MAX_COLUMNS):
            qc, pc, _ = series_ljung_box(_num(table.columns[c]), positions)
            if pc < DETECT_ALPHA:
                persistent.append((c, qc, pc))
        if persistent:
            cols = [c for c, _, _ in persistent[:MAX_COLUMNS]]
            out.append((FeatureSpec("rolling_features", {"columns": cols, "windows": [5, 20], **series_meta}),
                        {"test": "ljung_box", "on": cols,
                         "p_values": {c: pc for c, _, pc in persistent[:MAX_COLUMNS]}}))
    dts = [c for c, t in types.items() if t.type == "datetime"]
    if dts:
        out.append((FeatureSpec("calendar_cycles", {"columns": dts[:3]}),
                    {"test": "datetime_present", "on": dts[:3]}))
    if probe in _SCALE_SENSITIVE:
        skewed = []
        for c in numeric:
            sk = skewness(_num(table.columns[c]))
            if abs(sk) > SKEW_LIMIT:
                skewed.append((c, sk))
        if skewed:
            skewed.sort(key=lambda t: -abs(t[1]))
            cols = [c for c, _ in skewed[:MAX_COLUMNS]]
            out.append((FeatureSpec("log_transform", {"columns": cols}),
                        {"test": "skewness", "on": cols, "skewness": {c: round(s, 3) for c, s in skewed[:MAX_COLUMNS]}}))
    if probe in _ADDITIVE and (signals.get("interactions") or signals.get("nonlinear")):
        cols = _top_numeric(types, ranked, 4)
        if len(cols) >= 2:
            out.append((FeatureSpec("interactions", {"columns": cols}),
                        {"test": "landmark_signals", "interactions": signals.get("interactions"),
                         "nonlinear": signals.get("nonlinear"), "on": cols}))
    if entity_groups is not None:
        cols = _top_numeric(types, ranked, MAX_COLUMNS)
        if cols:
            out.append((FeatureSpec("group_aggregates", {"columns": cols, "groups": group_column}),
                        {"test": "grouped_structure", "on": cols}))
    if skipped:
        out.append((None, {"skipped": skipped}))
    return out


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------

_DESCRIPTION = {
    "target_lags": "lagged values and past rolling means of the target",
    "volatility": "past rolling volatility of the target",
    "rolling_features": "rolling means / standard deviations and first differences of persistent features",
    "calendar_cycles": "sine / cosine encodings of day of week, month, day of year (and hour)",
    "log_transform": "signed log transforms of heavily skewed features",
    "interactions": "pairwise products of the strongest features",
    "group_aggregates": "per-entity means, spreads and deviations of features",
}


def feature_opportunities(table: Table, types: Dict[str, ColumnType], y: np.ndarray, folds, *, task: str,
                          n_classes: int, metric: str, seed: int, probe: Tuple[str, Dict[str, Any]],
                          base, candidates: Sequence[Tuple[FeatureSpec, Dict[str, Any]]], order, series_groups,
                          entity_groups, compare, ledger=None, drop: Sequence[str] = (),
                          base_rows: Optional[np.ndarray] = None, pooled_factory=None
                          ) -> Tuple[Dict[str, Any], List[Finding]]:
    """
    ``base``: the probe family's ExperimentResult on the original features (same folds).
    ``compare(a, b, pooled)``: paired comparison returning a ``Comparison`` (a minus b, positive = a better).
    ``pooled_factory(results_rows)``: builds a ``PooledLosses`` for forward chaining, or None.
    """
    fam_name, cfg = probe
    family = get_family(fam_name)
    tested: List[Dict[str, Any]] = []
    comps = []
    notes: List[Dict[str, Any]] = []
    for spec, trigger in candidates:
        if spec is None:
            notes.extend(trigger.get("skipped") or [])
            continue
        if ledger is not None and (ledger.exhausted() or ledger.remaining_s() < 2 * max(base.fit_time_s, 0.1)):
            tested.append({"family": spec.family, "trigger": trigger, "status": "skipped_budget"})
            if ledger is not None:
                ledger.record_cut("features", spec.family, "not enough time left")
            continue
        try:
            aug = augment(table, [spec], y if spec.needs_target else None, order,
                          series_groups if spec.family != "group_aggregates" else entity_groups)
        except Exception as exc:                            # a generator must never break the run
            tested.append({"family": spec.family, "trigger": trigger, "status": "failed",
                           "error": f"{type(exc).__name__}: {exc}"})
            continue
        new = [c for c in aug.columns if c not in table.columns]
        atypes = dict(types)
        for c in new:
            atypes[c] = ColumnType(name=c, type="numeric_continuous", confidence="high", declared=True)
        rows = np.full(len(y), np.nan) if base_rows is not None else None
        r = evaluate_family(aug, atypes, y, folds, family, cfg, task=task, n_classes=n_classes, metric=metric,
                            seed=seed, stage="trial", drop=drop, budget=ledger, row_losses=rows)
        r = replace(r, config={**cfg, "features": spec.family})
        pooled = pooled_factory({r.key: rows, base.key: base_rows}) if pooled_factory and rows is not None else None
        c = compare(r, base, pooled)
        tested.append({"family": spec.family, "trigger": trigger, "status": r.status, "spec": spec.to_dict(),
                       "new_columns": new, "score": r.mean, "base_score": base.mean,
                       "gain": None if c is None else c.delta_mean,
                       "gain_interval": None if c is None else c.delta_interval,
                       "verdict": None if c is None else c.verdict, "p": None if c is None else c.p_value,
                       "method": None if c is None else c.method, "rope": None if c is None else c.rope,
                       "folds_improved": _folds_improved(r, base)})
        comps.append(c)
    adj = holm([t.get("p") for t in tested])
    findings: List[Finding] = []
    for t, q in zip(tested, adj):
        t["p_holm"] = q
        t["recommended"] = bool(t.get("verdict") == "better" and q is not None and q < GAIN_ALPHA)
        if not t["recommended"]:
            continue
        spec = FeatureSpec.from_dict(t["spec"])
        lo, hi = t["gain_interval"] or (None, None)
        needs_y = (" It uses past values of the target, so the target of earlier rows must be known when you "
                   "predict (one-step-ahead forecasting)." if spec.needs_target else "")
        findings.append(Finding(
            id=f"features.{spec.family}", category=Category.FEATURES, severity=Severity.ADVISORY,
            statement=(f"Engineering {_DESCRIPTION[spec.family]} improves {fam_name} by {t['gain']:+.4f} {metric} "
                       f"(95% CI {lo:+.4f} to {hi:+.4f}) on the same folds, {t['folds_improved']} of folds "
                       f"improved.{needs_y}"),
            evidence=(Evidence(test=t["trigger"].get("test", ""), statistic=t["trigger"].get("statistic"),
                               p_value=t["trigger"].get("p_value"), details={"trigger": t["trigger"]}),
                      Evidence(test="incremental_gain", statistic=t["gain"], interval=t["gain_interval"],
                               p_value=t["p_holm"], null_value=0.0,
                               details={"probe_family": fam_name, "with": t["score"], "without": t["base_score"],
                                        "method": t["method"], "new_columns": t["new_columns"]}, seed=seed)),
            confidence="high" if lo is not None and lo > 2 * t["rope"] else "moderate",
            confidence_basis="paired gain clears the ROPE, Holm-adjusted across the feature families tried",
            suggested_action=Action("add_feature_family", t["spec"]), default_applied=False))
    return {"probe_family": fam_name, "probe_config": cfg, "tested": tested, "not_tried": notes}, findings


SCALER_DESCRIPTION = {
    "robust": "robust scaling (median / interquartile range) of outlier-heavy columns",
    "quantile": "a quantile (rank-to-normal) transform of heavy-tailed columns",
}


def scaler_candidates(table: Table, types: Dict[str, ColumnType], probe: str) -> Tuple[List[Tuple[str, List[str], Dict[str, Any]]], Optional[str]]:
    """
    (method, columns, trigger) per scaler worth validating, plus a note when the
    model does not depend on scaling at all.
    """
    if probe not in _SCALE_SENSITIVE:
        return [], f"{probe} is invariant to monotone rescaling of features; the scaler does not matter"
    robust, quantile = [], []
    trig_r, trig_q = {}, {}
    for c, t in types.items():
        if t.type not in _NUMERIC:
            continue
        x = _num(table.columns[c])
        x = x[np.isfinite(x)]
        if len(x) < 50 or np.unique(x).size < 10:
            continue
        q1, q3 = np.percentile(x, [25, 75])
        iqr = q3 - q1
        out_frac = float(np.mean((x < q1 - 3 * iqr) | (x > q3 + 3 * iqr))) if iqr > 0 else 0.0
        sk = skewness(x)
        sd = x.std()
        kurt = float(np.mean(((x - x.mean()) / sd) ** 4) - 3.0) if sd > 0 else 0.0
        if out_frac >= 0.01:
            robust.append(c)
            trig_r[c] = round(out_frac, 4)
        if abs(sk) > SKEW_LIMIT or kurt > 7.0:
            quantile.append(c)
            trig_q[c] = {"skewness": round(sk, 3), "excess_kurtosis": round(kurt, 3)}
    out = []
    if robust:
        out.append(("robust", robust, {"test": "outlier_fraction_3iqr", "on": trig_r}))
    if quantile:
        out.append(("quantile", quantile, {"test": "skewness_kurtosis", "on": trig_q}))
    return out, None


def scaler_opportunities(table: Table, types: Dict[str, ColumnType], y: np.ndarray, folds, *, task: str,
                         n_classes: int, metric: str, seed: int, probe: Tuple[str, Dict[str, Any]], base,
                         compare, ledger=None, drop: Sequence[str] = (), base_rows: Optional[np.ndarray] = None,
                         pooled_factory=None) -> Tuple[Dict[str, Any], List[Finding]]:
    fam_name, cfg = probe
    family = get_family(fam_name)
    cands, note = scaler_candidates(table, {c: t for c, t in types.items() if c not in drop}, fam_name)
    tested: List[Dict[str, Any]] = []
    for method, cols, trigger in cands:
        if ledger is not None and (ledger.exhausted() or ledger.remaining_s() < 2 * max(base.fit_time_s, 0.1)):
            tested.append({"method": method, "trigger": trigger, "status": "skipped_budget"})
            ledger.record_cut("scaling", method, "not enough time left")
            continue
        opts = family.preprocessing
        opts = replace(opts, robust_columns=tuple(cols)) if method == "robust" else replace(
            opts, quantile_columns=tuple(cols))
        rows = np.full(len(y), np.nan) if base_rows is not None else None
        r = evaluate_family(table, types, y, folds, family, cfg, task=task, n_classes=n_classes, metric=metric,
                            seed=seed, stage="trial", drop=drop, budget=ledger, row_losses=rows,
                            preprocessing=opts)
        r = replace(r, config={**cfg, "scaler": method})
        pooled = pooled_factory({r.key: rows, base.key: base_rows}) if pooled_factory and rows is not None else None
        c = compare(r, base, pooled)
        tested.append({"method": method, "columns": cols, "trigger": trigger, "status": r.status,
                       "score": r.mean, "base_score": base.mean,
                       "gain": None if c is None else c.delta_mean,
                       "gain_interval": None if c is None else c.delta_interval,
                       "verdict": None if c is None else c.verdict, "p": None if c is None else c.p_value,
                       "rope": None if c is None else c.rope, "method_test": None if c is None else c.method,
                       "folds_improved": _folds_improved(r, base)})
    adj = holm([t.get("p") for t in tested])
    findings: List[Finding] = []
    for t, q in zip(tested, adj):
        t["p_holm"] = q
        t["recommended"] = bool(t.get("verdict") == "better" and q is not None and q < GAIN_ALPHA)
        if not t["recommended"]:
            continue
        lo, hi = t["gain_interval"]
        cols = t["columns"]
        findings.append(Finding(
            id=f"preprocess.scaler:method={t['method']}", category=Category.FEATURES, severity=Severity.ADVISORY,
            statement=(f"Using {SCALER_DESCRIPTION[t['method']]} ({', '.join(cols[:5])}"
                       f"{' ...' if len(cols) > 5 else ''}) instead of standard scaling improves {fam_name} by "
                       f"{t['gain']:+.4f} {metric} (95% CI {lo:+.4f} to {hi:+.4f}), {t['folds_improved']} of "
                       "folds improved."),
            evidence=(Evidence(test=t["trigger"]["test"], details={"trigger": t["trigger"]}),
                      Evidence(test="incremental_gain", statistic=t["gain"], interval=t["gain_interval"],
                               p_value=q, null_value=0.0,
                               details={"probe_family": fam_name, "with": t["score"], "without": t["base_score"],
                                        "method": t["method_test"]}, seed=seed)),
            confidence="high" if lo > 2 * t["rope"] else "moderate",
            confidence_basis="paired gain over standard scaling clears the ROPE (Holm-adjusted)",
            suggested_action=Action("scale", {"column": cols, "method": t["method"]}), default_applied=False))
    return {"tested": tested, "note": note}, findings


def _folds_improved(a, b) -> str:
    bm = dict(zip(b.fold_ids, b.scores))
    better = total = 0
    for fid, s in zip(a.fold_ids, a.scores):
        t = bm.get(fid)
        if s is None or t is None or not (math.isfinite(s) and math.isfinite(t)):
            continue
        total += 1
        better += (s > t) if a.higher_is_better else (s < t)
    return f"{better}/{total}"
