"""
Structure detection: are rows independent, or do they come in groups (several
rows per customer / patient / device) or in time order? The answer decides the
evaluation split, and a wrong split is the most common way a model looks far
better in validation than it will ever be in use.

Both tests are *split-gap* tests on a cheap landmark model -- they measure the
consequence that matters (how much a naive random split would overstate the
score), not just whether some statistic is non-zero:

* **Grouped**: random K-fold vs group K-fold on the same rows and features.
  ``gap = score(random) - score(group)`` with an interval from the two sets of
  fold scores (corrected variance). Flag when the interval clears the ROPE.

* **Temporal**: forward chaining trains on less data than K-fold, so a naive
  comparison would call any dataset "temporal". Instead each forward fold
  (train = the past, test = one later block) is *paired* with a control fold
  predicting the same test block from a random training set of the same
  size drawn from all other rows. The only difference is whether training
  comes from the past, so the paired gap isolates time dependence.

Without a time column, row order is tested the same way; an effect there is
reported (row order carries information) but not acted on, since discovery
cannot know whether row order is time.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..findings import Action, Category, Evidence, Finding, Severity
from ..rng import rng_for
from ..stats.compare import default_rope, rope_verdict
from ..stats.distributions import t_ppf
from ..stats.hypothesis import autocorrelation
from .data import Table, missing_mask
from .evaluate import evaluate_family
from .families.base import get_family
from .splits import ForwardChaining, GroupKFold, KFold, RepeatedSplitter, StratifiedKFold
from .types import ColumnType

MAX_GROUP_CANDIDATES = 5
MIN_GROUPS = 10


@dataclass(frozen=True)
class StructureProfile:
    kind: str                                    # "iid" | "grouped" | "temporal" | "grouped_temporal"
    group_column: Optional[str] = None
    time_column: Optional[str] = None            # None + temporal => explicit ``time=`` array
    groups: Optional[np.ndarray] = field(default=None, repr=False, compare=False)
    time: Optional[np.ndarray] = field(default=None, repr=False, compare=False)
    gap: int = 0                                  # forward-chaining gap (rows)
    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "group_column": self.group_column,
                "time_column": self.time_column, "forward_gap": self.gap, **self.evidence}


def _gap_interval(a: Sequence[Optional[float]], b: Sequence[Optional[float]], n_train: float,
                  n_test: float, higher_is_better: bool, level: float = 0.95) -> Tuple[float, float, float]:
    """(mean gap, lo, hi) of mean(a) - mean(b) for two independent sets of
    resampled-CV scores, each with Nadeau-Bengio corrected variance."""
    xa = np.array([v for v in a if v is not None], dtype=np.float64)
    xb = np.array([v for v in b if v is not None], dtype=np.float64)
    if len(xa) < 2 or len(xb) < 2:
        return float("nan"), -math.inf, math.inf
    sign = 1.0 if higher_is_better else -1.0
    corr = n_test / n_train
    va = (1.0 / len(xa) + corr) * xa.var(ddof=1)
    vb = (1.0 / len(xb) + corr) * xb.var(ddof=1)
    gap = sign * (xa.mean() - xb.mean())
    se = math.sqrt(va + vb)
    df = max(1, min(len(xa), len(xb)) - 1)                     # conservative df
    q = t_ppf(0.5 + level / 2.0, df)
    return gap, gap - q * se, gap + q * se


def _strat(y: np.ndarray, task: str) -> np.ndarray:
    if task in ("binary", "multiclass"):
        return y
    return np.searchsorted(np.quantile(y, [0.2, 0.4, 0.6, 0.8]), y)


def _candidate_groups(table: Table, types: Dict[str, ColumnType]) -> List[str]:
    n = table.n_rows
    out = []
    for name, col in table.columns.items():
        t = types[name].type
        if t not in ("identifier", "categorical_high", "numeric_discrete", "categorical_low"):
            continue
        miss = missing_mask(col)
        if miss.mean() > 0.2:
            continue
        keys = col[~miss]
        if keys.dtype == object:
            keys = np.array([str(v) for v in keys], dtype=object)
        n_groups = len(np.unique(keys))
        rows_per = (~miss).sum() / max(n_groups, 1)
        if n_groups >= MIN_GROUPS and rows_per >= 2.0 and n_groups <= n / 2:
            out.append((rows_per, name))
    out.sort(reverse=True)
    return [name for _, name in out[:MAX_GROUP_CANDIDATES]]


def _group_codes(col: np.ndarray) -> np.ndarray:
    if col.dtype == object:
        col = np.array(["\x00missing" if v is None else str(v) for v in col], dtype=object)
    return np.unique(col, return_inverse=True)[1].reshape(-1)


def _landmark_scores(table, types, y, folds, task, k, metric, seed, family, cfg):
    return evaluate_family(table, types, y, folds, get_family(family), cfg, task=task, n_classes=k,
                           metric=metric, seed=seed, stage="landmark")


def detect_structure(table: Table, types: Dict[str, ColumnType], y: np.ndarray, *, task: str,
                     n_classes: int, metric: str, seed: int, rope: Optional[float] = None,
                     probe_family: Tuple[str, Dict] = ("linear", {"alpha": 1e-3}),
                     repeats: int = 2) -> Tuple[StructureProfile, List[Finding]]:
    """
    ``y`` is the encoded target for every row of ``table``. ``probe_family`` is
    the landmark used for the split-gap tests (the caller passes the best
    cheap landmark). Declared ``groups`` / ``time`` on the table are honoured.
    """
    n = table.n_rows
    hib = metric not in ("rmse", "mae", "log_loss", "brier", "poisson_deviance")
    if rope is None:
        rope = default_rope(metric, target_std=float(np.std(y)) if task not in ("binary", "multiclass") else None,
                            scale=1.0)
    fam, cfg = probe_family
    strat = _strat(y, task)
    findings: List[Finding] = []
    evidence: Dict[str, Any] = {}
    base_split = StratifiedKFold(5, seed) if task in ("binary", "multiclass") else KFold(5, seed)
    random_folds = RepeatedSplitter(base_split, repeats).folds(n, y=strat)
    random_res = _landmark_scores(table, types, y, random_folds, task, n_classes, metric, seed, fam, cfg)
    n_tr = n * 4 / 5
    n_te = n / 5

    # ------------------------------------------------------------------ groups
    group_col: Optional[str] = table.groups_name
    groups_arr: Optional[np.ndarray] = None
    declared_groups = table.groups is not None
    best_group: Optional[Tuple[float, float, float, str, np.ndarray]] = None
    cands: List[Tuple[str, np.ndarray]] = []
    if declared_groups:
        cands = [(table.groups_name or "<groups>", _group_codes(table.groups))]
    else:
        cands = [(c, _group_codes(table.columns[c])) for c in _candidate_groups(table, types)]
    cands = [(name, codes) for name, codes in cands if len(np.unique(codes)) >= 5]
    # Bonferroni over undeclared candidates: the best of m intervals is reported
    level = 0.95 if declared_groups else 1.0 - 0.05 / max(1, len(cands))
    for name, codes in cands:
        gfolds = RepeatedSplitter(GroupKFold(5, seed), repeats).folds(n, groups=codes)
        gres = _landmark_scores(table, types, y, gfolds, task, n_classes, metric, seed, fam, cfg)
        gap, lo, hi = _gap_interval(random_res.scores, gres.scores, n_tr, n_te, hib, level)
        evidence.setdefault("group_tests", []).append(
            {"column": name, "gap": gap, "interval": [lo, hi],
             "random_cv": random_res.mean, "group_cv": gres.mean})
        if best_group is None or (math.isfinite(lo) and lo > best_group[1]):
            best_group = (gap, lo, hi, name, codes)
    grouped = False
    if best_group is not None:
        gap, lo, hi, name, codes = best_group
        verdict = rope_verdict(lo, hi, rope)
        ev = Evidence(test="split_gap_random_vs_group", statistic=gap, interval=(lo, hi), n=n,
                      null_value=0.0, effect_size=gap,
                      details={"group_column": name, "rope": rope, "interval_level": level, "n_candidates": len(cands), "landmark": fam, "metric": metric,
                               "random_cv": random_res.mean, "n_groups": int(len(np.unique(codes)))})
        if declared_groups:
            grouped = True
            group_col, groups_arr = name, codes
            findings.append(Finding(
                id=f"structure.grouped:col={name}", category=Category.STRUCTURE, severity=Severity.INFO,
                statement=f"Rows are grouped by '{name}' (declared); evaluation keeps each group in one fold. "
                          f"A random split would change the score by {gap:+.4f} {metric}.",
                evidence=(ev,), confidence="high", confidence_basis="declared by the user",
                suggested_action=Action("use_split", {"kind": "group", "groups": name}), default_applied=True))
        elif verdict == "better":
            grouped = True
            group_col, groups_arr = name, codes
            findings.append(Finding(
                id=f"structure.grouped:col={name}", category=Category.STRUCTURE, severity=Severity.CRITICAL,
                statement=f"Rows repeat per '{name}' and a random split overstates the score by {gap:.4f} "
                          f"{metric} ({level:.1%} CI [{lo:.4f}, {hi:.4f}]). Evaluation keeps each '{name}' in one fold.",
                evidence=(ev,), confidence="high" if lo > 2 * rope else "moderate",
                confidence_basis="split-gap interval clears the ROPE",
                suggested_action=Action("use_split", {"kind": "group", "groups": name}), default_applied=True))
        elif verdict == "undetermined" and gap > rope and lo > 0:
            findings.append(Finding(
                id=f"structure.possible_groups:col={name}", category=Category.STRUCTURE,
                severity=Severity.ADVISORY,
                statement=f"Rows repeat per '{name}'; a random split may overstate the score "
                          f"(gap {gap:+.4f} {metric}, {level:.1%} CI [{lo:.4f}, {hi:.4f}]: positive, but not "
                          f"conclusively larger than the {rope:.4f} practical threshold). If several "
                          f"rows can belong to the same real-world entity, declare groups='{name}'.",
                evidence=(ev,), confidence="low", confidence_basis="split-gap interval excludes 0 but overlaps the ROPE",
                suggested_action=Action("use_split", {"kind": "group", "groups": name}), default_applied=False))

    # ------------------------------------------------------------------ time
    time_col = table.time_name
    time_arr = table.time
    declared_time = time_arr is not None
    t_candidates: List[Tuple[str, Optional[np.ndarray]]] = []
    if declared_time:
        t_candidates = [(time_col or "<time>", time_arr)]
    else:
        dts = [c for c, ct in types.items() if ct.type == "datetime"]
        t_candidates = [(c, table.columns[c]) for c in dts[:2]]
        t_candidates.append(("<row order>", None))
    temporal = False
    gap_rows = 0
    for name, tvals in t_candidates:
        order = np.arange(n) if tvals is None else _sort_key(tvals)
        if order is None:
            continue
        gap_rows = _gap_rows(y[order])
        # A time block may hold a single class (e.g. rows sorted by target), where
        # AUC is undefined; log loss is defined on every block.
        t_metric = "log_loss" if task in ("binary", "multiclass") else metric
        # The candidate time column is excluded from the features for this test:
        # a model fitted on the past would otherwise extrapolate a spurious date
        # slope into the future, which is a property of that feature, not of
        # time structure in the data.
        t_types = {c: ct for c, ct in types.items() if c != name}
        res = _temporal_gap(table, t_types, y, order, gap_rows, task, n_classes, t_metric, seed, fam, cfg)
        if res is None:
            continue
        mean, lo, hi, nf, t_rope, metric_gap, dm_p = res
        verdict = rope_verdict(lo, hi, t_rope)
        ac = autocorrelation(y[order].astype(np.float64), 3)
        ev = Evidence(test="diebold_mariano_past_vs_matched_random", statistic=mean, interval=(lo, hi),
                      n=nf, null_value=0.0, p_value=dm_p, effect_size=metric_gap,
                      details={"time_column": name, "rope": t_rope, "metric": t_metric, "landmark": fam, "folds": nf,
                               "target_autocorrelation": [round(float(a), 3) for a in ac],
                               "forward_gap_rows": gap_rows})
        evidence.setdefault("time_tests", []).append({"column": name, "gap": mean, "interval": [lo, hi]})
        if name == "<row order>":
            rho, rho_p = _rank_trend(y)
            sorted_like = abs(rho) >= 0.3 and rho_p < 1e-3
            if verdict == "better" or sorted_like:
                evs = (ev, Evidence(test="spearman_row_index_vs_target", statistic=rho, p_value=rho_p, n=n))
                why = (f"the target trends with row position (Spearman rho = {rho:+.2f})" if sorted_like else
                       f"predicting a block of rows from earlier rows is worse than from random rows by "
                       f"{metric_gap:.4f} {t_metric} (Diebold-Mariano p = {dm_p:.2g})")
                findings.append(Finding(
                    id="structure.row_order_informative", category=Category.STRUCTURE,
                    severity=Severity.WARNING,
                    statement=f"Row order carries information about the target: {why}. If rows are in time "
                              "order, declare time= so evaluation never trains on the future; if the file is "
                              "sorted by the target or an id, check how it was produced.",
                    evidence=evs, confidence="high" if sorted_like else "moderate",
                    confidence_basis="rank correlation with row index" if sorted_like
                    else "Diebold-Mariano test, HAC variance"))
            continue
        if declared_time or verdict == "better":
            temporal = True
            time_col = name
            time_arr = tvals
            sev = Severity.INFO if declared_time else Severity.CRITICAL
            stmt = (f"Time order matters: training only on the past scores {metric_gap:.4f} {t_metric} worse than "
                    f"training on random rows (Diebold-Mariano p = {dm_p:.2g}). A random split would let the "
                    f"model see the future; evaluation uses forward chaining on '{name}'.")
            if declared_time:
                stmt = (f"Time order declared ('{name}'); evaluation uses forward chaining (gap {gap_rows} rows). "
                        f"Training on random rows instead would change the score by {metric_gap:+.4f} {t_metric}.")
            findings.append(Finding(
                id=f"structure.temporal:col={name}", category=Category.STRUCTURE, severity=sev,
                statement=stmt, evidence=(ev,),
                confidence="high" if declared_time or lo > 2 * rope else "moderate",
                confidence_basis="declared by the user" if declared_time else "paired split-gap interval",
                suggested_action=Action("use_split", {"kind": "forward", "time": name, "gap": gap_rows}),
                default_applied=True))
            break

    if grouped and temporal:
        kind = "grouped_temporal"
    elif grouped:
        kind = "grouped"
    elif temporal:
        kind = "temporal"
    else:
        kind = "iid"
    profile = StructureProfile(kind=kind, group_column=group_col if grouped else None,
                               time_column=time_col if temporal else None,
                               groups=groups_arr if grouped else None,
                               time=(_time_numeric(time_arr) if temporal and time_arr is not None else None),
                               gap=gap_rows if temporal else 0,
                               evidence={"landmark": fam, "rope": rope, **evidence})
    return profile, findings


def _rank_trend(y: np.ndarray) -> Tuple[float, float]:
    """Spearman correlation between row position and the target, with its
    large-sample two-sided p-value (z = rho * sqrt(n - 1))."""
    from ..stats.distributions import norm_sf
    from ..stats.metrics import rankdata
    n = len(y)
    if n < 10:
        return 0.0, 1.0
    ry = rankdata(y.astype(np.float64))
    rx = np.arange(1, n + 1, dtype=np.float64)
    if ry.std() == 0:
        return 0.0, 1.0
    rho = float(np.corrcoef(rx, ry)[0, 1])
    return rho, float(min(1.0, 2.0 * norm_sf(abs(rho) * math.sqrt(n - 1))))


def _dummy_log_loss(y: np.ndarray, k: int) -> float:
    p = np.bincount(y.astype(int), minlength=k) / len(y)
    p = np.clip(p, 1e-12, 1.0)
    return float(-np.sum(p * np.log(p)))


def _time_numeric(t: np.ndarray) -> np.ndarray:
    t = np.asarray(t)
    if t.dtype.kind == "M":
        out = t.astype("datetime64[ns]").astype(np.int64).astype(np.float64)
        out[np.isnat(t)] = np.nan
        return out
    if t.dtype == object:
        from .preprocess import _as_datetime
        dt = _as_datetime(t)
        if (~np.isnat(dt)).mean() > 0.9:
            return _time_numeric(dt)
        from .preprocess import _as_float
        return _as_float(t)
    return t.astype(np.float64)


def _sort_key(tvals: np.ndarray) -> Optional[np.ndarray]:
    t = _time_numeric(tvals)
    if np.isnan(t).mean() > 0.1:
        return None
    t = np.where(np.isnan(t), np.nanmax(t), t)
    return np.argsort(t, kind="stable")


def _gap_rows(y_ordered: np.ndarray) -> int:
    """First lag where the target's autocorrelation drops below 2/sqrt(n) (capped)."""
    n = len(y_ordered)
    max_lag = min(50, n // 20)
    if max_lag < 1:
        return 0
    ac = autocorrelation(y_ordered.astype(np.float64), max_lag)
    thr = 2.0 / math.sqrt(n)
    for lag, a in enumerate(ac, start=1):
        if abs(a) < thr:
            return lag - 1
    return max_lag


def _per_row_loss(est, X: np.ndarray, y: np.ndarray, task: str) -> np.ndarray:
    if task in ("binary", "multiclass"):
        P = est.predict_proba(X)
        p = np.clip(P[np.arange(len(y)), y.astype(int)], 1e-15, 1.0)
        return -np.log(p)                                       # per-row log loss
    pred = est.predict(X)
    return (pred - y.astype(np.float64)) ** 2                   # per-row squared error


def andrews_lag(d: np.ndarray, floor: int = 0) -> int:
    """
    Data-driven Bartlett-kernel bandwidth (Andrews 1991, AR(1) plug-in):
    ``1.1447 * (alpha * n) ** (1/3)`` with ``alpha = 4 rho^2 / ((1 - rho)^2 (1 + rho)^2)``,
    ``rho`` the lag-1 autocorrelation of ``d``. Persistent loss differentials get
    a long enough window; the result is at least ``floor`` and at most n / 4.
    """
    n = len(d)
    if n < 8:
        return max(floor, 1)
    dc = d - d.mean()
    den = float(dc @ dc)
    rho = float(dc[1:] @ dc[:-1]) / den if den > 0 else 0.0
    rho = min(max(rho, -0.97), 0.97)
    alpha = 4.0 * rho * rho / ((1.0 - rho) ** 2 * (1.0 + rho) ** 2)
    lag = int(math.ceil(1.1447 * (alpha * n) ** (1.0 / 3.0)))
    return int(min(max(lag, floor, 1), n // 4))


def newey_west_se(d: np.ndarray, lag: int) -> float:
    """HAC standard error of the mean of a (possibly autocorrelated) series."""
    n = len(d)
    dc = d - d.mean()
    var = float(dc @ dc) / n
    for k in range(1, min(lag, n - 1) + 1):
        w = 1.0 - k / (lag + 1.0)
        var += 2.0 * w * float(dc[k:] @ dc[:-k]) / n
    return math.sqrt(max(var, 0.0) / n)


def _temporal_gap(table, types, y, order, gap_rows, task, k, metric, seed, fam, cfg):
    """
    Diebold-Mariano comparison of two forecasters on the same time-ordered test
    rows: one trained only on the past (forward chaining), one trained on an
    equal-size random sample of all other rows. ``d_i = loss_past_i -
    loss_random_i`` (positive = the past alone is worse); the mean of ``d`` gets
    a Newey-West (HAC) standard error, so serial correlation in the losses is
    accounted for.

    Returns (mean d, lo, hi, n test rows, rope on the loss scale, pooled metric
    gap, p-value) or None when there are not enough rows.
    """
    from ..rng import int_seed_for
    from .preprocess import ColumnPipeline
    from ..stats.distributions import norm_sf

    n = len(order)
    try:
        fw = ForwardChaining(10, gap=gap_rows, min_train_frac=0.3).split(n)
    except ValueError:
        return None
    family = get_family(fam)
    rng = rng_for(seed, "temporal_control")
    d_parts, y_parts, lf_parts, lc_parts = [], [], [], []
    for j, (tr_pos, te_pos) in enumerate(fw):
        tr, te = order[tr_pos], order[te_pos]                  # te in time order
        pool = np.setdiff1d(np.arange(n), te)
        if len(pool) < len(tr) or len(tr) < 20:
            continue
        ctl = rng.choice(pool, len(tr), replace=False)
        losses = []
        for name, train_rows in (("past", np.sort(tr)), ("random", np.sort(ctl))):
            pipe = ColumnPipeline(types, family.preprocessing, task=task, n_classes=k,
                                  seed=int_seed_for(seed, "temporal", name, j))
            Xtr = pipe.fit_transform(table, train_rows, y[train_rows])
            est = family.make(cfg, int_seed_for(seed, "temporal_est", name, j), task, k)
            est.fit(Xtr, y[train_rows])
            losses.append(_per_row_loss(est, pipe.transform(table, te), y[te], task))
        lf_parts.append(losses[0])
        lc_parts.append(losses[1])
        d_parts.append(losses[0] - losses[1])
        y_parts.append(y[te])
    if len(d_parts) < 3:
        return None
    d = np.concatenate(d_parts)
    lf, lc = np.concatenate(lf_parts), np.concatenate(lc_parts)
    lag = andrews_lag(d, floor=max(gap_rows, int(4 * (len(d) / 100.0) ** (2.0 / 9.0))))
    se = newey_west_se(d, lag)
    mean = float(d.mean())
    z = 1.959963984540054
    lo, hi = (mean - z * se, mean + z * se) if se > 0 else (mean, mean)
    if se > 0:
        p = float(min(1.0, 2.0 * norm_sf(abs(mean) / se)))
    else:
        p = 0.0 if mean != 0 else 1.0
    if task in ("binary", "multiclass"):
        loss_rope = default_rope("log_loss", scale=_dummy_log_loss(y, k))
        metric_gap = float(lf.mean() - lc.mean())                # log-loss gap (per row)
    else:
        rmse_c = math.sqrt(float(lc.mean()))
        rope_rmse = default_rope("rmse", target_std=float(np.std(y)))
        loss_rope = 2.0 * rmse_c * rope_rmse + rope_rmse ** 2     # ROPE translated to the MSE scale
        metric_gap = math.sqrt(float(lf.mean())) - rmse_c         # RMSE gap
    return mean, lo, hi, int(len(d)), loss_rope, metric_gap, p
