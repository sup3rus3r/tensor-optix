"""
Distribution shift: does the data the model will be used on look like the
data it is trained on?

Two settings:
  * ``test`` given (the rows the model will score): adversarial validation --
    an out-of-fold classifier trying to tell train rows from test rows -- plus
    per-feature two-sample tests, a train/test row-overlap check and columns
    missing from ``test``.
  * time-ordered data without ``test``: the earliest third vs the latest third
    (per-feature tests plus the target), i.e. drift inside the training data.

Per-feature tests: numeric -> Kolmogorov-Smirnov + PSI; categorical ->
chi-square on the category table (rare categories pooled so every expected
count is >= 5). p-values are Holm-adjusted over the features tested. A feature
is reported only when the adjusted p < 0.01 *and* the effect is large enough
to matter (KS D or PSI >= 0.1, total-variation distance >= 0.1): with many
rows, tiny shifts are always "significant".

Datetime columns are compared by range only (a test period after the
training period is the normal forecasting setup, not a problem) and are left
out of adversarial validation, where they would separate the sets trivially.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..findings import Category, Evidence, Finding, Severity
from ..rng import rng_for
from ..stats.compare import holm
from ..stats.distributions import chi2_sf
from ..stats.hypothesis import ks_2samp, psi
from .data import Table, missing_mask
from .evaluate import evaluate_family
from .families.base import get_family
from .quality import column_codes, row_group_ids
from .splits import RepeatedSplitter, StratifiedKFold
from .types import ColumnType, parse_embedded_number

ALPHA = 0.01
EFFECT = 0.1
MAJOR = {"psi": 0.25, "ks": 0.2, "tv": 0.2}
ADV_ADVISORY_LO = 0.55
ADV_WARNING = 0.70
MAX_ROWS_PER_SIDE = 20000
_SKIP = {"identifier", "text", "url", "list", "constant", "unknown"}
_NUMERIC = {"numeric_continuous", "numeric_discrete", "boolean", "embedded_number"}


def _as_float(col: np.ndarray) -> np.ndarray:
    if col.dtype.kind == "M":
        out = col.astype("datetime64[ns]").astype(np.int64).astype(np.float64)
        out[np.isnat(col)] = np.nan
        return out
    if col.dtype.kind in "biuf":
        return col.astype(np.float64)
    return np.fromiter((parse_embedded_number(v) for v in col), dtype=np.float64, count=len(col))


def _chi2_categorical(a: np.ndarray, b: np.ndarray) -> Tuple[float, float, int]:
    """(total-variation distance, chi-square p, df) between two categorical samples."""
    codes = column_codes(np.concatenate([a, b]))
    ca, cb = codes[:len(a)], codes[len(a):]
    k = int(codes.max()) + 1
    na, nb = np.bincount(ca, minlength=k).astype(np.float64), np.bincount(cb, minlength=k).astype(np.float64)
    tv = 0.5 * float(np.abs(na / len(a) - nb / len(b)).sum())
    tot = na + nb
    n = tot.sum()
    exp_min = np.minimum(tot * len(a) / n, tot * len(b) / n)
    rare = exp_min < 5
    if rare.any():
        na = np.append(na[~rare], na[rare].sum())
        nb = np.append(nb[~rare], nb[rare].sum())
        keep = (na + nb) > 0
        na, nb = na[keep], nb[keep]
    if len(na) < 2:
        return tv, 1.0, 0
    tot = na + nb
    ea, eb = tot * len(a) / n, tot * len(b) / n
    stat = float(((na - ea) ** 2 / ea).sum() + ((nb - eb) ** 2 / eb).sum())
    df = len(na) - 1
    return tv, chi2_sf(stat, df), df


def compare_features(a: Table, b: Table, types: Dict[str, ColumnType], columns: Sequence[str]
                     ) -> List[Dict[str, Any]]:
    """Per-feature two-sample comparison of ``a`` vs ``b`` with Holm-adjusted p-values."""
    rows: List[Dict[str, Any]] = []
    for c in columns:
        ct = types[c].type
        xa, xb = a.columns[c], b.columns[c]
        ma, mb = missing_mask(xa), missing_mask(xb)
        if (~ma).sum() < 10 or (~mb).sum() < 10:
            continue
        if ct in _NUMERIC:
            fa, fb = _as_float(xa[~ma]), _as_float(xb[~mb])
            fa, fb = fa[np.isfinite(fa)], fb[np.isfinite(fb)]
            if len(fa) < 10 or len(fb) < 10:
                continue
            ks = ks_2samp(fa, fb)
            p_ = psi(fa, fb)
            rows.append({"column": c, "test": "ks", "statistic": ks.statistic, "psi": p_, "p": ks.p_value,
                         "effect": max(ks.statistic, p_),
                         "major": ks.statistic >= MAJOR["ks"] or p_ >= MAJOR["psi"]})
        else:
            tv, p, df = _chi2_categorical(xa[~ma], xb[~mb])
            rows.append({"column": c, "test": "chi2", "statistic": tv, "df": df, "p": p, "effect": tv,
                         "major": tv >= MAJOR["tv"]})
    adj = holm([r["p"] for r in rows])
    for r, q in zip(rows, adj):
        r["p_holm"] = q
        r["shifted"] = q is not None and q < ALPHA and r["effect"] >= EFFECT
    return rows


def _cap(t: Table, rng: np.random.Generator) -> Table:
    if t.n_rows <= MAX_ROWS_PER_SIDE:
        return t
    return t.take(np.sort(rng.choice(t.n_rows, MAX_ROWS_PER_SIDE, replace=False)))


def adversarial_validation(train: Table, test: Table, types: Dict[str, ColumnType], columns: Sequence[str],
                           *, seed: int) -> Optional[Dict[str, Any]]:
    if not columns:
        return None
    rng = rng_for(seed, "adversarial")
    a, b = _cap(train, rng), _cap(test, rng)
    cols = {c: np.concatenate([a.columns[c], b.columns[c]]) for c in columns}
    side = np.concatenate([np.zeros(a.n_rows, dtype=np.int64), np.ones(b.n_rows, dtype=np.int64)])
    joint = Table(columns=cols, target=side)
    jtypes = {c: types[c] for c in columns}
    folds = RepeatedSplitter(StratifiedKFold(5, seed), 1).folds(len(side), y=side)
    r = evaluate_family(joint, jtypes, side, folds, get_family("tree"), {"depth": 3}, task="binary",
                        n_classes=2, metric="auc", seed=seed, stage="landmark")
    if r.mean is None:
        return None
    return {"auc": r.mean, "interval": r.interval, "n_train": a.n_rows, "n_test": b.n_rows}


def _feature_findings(rows, *, prefix: str, where: str, seed: int) -> List[Finding]:
    out = []
    for r in rows:
        if not r["shifted"]:
            continue
        c = r["column"]
        if r["test"] == "ks":
            what = f"KS D {r['statistic']:.3f}, PSI {r['psi']:.3f}"
        else:
            what = f"total-variation distance {r['statistic']:.3f}"
        out.append(Finding(
            id=f"{prefix}:col={c}", category=Category.SHIFT,
            severity=Severity.WARNING if r["major"] else Severity.ADVISORY,
            statement=f"'{c}' is distributed differently {where} ({what}; Holm-adjusted p {r['p_holm']:.2g}).",
            evidence=(Evidence(test=r["test"], statistic=r["statistic"], p_value=r["p_holm"],
                               effect_size=r["effect"],
                               details={k: v for k, v in r.items() if k not in ("column", "shifted")},
                               seed=seed),),
            confidence="high" if r["major"] else "moderate",
            confidence_basis="two-sample test, Holm-adjusted over all features"))
    return out


def detect_shift(table: Table, types: Dict[str, ColumnType], *, test: Optional[Table] = None,
                 y: Optional[np.ndarray] = None, time_order: Optional[np.ndarray] = None,
                 exclude: Sequence[str] = (), seed: int = 0) -> Tuple[Dict[str, Any], List[Finding]]:
    findings: List[Finding] = []
    summary: Dict[str, Any] = {}
    skip = set(exclude)
    feats = [c for c, ct in types.items() if ct.type not in _SKIP and c not in skip]

    if test is not None:
        missing = [c for c in table.names if c not in test.columns and c not in skip]
        for c in missing:
            findings.append(Finding(
                id=f"shift.missing_in_test:col={c}", category=Category.SHIFT,
                severity=Severity.WARNING if c in feats else Severity.ADVISORY,
                statement=f"'{c}' is in the training data but not in the test data; a model that uses it "
                          "cannot score the test rows.",
                evidence=(Evidence(test="schema_compare"),), confidence="high", confidence_basis="column names"))
        common = [c for c in feats if c in test.columns]
        dt_cols = [c for c in common if types[c].type == "datetime"]
        cmp_cols = [c for c in common if types[c].type != "datetime"]
        for c in dt_cols:
            tr, te = _as_float(table.columns[c]), _as_float(test.columns[c])
            tr, te = tr[np.isfinite(tr)], te[np.isfinite(te)]
            if len(tr) and len(te):
                after = float(np.mean(te > tr.max()))
                findings.append(Finding(
                    id=f"shift.time_range:col={c}", category=Category.SHIFT, severity=Severity.INFO,
                    statement=f"{after:.0%} of the test rows' '{c}' lies after the latest training value"
                              + (" -- the test set is a later period, so validate with forward chaining."
                                 if after > 0.5 else "."),
                    evidence=(Evidence(test="range_compare", statistic=after),),
                    confidence="high", confidence_basis="value ranges"))
        rows = compare_features(table, test, types, cmp_cols)
        summary["features"] = rows
        findings += _feature_findings(rows, prefix="shift.feature", where="in the test data than in training",
                                      seed=seed)
        adv = adversarial_validation(table, test, types, cmp_cols, seed=seed)
        summary["adversarial"] = adv
        if adv is not None and adv["interval"] is not None:
            lo = adv["interval"][0]
            if lo > ADV_ADVISORY_LO:
                top = [r["column"] for r in sorted(rows, key=lambda r: -r["effect"]) if r["shifted"]][:5]
                findings.append(Finding(
                    id="shift.adversarial", category=Category.SHIFT,
                    severity=Severity.WARNING if adv["auc"] >= ADV_WARNING else Severity.ADVISORY,
                    statement=(f"A classifier can tell training rows from test rows (AUC {adv['auc']:.3f}, "
                               f"95% CI {adv['interval'][0]:.3f}-{adv['interval'][1]:.3f}); cross-validated "
                               "scores will overstate test performance"
                               + (f". Most shifted: {', '.join(top)}." if top else ".")),
                    evidence=(Evidence(test="adversarial_validation", statistic=adv["auc"],
                                       interval=adv["interval"], null_value=0.5,
                                       details={"top_shifted": top, "n_train": adv["n_train"],
                                                "n_test": adv["n_test"]}, seed=seed),),
                    confidence="high" if lo > 0.6 else "moderate",
                    confidence_basis="out-of-fold AUC of a train-vs-test classifier"))
        if cmp_cols:
            ids = row_group_ids(Table(columns={c: np.concatenate([table.columns[c], test.columns[c]])
                                               for c in cmp_cols}), cmp_cols)
            tr_ids, te_ids = ids[:table.n_rows], ids[table.n_rows:]
            overlap = float(np.isin(te_ids, tr_ids).mean())
            summary["test_rows_in_train"] = overlap
            if overlap >= 0.01:
                findings.append(Finding(
                    id="shift.train_test_overlap", category=Category.SHIFT,
                    severity=Severity.WARNING if overlap >= 0.1 else Severity.ADVISORY,
                    statement=f"{overlap:.1%} of the test rows also appear in the training data (identical "
                              "feature values). Scores on them reflect memorisation; check how the sets were cut.",
                    evidence=(Evidence(test="exact_row_match", statistic=overlap),),
                    confidence="high", confidence_basis="exact match over all compared features"))

    if time_order is not None and len(time_order) >= 60:
        n = len(time_order)
        early, late = time_order[: n // 3], time_order[n - n // 3:]
        te_, tl_ = table.take(early), table.take(late)
        rows = compare_features(te_, tl_, types, [c for c in feats if types[c].type != "datetime"])
        summary["drift"] = rows
        findings += _feature_findings(rows, prefix="shift.drift_over_time",
                                      where="in the latest third of the data than in the earliest", seed=seed)
        if y is not None:
            ya, yb = y[early], y[late]
            discrete = len(np.unique(y)) <= 20 and bool(np.all(np.mod(y, 1) == 0))
            yt = ColumnType(name="<target>", type="categorical_low" if discrete else "numeric_continuous",
                            confidence="high", declared=True)
            r = compare_features(Table(columns={"<target>": ya}), Table(columns={"<target>": yb}),
                                 {"<target>": yt}, ["<target>"])
            summary["target_drift"] = r[0] if r else None
            if r and r[0]["shifted"]:
                findings.append(Finding(
                    id="shift.target_drift", category=Category.SHIFT, severity=Severity.ADVISORY,
                    statement=f"The target's distribution changes over time (earliest vs latest third, "
                              f"{r[0]['test']} effect {r[0]['effect']:.3f}). Recent data is the better guide to "
                              "future performance; weight or window it accordingly.",
                    evidence=(Evidence(test=r[0]["test"], statistic=r[0]["statistic"], p_value=r[0]["p_holm"],
                                       effect_size=r[0]["effect"], seed=seed),),
                    confidence="moderate", confidence_basis="two-sample test, earliest vs latest third"))
    return summary, findings
