"""
Data-quality checks: missingness, duplicate rows (and duplicates with
conflicting labels), near-perfectly correlated numeric columns, and extreme
outliers. These are facts about the data, reported with exact counts.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..findings import Action, Category, Evidence, Finding, Severity
from .data import Table, missing_mask
from .types import ColumnType, NUMERIC_TYPES

HIGH_MISSING = 0.4
COLLINEAR_R = 0.98
OUTLIER_MAD = 10.0


def column_codes(col: np.ndarray) -> np.ndarray:
    """Integer code per value (equal values -> equal codes; all missing values share one code)."""
    if col.dtype == object:
        keys = np.array(["\x00<missing>" if v is None else f"{type(v).__name__}:{v!r}" for v in col], dtype=object)
        return np.unique(keys, return_inverse=True)[1].astype(np.int64)
    if col.dtype.kind == "M":
        col = col.astype("datetime64[ns]").astype(np.int64)
    return np.unique(col, return_inverse=True, equal_nan=True)[1].astype(np.int64)


def row_group_ids(table: Table, columns: List[str]) -> np.ndarray:
    """Same id for rows whose values are identical over ``columns`` (vectorised)."""
    if not columns:
        return np.arange(table.n_rows, dtype=np.int64)
    codes = np.column_stack([column_codes(table.columns[c]) for c in columns])
    return np.unique(codes, axis=0, return_inverse=True)[1].reshape(-1).astype(np.int64)


def quality_profile(table: Table, types: Dict[str, ColumnType], y: Optional[np.ndarray]
                    ) -> Tuple[Dict[str, Any], List[Finding], np.ndarray]:
    """
    Returns (summary, findings, duplicate_group_ids). ``duplicate_group_ids``
    assigns the same integer to identical feature rows (used later so exact
    duplicates never straddle a train / test split).
    """
    findings: List[Finding] = []
    n = table.n_rows
    usable = [c for c in table.names if types[c].type not in ("constant", "identifier")]
    # ---- missingness
    miss = {c: float(missing_mask(table.columns[c]).mean()) for c in table.names}
    high = sorted((c for c, f in miss.items() if f >= HIGH_MISSING), key=lambda c: -miss[c])
    for c in high:
        findings.append(Finding(
            id=f"quality.high_missing:col={c}", category=Category.QUALITY, severity=Severity.ADVISORY,
            statement=f"'{c}' is missing in {miss[c]:.0%} of rows; values are imputed and a missing-indicator "
                      "is added, so the model can use the fact that it is missing.",
            evidence=(Evidence(test="missing_fraction", statistic=miss[c], n=n),),
            confidence="high", confidence_basis="exact count"))
    rows_any_missing = float(np.mean(np.any(np.column_stack(
        [missing_mask(table.columns[c]) for c in table.names]), axis=1))) if table.names else 0.0
    # ---- duplicates
    dup_ids = row_group_ids(table, usable or table.names)
    counts = np.bincount(dup_ids)
    dup_rows = int(np.sum(counts[dup_ids] > 1))
    n_conflicting = 0
    if y is not None and dup_rows:
        for g in np.flatnonzero(counts > 1):
            labels = y[dup_ids == g]
            if len(np.unique(labels)) > 1:
                n_conflicting += int(len(labels))
    if dup_rows:
        frac = dup_rows / n
        sev = Severity.WARNING if frac >= 0.05 else Severity.ADVISORY
        conflict_txt = (f" {n_conflicting} of them have identical features but different targets "
                        "(a ceiling on achievable accuracy, or a labelling problem).") if n_conflicting else ""
        findings.append(Finding(
            id="quality.duplicate_rows", category=Category.QUALITY, severity=sev,
            statement=f"{dup_rows} rows ({frac:.1%}) are exact duplicates of another row. Duplicates are kept "
                      "in the same fold during evaluation so they cannot inflate scores." + conflict_txt,
            evidence=(Evidence(test="exact_duplicates", statistic=frac, n=n,
                               details={"duplicate_rows": dup_rows, "conflicting_label_rows": n_conflicting}),),
            confidence="high", confidence_basis="exact row hashes",
            suggested_action=Action("dedupe", {"strategy": "group_in_folds"}), default_applied=True))
    # ---- collinearity among numeric columns
    num_cols = [c for c in usable if types[c].type in NUMERIC_TYPES and table.columns[c].dtype.kind in "biuf"]
    pairs = []
    if 2 <= len(num_cols) <= 400:
        X = np.column_stack([table.columns[c].astype(np.float64) for c in num_cols])
        ok = ~np.any(np.isnan(X), axis=1)
        if ok.sum() >= 30:
            Xo = X[ok]
            sd = Xo.std(axis=0)
            valid = sd > 0
            if valid.sum() >= 2:
                Z = (Xo[:, valid] - Xo[:, valid].mean(axis=0)) / sd[valid]
                R = (Z.T @ Z) / len(Z)
                names = [c for c, v in zip(num_cols, valid) if v]
                iu = np.triu_indices(len(names), 1)
                for i, j in zip(*iu):
                    if abs(R[i, j]) >= COLLINEAR_R:
                        pairs.append((names[i], names[j], float(R[i, j])))
    for a, b, r in pairs[:20]:
        findings.append(Finding(
            id=f"quality.collinear:cols={a}|{b}", category=Category.QUALITY, severity=Severity.INFO,
            statement=f"'{a}' and '{b}' are almost perfectly correlated (r = {r:.3f}); one is redundant.",
            evidence=(Evidence(test="pearson_r", statistic=r, n=n),),
            confidence="high", confidence_basis="sample correlation"))
    # ---- extreme outliers (robust z beyond OUTLIER_MAD)
    outliers = {}
    for c in num_cols:
        x = table.columns[c].astype(np.float64)
        x = x[np.isfinite(x)]
        if len(x) < 30:
            continue
        med = np.median(x)
        mad = 1.4826 * np.median(np.abs(x - med))
        if mad == 0:
            continue
        frac = float(np.mean(np.abs(x - med) / mad > OUTLIER_MAD))
        if frac > 0:
            outliers[c] = frac
    for c, f in sorted(outliers.items(), key=lambda kv: -kv[1])[:10]:
        if f >= 0.001:
            findings.append(Finding(
                id=f"quality.outliers:col={c}", category=Category.QUALITY, severity=Severity.INFO,
                statement=f"'{c}' has extreme values ({f:.2%} of rows more than {OUTLIER_MAD:g} robust SDs from "
                          "the median); robust scaling is used for it.",
                evidence=(Evidence(test="robust_z_outliers", statistic=f, n=n),),
                confidence="moderate", confidence_basis="median / MAD rule"))
    summary = {
        "rows_with_any_missing": round(rows_any_missing, 4),
        "columns_with_missing": int(sum(1 for f in miss.values() if f > 0)),
        "duplicate_rows": dup_rows,
        "conflicting_duplicate_rows": n_conflicting,
        "collinear_pairs": len(pairs),
        "columns_with_extreme_outliers": len(outliers),
    }
    return summary, findings, dup_ids.astype(np.int64)
