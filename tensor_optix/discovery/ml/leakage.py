"""
Leakage detection: features that contain information the model will not have
when it is used -- the outcome itself, something recorded after it, or an
artefact of how the file was produced.

All scores are out-of-fold on the *structure-correct* split passed in by the
caller (group / forward folds when the data needs them), so an id that only
"predicts" through group memorisation is not mistaken for a leak, and our own
evaluation cannot leak.

Skill is expressed as the fraction of the achievable improvement over the
dummy baseline a model reaches: ``e = (score - dummy) / (perfect - dummy)``
(perfect = 1 for AUC / accuracy / R2, 0 for losses), so one threshold works
for every metric.

Rules (deliberately conservative -- a strong legitimate feature is common, a
false "leak" alarm erodes trust):
  * near-perfect single feature (e >= 0.90; AUC >= 0.95) -> WARNING, dropped in discovery's
                                                        own experiments, ablation attached;
                                                        CRITICAL if its name matches the target's
  * one feature carries most of the full model's skill -> INFO only
  * an identifier predicts the target (e >= 0.2)       -> WARNING (no action: ids are dropped anyway)
  * missingness predicts the target (e >= 0.1)         -> ADVISORY
  * a feature equals the target shifted into the future (time-ordered data) -> CRITICAL
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..findings import Action, Category, Evidence, Finding, Severity
from ..stats.metrics import get_metric
from .data import Table, missing_mask
from .evaluate import FoldSpec, evaluate_family
from .families.base import get_family
from .types import ColumnType

NEAR_PERFECT = 0.90
DOMINANT_SHARE = 0.9
ID_SKILL = 0.2
MISSING_SKILL = 0.1
_PERFECT = {"auc": 1.0, "macro_auc": 1.0, "accuracy": 1.0, "balanced_accuracy": 1.0, "macro_f1": 1.0,
            "r2": 1.0, "rmse": 0.0, "mae": 0.0, "log_loss": 0.0, "brier": 0.0, "poisson_deviance": 0.0}
_UNUSABLE = {"identifier", "text", "url", "list", "constant", "unknown"}


def skill(score: Optional[float], dummy: float, metric: str) -> Optional[float]:
    if score is None or dummy is None:
        return None
    perfect = _PERFECT.get(metric)
    if perfect is None or perfect == dummy:
        return None
    return (score - dummy) / (perfect - dummy)


def _norm_name(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _name_matches_target(col: str, target: Optional[str]) -> bool:
    if not target:
        return False
    a, b = _norm_name(col), _norm_name(target)
    return len(b) >= 3 and (b in a or a in b)


def _single(table, types, y, folds, column: str, *, task, k, metric, seed, as_type: Optional[str] = None):
    ct = types[column]
    if as_type is not None:
        ct = ColumnType(name=column, type=as_type, confidence="high", declared=True)
    res = evaluate_family(table, {column: ct}, y, folds, get_family("tree"), {"depth": 3},
                          task=task, n_classes=k, metric=metric, seed=seed, stage="landmark")
    return res


def _missing_table(table: Table, column: str) -> Table:
    ind = missing_mask(table.columns[column]).astype(np.float64)
    return Table(columns={f"{column}__is_missing": ind}, target=table.target)


def detect_leakage(table: Table, types: Dict[str, ColumnType], y: np.ndarray, folds: Sequence[FoldSpec], *,
                   task: str, n_classes: int, metric: str, seed: int, dummy_score: float,
                   full_score: Optional[float], full_family: Tuple[str, Dict] = ("linear", {"alpha": 1e-3}),
                   time_order: Optional[np.ndarray] = None, max_columns: int = 200
                   ) -> Tuple[Dict[str, Any], List[Finding]]:
    findings: List[Finding] = []
    folds = list(folds)
    usable = [c for c, ct in types.items() if ct.type not in _UNUSABLE][:max_columns]
    e_full = skill(full_score, dummy_score, metric)
    per_feature: List[Dict[str, Any]] = []
    for c in usable:
        r = _single(table, types, y, folds, c, task=task, k=n_classes, metric=metric, seed=seed)
        per_feature.append({"column": c, "score": r.mean, "interval": r.interval,
                            "skill": skill(r.mean, dummy_score, metric)})
    ranked = sorted((p for p in per_feature if p["skill"] is not None), key=lambda p: -p["skill"])
    for i, p in enumerate(ranked):
        c, e = p["column"], p["skill"]
        nxt = ranked[i + 1]["skill"] if i + 1 < len(ranked) else None
        details = {"single_feature_score": p["score"], "single_feature_skill": round(e, 4),
                   "next_best_skill": None if nxt is None else round(nxt, 4),
                   "full_model_score": full_score, "dummy_score": dummy_score, "metric": metric}
        if e >= NEAR_PERFECT:
            abl = evaluate_family(table, types, y, folds, get_family(full_family[0]), full_family[1],
                                  task=task, n_classes=n_classes, metric=metric, seed=seed,
                                  stage="landmark", drop=(c,))
            details["full_model_without_column"] = abl.mean
            named = _name_matches_target(c, table.target_name)
            findings.append(Finding(
                id=f"leakage.single_feature:col={c}", category=Category.LEAKAGE,
                severity=Severity.CRITICAL if named else Severity.WARNING,
                statement=(f"'{c}' alone predicts the target almost perfectly ({metric} {p['score']:.4f}"
                           f"{'; its name also matches the target' if named else ''}). That usually means it "
                           "is recorded after the outcome or derived from it. It is excluded from discovery's "
                           "experiments -- verify it is available at prediction time before using it."),
                evidence=(Evidence(test="single_feature_cv", statistic=p["score"], interval=p["interval"],
                                   null_value=dummy_score, effect_size=e, details=details, seed=seed),),
                confidence="high" if e >= 0.99 or named else "moderate",
                confidence_basis="out-of-fold single-feature score on the structure-correct split",
                suggested_action=Action("drop_column", {"column": c}), default_applied=True))
        elif (e_full is not None and e_full > 0.3 and e >= DOMINANT_SHARE * e_full
              and (nxt is None or nxt <= 0.5 * e)):
            findings.append(Finding(
                id=f"leakage.dominant_feature:col={c}", category=Category.LEAKAGE, severity=Severity.INFO,
                statement=f"'{c}' carries most of the predictive signal on its own ({metric} {p['score']:.4f} "
                          f"vs {full_score:.4f} for all features). Make sure it is known at prediction time.",
                evidence=(Evidence(test="single_feature_cv", statistic=p["score"], interval=p["interval"],
                                   null_value=dummy_score, effect_size=e, details=details, seed=seed),),
                confidence="moderate", confidence_basis="share of full-model skill"))

    # ---- identifiers that predict the target
    for c, ct in types.items():
        if ct.type != "identifier" or table.columns[c].dtype.kind not in "iuf":
            continue
        r = _single(table, types, y, folds, c, task=task, k=n_classes, metric=metric, seed=seed,
                    as_type="numeric_continuous")
        e = skill(r.mean, dummy_score, metric)
        lo = skill(r.interval[0] if get_metric(metric).higher_is_better else r.interval[1],
                   dummy_score, metric) if r.interval else None
        if e is not None and e >= ID_SKILL and lo is not None and lo > 0:
            findings.append(Finding(
                id=f"leakage.identifier_predicts:col={c}", category=Category.LEAKAGE, severity=Severity.WARNING,
                statement=f"The identifier '{c}' predicts the target ({metric} {r.mean:.4f}): ids were assigned "
                          "in an order related to the outcome, or after it. It is not used as a feature, but "
                          "check whether the rows' order or other id-like fields carry the same leak.",
                evidence=(Evidence(test="single_feature_cv", statistic=r.mean, interval=r.interval,
                                   null_value=dummy_score, effect_size=e, seed=seed),),
                confidence="moderate", confidence_basis="out-of-fold score of the id as a number"))

    # ---- missingness as signal
    for c in usable:
        frac = float(missing_mask(table.columns[c]).mean())
        if not 0.01 <= frac <= 0.99:
            continue
        mt = _missing_table(table, c)
        name = f"{c}__is_missing"
        mtypes = {name: ColumnType(name=name, type="numeric_continuous", confidence="high", declared=True)}
        r = evaluate_family(mt, mtypes, y, folds, get_family("tree"), {"depth": 1}, task=task,
                            n_classes=n_classes, metric=metric, seed=seed, stage="landmark")
        e = skill(r.mean, dummy_score, metric)
        if e is not None and e >= MISSING_SKILL:
            findings.append(Finding(
                id=f"leakage.missingness_predicts:col={c}", category=Category.LEAKAGE, severity=Severity.ADVISORY,
                statement=f"Whether '{c}' is missing predicts the target ({metric} {r.mean:.4f} from the "
                          f"missing-indicator alone; {frac:.0%} missing). That can be genuine, or a sign the field "
                          "is filled in after the outcome.",
                evidence=(Evidence(test="missing_indicator_cv", statistic=r.mean, interval=r.interval,
                                   null_value=dummy_score, effect_size=e, seed=seed),),
                confidence="moderate", confidence_basis="out-of-fold score of the missing indicator"))

    # ---- future target encoded in a feature (time-ordered data only)
    if time_order is not None:
        yo = y[time_order].astype(np.float64)
        for c in usable:
            col = table.columns[c]
            if col.dtype.kind not in "biuf":
                continue
            xo = col[time_order].astype(np.float64)
            for h in (1, 2, 3):
                if len(yo) <= h + 10:
                    break
                a, b = xo[:-h], yo[h:]
                ok = np.isfinite(a)
                if ok.sum() < 20:
                    continue
                tol = 1e-9 * max(1.0, float(np.nanmax(np.abs(b))))
                match = float(np.mean(np.abs(a[ok] - b[ok]) <= tol))
                if match >= 0.9:
                    findings.append(Finding(
                        id=f"leakage.future_target:col={c}", category=Category.LEAKAGE,
                        severity=Severity.CRITICAL,
                        statement=f"'{c}' equals the target {h} step(s) later in {match:.0%} of rows: it encodes "
                                  "the future. It is excluded from discovery's experiments.",
                        evidence=(Evidence(test="lagged_target_equality", statistic=match,
                                           details={"lead_steps": h}),),
                        confidence="high", confidence_basis="exact value match",
                        suggested_action=Action("drop_column", {"column": c}), default_applied=True))
                    break

    summary = {"single_feature": [{k: v for k, v in p.items() if k != "interval"} for p in ranked[:30]],
               "full_model_skill": e_full}
    return summary, findings
