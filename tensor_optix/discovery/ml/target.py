"""
Target characterisation: what kind of prediction problem this is, decided
from the target *values* (not their storage dtype), plus the facts that change
how it must be evaluated (class balance, rare classes, skew).

The chosen metric is reported as a finding the user can reject.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..findings import Action, Category, Evidence, Finding, Severity
from .data import missing_mask

TASKS = ("binary", "multiclass", "regression", "count")
MULTICLASS_MAX = 50              # more integer levels than this -> numeric target
RARE_CLASS_MIN = 30


@dataclass(frozen=True)
class TargetProfile:
    task: str
    n: int                                   # rows with a target value
    n_missing: int
    classes: Tuple[Any, ...] = ()            # original labels, in encoded order
    class_counts: Tuple[int, ...] = ()
    metric: str = ""
    secondary_metrics: Tuple[str, ...] = ()
    task_reason: str = ""
    stats: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_classes(self) -> int:
        return len(self.classes)

    def to_dict(self) -> Dict[str, Any]:
        d = {"task": self.task, "rows_with_target": self.n, "rows_missing_target": self.n_missing,
             "metric": self.metric, "secondary_metrics": list(self.secondary_metrics),
             "how_task_was_decided": self.task_reason}
        if self.classes:
            d["classes"] = [str(c) for c in self.classes]
            d["class_counts"] = list(self.class_counts)
        d.update(self.stats)
        return d


def _labels(y: np.ndarray) -> np.ndarray:
    if y.dtype == object:
        return np.array([str(v) if not isinstance(v, (bool, np.bool_)) else bool(v) for v in y], dtype=object)
    return y


def decide_task(y: np.ndarray, override: Optional[str] = None) -> Tuple[str, str]:
    """``y`` without missing values. Returns (task, reason)."""
    if override is not None:
        if override not in TASKS:
            raise ValueError(f"task must be one of {TASKS}, got {override!r}")
        return override, "declared by the user"
    if y.dtype.kind == "b" or y.dtype == object:
        k = len(set(_labels(y).tolist()))
        if k < 2:
            raise ValueError("the target has a single value; there is nothing to predict")
        return ("binary" if k == 2 else "multiclass"), "non-numeric labels"
    yf = y.astype(np.float64)
    uniq = np.unique(yf)
    if len(uniq) < 2:
        raise ValueError("the target has a single value; there is nothing to predict")
    integral = bool(np.all(np.floor(yf) == yf))
    if len(uniq) == 2:
        return "binary", "two distinct values"
    if integral and len(uniq) <= MULTICLASS_MAX and len(uniq) <= max(3, len(yf) // 20):
        # Few integer codes relative to rows: treat as classes. Counts with few
        # values are ambiguous; the finding below tells the user how to change it.
        return "multiclass", f"{len(uniq)} integer values"
    if integral and uniq[0] >= 0:
        return "count", "non-negative integers with many values"
    return "regression", "continuous values"


def encode_classes(y: np.ndarray) -> Tuple[np.ndarray, Tuple[Any, ...]]:
    """Map labels to 0..K-1 in a deterministic (sorted) order."""
    lab = _labels(y)
    try:
        classes = tuple(sorted(set(lab.tolist())))
    except TypeError:
        classes = tuple(sorted(set(lab.tolist()), key=str))
    index = {c: i for i, c in enumerate(classes)}
    return np.array([index[v] for v in lab.tolist()], dtype=np.int64), classes


def profile_target(y_raw: np.ndarray, task_override: Optional[str] = None,
                   metric_override: Optional[str] = None
                   ) -> Tuple[TargetProfile, Optional[np.ndarray], np.ndarray, List[Finding]]:
    """
    Returns (profile, encoded target for rows that have one, mask of rows that
    have a target, findings).
    """
    miss = missing_mask(y_raw)
    keep = ~miss
    y = y_raw[keep]
    if len(y) < 10:
        raise ValueError(f"only {len(y)} rows have a target value; discovery needs at least 10")
    task, reason = decide_task(y, task_override)
    findings: List[Finding] = []
    stats: Dict[str, Any] = {}
    classes: Tuple[Any, ...] = ()
    counts: Tuple[int, ...] = ()
    if task in ("binary", "multiclass"):
        enc, classes = encode_classes(y)
        if task == "binary" and len(classes) != 2:
            raise ValueError(f"task='binary' but the target has {len(classes)} classes")
        counts = tuple(int(c) for c in np.bincount(enc, minlength=len(classes)))
        minority = min(counts)
        share = minority / len(enc)
        stats.update(minority_share=round(share, 4), imbalance_ratio=round(max(counts) / max(minority, 1), 2),
                     effective_classes=round(float(np.exp(-np.sum((np.array(counts) / len(enc)) *
                                                                  np.log(np.maximum(np.array(counts) / len(enc), 1e-12))))), 2))
        if task == "binary":
            metric, secondary = "auc", ("log_loss", "balanced_accuracy")
        else:
            metric, secondary = "macro_auc", ("log_loss", "balanced_accuracy")
        y_enc: Optional[np.ndarray] = enc
        if share < 0.1:
            findings.append(Finding(
                id="target.imbalance", category=Category.TARGET, severity=Severity.ADVISORY,
                statement=f"Classes are imbalanced: the rarest class is {share:.1%} of rows. Accuracy would "
                          "be misleading, so ranking metrics (AUC) and balanced accuracy are used.",
                evidence=(Evidence(test="class_balance", statistic=share, n=len(enc),
                                   details={"class_counts": dict(zip(map(str, classes), counts))}),),
                confidence="high", confidence_basis="exact counts"))
        rare = [str(c) for c, k in zip(classes, counts) if k < RARE_CLASS_MIN]
        if rare:
            findings.append(Finding(
                id="target.rare_classes", category=Category.TARGET, severity=Severity.WARNING,
                statement=f"{len(rare)} class(es) have fewer than {RARE_CLASS_MIN} examples ({', '.join(rare[:5])}"
                          f"{'…' if len(rare) > 5 else ''}); scores for them are unreliable.",
                evidence=(Evidence(test="class_counts", n=len(enc),
                                   details={"rare_classes": rare[:50]}),),
                confidence="high", confidence_basis="exact counts"))
    else:
        yf = y.astype(np.float64)
        y_enc = yf
        sd = float(yf.std())
        mean = float(yf.mean())
        m3 = float(np.mean((yf - mean) ** 3))
        skew = m3 / sd ** 3 if sd > 0 else 0.0
        stats.update(mean=round(mean, 6), std=round(sd, 6), min=float(yf.min()), max=float(yf.max()),
                     skewness=round(skew, 3), zero_fraction=round(float(np.mean(yf == 0)), 4))
        if task == "count":
            metric, secondary = "rmse", ("poisson_deviance", "mae", "r2")
        else:
            metric, secondary = "rmse", ("mae", "r2")
        if task == "count" and task_override is None:
            findings.append(Finding(
                id="target.count_like", category=Category.TARGET, severity=Severity.INFO,
                statement="The target is non-negative integers (a count); it is modelled as a number and "
                          "Poisson deviance is reported alongside RMSE.",
                evidence=(Evidence(test="target_values", n=len(yf), details={"min": float(yf.min()),
                                                                             "max": float(yf.max())}),),
                confidence="moderate", confidence_basis="value pattern"))
        if abs(skew) > 2.0:
            findings.append(Finding(
                id="target.skewed", category=Category.TARGET, severity=Severity.INFO,
                statement=f"The target is strongly skewed (skewness {skew:.1f}); a few large values dominate "
                          "squared-error metrics.",
                evidence=(Evidence(test="skewness", statistic=skew, n=len(yf)),),
                confidence="moderate", confidence_basis="sample skewness"))
    if metric_override is not None:
        metric = metric_override
    if task == "multiclass" and task_override is None and reason.endswith("integer values"):
        findings.append(Finding(
            id="target.integer_classes", category=Category.TARGET, severity=Severity.ADVISORY,
            statement=f"The target has {len(classes)} integer values and is treated as classes. If it is an "
                      "ordered quantity, declare task='regression' (or 'count').",
            evidence=(Evidence(test="target_values", n=len(y), details={"levels": len(classes)}),),
            confidence="moderate", confidence_basis="level count relative to rows"))
    findings.append(Finding(
        id="target.metric", category=Category.TARGET, severity=Severity.INFO,
        statement=f"Task: {task} ({reason}). Families are compared on {metric}"
                  f"{' (also reported: ' + ', '.join(secondary) + ')' if secondary else ''}.",
        evidence=(Evidence(test="task_detection", n=len(y), details={"task": task, "reason": reason}),),
        confidence="high" if task_override or reason == "non-numeric labels" else "moderate",
        confidence_basis="target values",
        suggested_action=Action("set_metric", {"metric": metric}), default_applied=True))
    if miss.any():
        findings.append(Finding(
            id="target.missing", category=Category.TARGET, severity=Severity.WARNING,
            statement=f"{int(miss.sum())} row(s) ({miss.mean():.1%}) have no target value and are excluded.",
            evidence=(Evidence(test="missing_target", statistic=float(miss.mean()), n=len(y_raw)),),
            confidence="high", confidence_basis="exact count"))
    prof = TargetProfile(task=task, n=int(keep.sum()), n_missing=int(miss.sum()), classes=classes,
                         class_counts=counts, metric=metric, secondary_metrics=secondary,
                         task_reason=reason, stats=stats)
    return prof, y_enc, keep, findings
