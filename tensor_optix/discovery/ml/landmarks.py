"""
Stage 1: landmarks -- cheap, well-understood models run on the same folds.
Their *relative* scores describe the problem:

  * anything better than the dummy?               -> is there learnable signal at all
  * depth-3 tree / kNN better than linear?        -> non-linear structure
  * linear equivalent to or better than both      -> a linear model is probably enough
  * depth-3 tree better than a stump              -> interactions / thresholds beyond one split
  * Gaussian NB close to the best                 -> features are largely independent given the class

Every comparison is paired on identical folds with the Nadeau-Bengio
corrected interval and a ROPE verdict, so a "hint" is only emitted when the
interval clears the ROPE. These are inputs to the family recommendation
(phase 4), reported as INFO; the single exception is "no signal", which is a
WARNING because every later stage depends on it.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..findings import Category, Evidence, Finding, Severity
from ..schema import Comparison, ExperimentResult
from ..stats.compare import compare_paired, default_rope
from ..stats.metrics import get_metric
from .data import Table
from .evaluate import FoldSpec, evaluate_family
from .families.base import get_family
from .types import ColumnType

CLASSIFICATION = ("binary", "multiclass")


def landmark_specs(task: str, n: int, p: int) -> List[Tuple[str, Dict[str, Any]]]:
    specs: List[Tuple[str, Dict[str, Any]]] = [("baseline", {}), ("linear", {"alpha": 1e-3}),
                                               ("tree", {"depth": 1}), ("tree", {"depth": 3}),
                                               ("knn", get_family("knn").configs(n, p, task)[0])]
    if task in CLASSIFICATION:
        specs += [("naive_bayes", {}), ("lda", {})]
    return specs


def metric_rope(metric: str, y: np.ndarray, dummy: Optional[ExperimentResult]) -> float:
    scale = None
    if dummy is not None and dummy.mean is not None:
        scale = abs(dummy.mean)
    return default_rope(metric, target_std=float(np.std(y)), scale=scale)


def compare(a: ExperimentResult, b: ExperimentResult, *, rope: float, n_train: float, n_test: float
            ) -> Optional[Comparison]:
    if a.fold_ids != b.fold_ids:
        raise ValueError("landmarks must share folds to be compared")
    try:
        verdict, iv = compare_paired(a.scores, b.scores, rope=rope, higher_is_better=a.higher_is_better,
                                     n_train=n_train, n_test=n_test)
    except ValueError:
        return None
    lo, hi = (iv.lo, iv.hi) if np.isfinite(iv.lo) and np.isfinite(iv.hi) else (None, None)
    return Comparison(a=a.key, b=b.key, n_pairs=iv.n, delta_mean=iv.mean,
                      delta_interval=None if lo is None else (lo, hi), rope=rope, verdict=verdict,
                      p_value=iv.p_two_sided, method=iv.method)


def run_landmarks(table: Table, types: Dict[str, ColumnType], y: np.ndarray, folds: Sequence[FoldSpec], *,
                  task: str, n_classes: int, metric: str, seed: int, drop: Sequence[str] = (),
                  secondary: Sequence[str] = (), budget=None
                  ) -> Tuple[Dict[str, Any], List[ExperimentResult], List[Comparison], List[Finding]]:
    folds = list(folds)
    n_train = float(np.mean([len(tr) for _, tr, _ in folds]))
    n_test = float(np.mean([len(te) for _, _, te in folds]))
    p = sum(1 for c in types if c not in set(drop))
    results: Dict[str, ExperimentResult] = {}
    for fam, cfg in landmark_specs(task, len(y), p):
        label = fam if fam != "tree" else f"tree_d{cfg['depth']}"
        results[label] = evaluate_family(table, types, y, folds, get_family(fam), cfg, task=task,
                                         n_classes=n_classes, metric=metric, secondary=secondary,
                                         seed=seed, stage="landmark", drop=drop, budget=budget)
    rope = metric_rope(metric, y, results.get("baseline"))
    hib = get_metric(metric).higher_is_better
    usable = {k: r for k, r in results.items() if r.mean is not None}

    comps: List[Comparison] = []
    verdicts: Dict[str, str] = {}

    def cmp(a: str, b: str) -> Optional[str]:
        if a not in usable or b not in usable:
            return None
        c = compare(usable[a], usable[b], rope=rope, n_train=n_train, n_test=n_test)
        if c is None:
            return None
        comps.append(c)
        verdicts[f"{a}_vs_{b}"] = c.verdict
        return c.verdict

    models = [k for k in usable if k != "baseline"]
    best = None
    if models:
        best = max(models, key=lambda k: usable[k].mean if hib else -usable[k].mean)
    for k in models:
        cmp(k, "baseline")
    v_tree_lin = cmp("tree_d3", "linear")
    v_knn_lin = cmp("knn", "linear")
    cmp("tree_d3", "tree_d1")
    v_knn_tree = cmp("knn", "tree_d3")
    if task in CLASSIFICATION and best and best != "naive_bayes":
        cmp("naive_bayes", best)

    beats_dummy = [k for k in models if verdicts.get(f"{k}_vs_baseline") == "better"]
    signals = {
        "rope": rope, "metric": metric, "best": best,
        "scores": {k: r.mean for k, r in results.items()},
        "intervals": {k: r.interval for k, r in results.items()},
        "failed": [k for k, r in results.items() if r.mean is None],
        "verdicts": verdicts,
        "signal": bool(beats_dummy),
        "nonlinear": v_tree_lin == "better" or v_knn_lin == "better",
        "linear_sufficient": (v_tree_lin in ("worse", "equivalent")) and (v_knn_lin in ("worse", "equivalent")),
        "interactions": verdicts.get("tree_d3_vs_tree_d1") == "better",
        "local_structure": v_knn_lin == "better" and v_knn_tree == "better",
        "nb_competitive": verdicts.get(f"naive_bayes_vs_{best}") == "equivalent" if best else False,
    }

    findings: List[Finding] = []
    base = usable.get("baseline")
    if not beats_dummy and base is not None and best is not None:
        findings.append(Finding(
            id="family.no_signal", category=Category.FAMILY, severity=Severity.WARNING,
            statement=(f"None of the quick reference models beats predicting the "
                       f"{'class prior' if task in CLASSIFICATION else 'mean'} by more than the "
                       f"practical threshold ({metric} {base.mean:.4f} for the dummy, best "
                       f"{usable[best].mean:.4f} from {best}). Either the features carry little signal about "
                       "the target or it needs a far more flexible model -- later stages test that, but "
                       "expect a weak result."),
            evidence=(Evidence(test="landmark_vs_dummy", statistic=usable[best].mean, null_value=base.mean,
                               details={"verdicts": {k: v for k, v in verdicts.items() if k.endswith("_vs_baseline")},
                                        "rope": rope}, seed=seed),),
            confidence="moderate", confidence_basis="paired corrected-t intervals vs the dummy, all within ROPE"))
    elif signals["nonlinear"]:
        which = [k for k, v in (("tree_d3", v_tree_lin), ("knn", v_knn_lin)) if v == "better"]
        findings.append(Finding(
            id="family.nonlinear_signal", category=Category.FAMILY, severity=Severity.INFO,
            statement=f"{' and '.join(which)} beat the linear model on identical folds: the relationship is "
                      "non-linear (thresholds, interactions or local structure). Tree ensembles and neural "
                      "networks are the candidates to watch.",
            evidence=tuple(Evidence(test="paired_corrected_t", statistic=c.delta_mean, interval=c.delta_interval,
                                    p_value=c.p_value, details={"a": c.a, "b": c.b, "rope": rope}, seed=seed)
                           for c in comps if c.b.startswith("landmark:linear") and c.verdict == "better"),
            confidence="high", confidence_basis="interval of the paired difference clears the ROPE"))
    elif signals["linear_sufficient"]:
        findings.append(Finding(
            id="family.linear_sufficient", category=Category.FAMILY, severity=Severity.INFO,
            statement="Neither a depth-3 tree nor k-nearest-neighbours beats the linear model: the signal "
                      "looks mostly linear in the (preprocessed) features. A regularised linear model is a "
                      "strong, cheap default here.",
            evidence=tuple(Evidence(test="paired_corrected_t", statistic=c.delta_mean, interval=c.delta_interval,
                                    p_value=c.p_value, details={"a": c.a, "b": c.b, "rope": rope}, seed=seed)
                           for c in comps if c.b.startswith("landmark:linear")),
            confidence="moderate", confidence_basis="paired intervals vs linear are equivalent or worse"))
    return signals, list(results.values()), comps, findings
