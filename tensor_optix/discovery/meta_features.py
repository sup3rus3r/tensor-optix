"""
Versioned meta-features of a discovery run: a compact, numeric description of
the dataset and of how the reference models did on it.

They are *stored* (in the report profile and, with ``store=``, in the ledger's
``discovery.profile`` event) so that a future cross-task layer can learn from
them. Discovery itself never reads them back and never uses them, or any
other task's history, to make a decision.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

META_FEATURES_VERSION = 1

_SKILL_PERFECT = {"auc": 1.0, "macro_auc": 1.0, "accuracy": 1.0, "balanced_accuracy": 1.0, "macro_f1": 1.0,
                  "r2": 1.0, "rmse": 0.0, "mae": 0.0, "log_loss": 0.0, "brier": 0.0, "poisson_deviance": 0.0}


def _skill(score: Optional[float], dummy: Optional[float], metric: str) -> Optional[float]:
    perfect = _SKILL_PERFECT.get(metric)
    if score is None or dummy is None or perfect is None or perfect == dummy:
        return None
    return round((score - dummy) / (perfect - dummy), 6)


def ml_meta_features(sections: Dict[str, Any]) -> Dict[str, Any]:
    cols = sections.get("columns") or []
    target = sections.get("target") or {}
    structure = sections.get("structure") or {}
    lm = sections.get("landmarks") or {}
    counts: Dict[str, int] = {}
    for c in cols:
        counts[c.get("type", "unknown")] = counts.get(c.get("type", "unknown"), 0) + 1
    miss = [float(c.get("missing_fraction") or 0.0) for c in cols]
    scores = lm.get("scores") or {}
    metric = lm.get("metric") or target.get("metric") or ""
    dummy = scores.get("baseline")
    out: Dict[str, Any] = {
        "version": META_FEATURES_VERSION,
        "task": target.get("task"),
        "n_rows": target.get("rows_with_target"),
        "n_features": len(cols),
        "n_classes": len(target.get("classes") or []) or None,
        "minority_share": target.get("minority_share"),
        "type_counts": dict(sorted(counts.items())),
        "mean_missing_fraction": round(sum(miss) / len(miss), 6) if miss else 0.0,
        "max_missing_fraction": round(max(miss), 6) if miss else 0.0,
        "structure": structure.get("evaluation_split"),
        "metric": metric,
        "landmark_skill": {k: _skill(v, dummy, metric) for k, v in sorted(scores.items()) if k != "baseline"},
        "signals": {k: lm.get(k) for k in ("signal", "nonlinear", "linear_sufficient", "interactions",
                                           "local_structure", "nb_competitive")},
    }
    return out
