"""
``discover()``: the single entry point. Detects whether it was given a
dataset (supervised ML) or an environment (RL) and runs the matching pipeline.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

from .budget import Budget
from .schema import DiscoveryReport


def _looks_like_env(obj: Any) -> bool:
    """gymnasium / gym env, PettingZoo env, or a zero-arg factory returning one (duck-typed)."""
    if hasattr(obj, "reset") and (hasattr(obj, "step") or hasattr(obj, "last")) and \
            (hasattr(obj, "action_space") or hasattr(obj, "action_spaces") or hasattr(obj, "possible_agents")):
        return True
    return False


def detect_kind(data: Any) -> str:
    return "rl" if _looks_like_env(data) else "ml"


def discover(data: Any, target: Any = None, *, kind: str = "auto", task: Optional[str] = None,
             groups: Any = None, time: Any = None, test: Any = None,
             column_names: Optional[Sequence[str]] = None, column_types: Optional[Dict[str, str]] = None,
             metric: Optional[str] = None, families: Optional[Sequence[str]] = None,
             budget: Optional[Budget] = None, decisions: Any = None, seed: int = 0,
             holdout: float = 0.15, store: Any = None, verbose: bool = False) -> DiscoveryReport:
    """
    Understand a dataset, find the model family that fits it, and explain why.

    Args:
        data:   pandas DataFrame, dict of columns, 2-D numpy array, structured
                array, or an ``(X, y)`` tuple. (RL environments: coming with RL
                discovery.)
        target: target column name / index, or an array of targets.
        kind:   "auto" (default), "ml" or "rl".
        task:   force "binary" | "multiclass" | "regression" | "count".
        groups, time: declare the group column / time column (name or array).
        test:   the rows the model will be used on (features only) -- enables
                train-vs-test shift checks.
        column_types: {column: type} overrides of the inferred feature types.
        metric: force the comparison metric.
        families: restrict the candidate families (names).
        budget: ``Budget(time_s=..., max_rows=..., max_fits=...)``.
        decisions: {finding_id: "accept" | "reject" | Decision} from a previous
                report (``report.decisions()``), so the experiments follow them.
        seed:   everything is reproducible given the seed.
        holdout: fraction of rows kept aside to confirm the recommendation (0 = none).
        store:  an ``ExperimentStore`` (or a ledger path) to record the run in; opt-in.
        verbose: print progress.
    """
    if kind not in ("auto", "ml", "rl"):
        raise ValueError("kind must be 'auto', 'ml' or 'rl'")
    k = detect_kind(data) if kind == "auto" else kind
    if k == "rl":
        raise NotImplementedError(
            "RL discovery (environments) is not available yet in this version; "
            "discover() currently supports datasets.")
    from .ml.pipeline import discover_ml
    return discover_ml(data, target, task=task, groups=groups, time=time, test=test,
                       column_names=column_names, column_types=column_types, metric=metric,
                       families=families, budget=budget, decisions=decisions, seed=seed,
                       holdout=holdout, store=store, verbose=verbose)


def discover_ml(*args, **kwargs) -> DiscoveryReport:
    from .ml.pipeline import discover_ml as _d
    return _d(*args, **kwargs)
