"""
Stages 2-3: short trials of the candidate families, then refinement.

Stage 2 runs every available candidate family's small fixed portfolio on the
development folds (identical folds for everyone, so all comparisons are
paired). Families run in simplicity order; before each experiment the
remaining time is compared with a cost prediction from what has already been
measured, and an experiment that would not fit is skipped and recorded as a
budget cut -- never silently dropped.

Stage 3 adds CV repeats for the leading families while their comparison with
the leader is still undetermined and budget remains (more paired folds narrow
the interval). Forward-chaining CV is deterministic, so it is never repeated.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..budget import BudgetLedger
from ..schema import ExperimentResult
from ..stats.intervals import corrected_resampled_t
from .data import Table
from .evaluate import FoldSpec, evaluate_family
from .families.base import Family, all_families, get_family
from .types import ColumnType


@dataclass
class TrialState:
    """Everything stage 2/3 produced, keyed so the recommendation can find it."""
    results: List[ExperimentResult] = field(default_factory=list)
    best: Dict[str, ExperimentResult] = field(default_factory=dict)       # family -> best config result
    skipped: List[Tuple[str, Dict[str, Any], str]] = field(default_factory=list)   # (family, config, reason)
    refined: List[str] = field(default_factory=list)
    row_losses: Dict[str, np.ndarray] = field(default_factory=dict)      # result key -> per-row losses


def candidate_families(task: str, names: Optional[Sequence[str]] = None) -> List[Family]:
    fams = [f for f in all_families() if f.candidate and f.supports(task)]
    if names is not None:
        wanted = set(names)
        unknown = wanted - {f.name for f in all_families()}
        if unknown:
            raise ValueError(f"unknown families {sorted(unknown)}; known: {sorted(f.name for f in all_families())}")
        fams = [f for f in fams if f.name in wanted]
    return sorted(fams, key=lambda f: (f.cost_rank, f.name))


def _better(a: ExperimentResult, b: Optional[ExperimentResult]) -> bool:
    if b is None or b.mean is None:
        return a.mean is not None
    if a.mean is None:
        return False
    return a.mean > b.mean if a.higher_is_better else a.mean < b.mean


def _merge(a: ExperimentResult, b: ExperimentResult, n_train: float, n_test: float) -> ExperimentResult:
    """Append ``b``'s folds (new repeats) to ``a``; interval / mean recomputed."""
    scores = tuple(a.scores) + tuple(b.scores)
    ok = [s for s in scores if s is not None]
    mean = float(np.mean(ok)) if ok else None
    interval = a.interval
    if len(ok) >= 2:
        iv = corrected_resampled_t(ok, n_train, n_test)
        interval = (iv.lo, iv.hi)
    sec = {k: tuple(v) + tuple(b.secondary.get(k, (None,) * len(b.scores))) for k, v in a.secondary.items()}
    return replace(a, fold_ids=tuple(a.fold_ids) + tuple(b.fold_ids), scores=scores, secondary=sec,
                   mean=mean, interval=interval, fit_time_s=round(a.fit_time_s + b.fit_time_s, 4),
                   stop_reasons=tuple(a.stop_reasons) + tuple(b.stop_reasons),
                   status="ok" if ok else a.status)


def run_trials(table: Table, types: Dict[str, ColumnType], y: np.ndarray, folds: Sequence[FoldSpec], *,
               task: str, n_classes: int, metric: str, seed: int, ledger: BudgetLedger,
               families: Sequence[Family], drop: Sequence[str] = (), secondary: Sequence[str] = (),
               reserve_s: float = 0.0, collect_rows: bool = False) -> TrialState:
    """``reserve_s`` is kept back for later stages (refinement, holdout refit).
    ``collect_rows`` keeps per-row test losses (forward chaining: rows are tested once)."""
    state = TrialState()
    folds = list(folds)
    n, p = len(y), len(types)
    per_fold_time: Dict[str, float] = {}
    for fam in families:
        for cfg in fam.configs(n, p, task):
            predicted = per_fold_time.get(fam.name)
            if predicted is not None:
                need = predicted * len(folds)
                if need > ledger.remaining_s() - reserve_s:
                    state.skipped.append((fam.name, cfg, f"predicted {need:.1f}s > remaining budget"))
                    ledger.record_cut("trials", f"{fam.name} {cfg}", "not enough time left for another configuration")
                    continue
            if ledger.exhausted() or ledger.remaining_s() <= reserve_s:
                state.skipped.append((fam.name, cfg, "budget exhausted"))
                ledger.record_cut("trials", f"{fam.name} {cfg}", "budget exhausted")
                continue
            rows = np.full(len(y), np.nan) if collect_rows else None
            r = evaluate_family(table, types, y, folds, fam, cfg, task=task, n_classes=n_classes, metric=metric,
                                secondary=secondary, seed=seed, stage="trial", drop=drop, budget=ledger,
                                row_losses=rows)
            if rows is not None:
                state.row_losses[r.key] = rows
            done = sum(1 for s in r.scores if s is not None) or 1
            per_fold_time[fam.name] = max(per_fold_time.get(fam.name, 0.0), r.fit_time_s / done)
            state.results.append(r)
            if r.status == "skipped_budget":
                ledger.record_cut("trials", f"{fam.name} {cfg}", "budget ran out during the experiment")
            if _better(r, state.best.get(fam.name)):
                state.best[fam.name] = r
    return state


def refine(table: Table, types: Dict[str, ColumnType], y: np.ndarray, state: TrialState, *,
           splitter_folds: Callable[[Sequence[int]], List[FoldSpec]], next_repeat: int, max_repeats: int,
           undetermined: Callable[[TrialState], List[str]], task: str, n_classes: int, metric: str, seed: int,
           ledger: BudgetLedger, drop: Sequence[str] = (), secondary: Sequence[str] = (),
           reserve_s: float = 0.0, top: int = 3) -> TrialState:
    """
    Add one CV repeat at a time to the leader and the families whose verdict
    against it is still undetermined (at most ``top`` families), until nothing
    is undetermined, ``max_repeats`` is reached or the budget would overrun.
    """
    rep = next_repeat
    while rep < max_repeats:
        fams = undetermined(state)
        if not fams:
            break
        fams = fams[:top]
        extra = splitter_folds([rep])
        cost = sum(state.best[f].fit_time_s / max(1, len(state.best[f].fold_ids)) for f in fams) * len(extra)
        if cost > ledger.remaining_s() - reserve_s or ledger.exhausted():
            ledger.record_cut("refinement", f"repeat {rep} for {', '.join(fams)}", "not enough time left")
            break
        for f in fams:
            r0 = state.best[f]
            r = evaluate_family(table, types, y, extra, get_family(f), r0.config, task=task, n_classes=n_classes,
                                metric=metric, secondary=secondary, seed=seed, stage="trial", drop=drop,
                                budget=ledger)
            merged = _merge(r0, r, float(np.mean([len(tr) for _, tr, _ in extra])),
                            float(np.mean([len(te) for _, _, te in extra])))
            state.results = [merged if x is r0 else x for x in state.results]
            state.best[f] = merged
            if f not in state.refined:
                state.refined.append(f)
        rep += 1
    return state
