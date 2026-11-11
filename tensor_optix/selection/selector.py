"""Evidence-based experiment selection over ledger history.

StrategySelector reads ExperimentRecord history already written to the
ledger and makes simple, explicit, non-learned decisions about how to
allocate a trial budget across competing methods (e.g. TPE vs random
search). It is NOT a predictive model — propose() does not accept task
features in this phase; that is reserved for the Phase 4+ cross-task
meta-model.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from tensor_optix.ledger.records import ExperimentState
from tensor_optix.ledger.store import ExperimentStore

from .backfill import backfill_study_to_ledger
from .uncertainty import ScoreInterval, score_interval

if TYPE_CHECKING:
    from tensor_optix.orchestrator import TrialOrchestrator


def record_matches_method(record, method: str) -> bool:
    """
    True when a ledger experiment is evidence for `method`.

    Two kinds of experiment carry a method label in their config:
      - trial-search runs backfilled from an Optuna study: config["sampler"]
      - strategy-candidate runs (StrategyTrialRunner): config["op_name"],
        excluding the paired no-strategy baseline run (config["is_baseline"]),
        which is evidence for the baseline, not for the op.
    """
    config = record.config or {}
    if config.get("sampler") == method:
        return True
    return config.get("op_name") == method and not config.get("is_baseline", False)


@dataclass
class MethodStats:
    method: str
    n_experiments: int
    n_completed: int
    best_score: Optional[float]
    score_interval: Optional[ScoreInterval]
    success_rate: float


@dataclass
class SelectionDecision:
    allocations: Dict[str, int]
    rationale: str


class StrategySelector:
    def __init__(self, store: ExperimentStore):
        self._store = store

    def stats_for(self, *, task_env: str, method: str) -> MethodStats:
        """Compute MethodStats for one (task_env, method) pair.

        Filters client-side on record.config["sampler"] rather than
        study_id, because study_id varies per TrialOrchestrator
        instantiation while config["sampler"] is the stable discriminator
        we actually want to compare on across runs.
        """
        records = self._store.query_experiments(task_env=task_env, limit=1000)
        matching = [r for r in records if record_matches_method(r, method)]

        n_experiments = len(matching)
        completed = [r for r in matching if r.state == ExperimentState.COMPLETED]
        n_completed = len(completed)
        scores = [r.best_primary_score for r in completed if r.best_primary_score is not None]

        return MethodStats(
            method=method,
            n_experiments=n_experiments,
            n_completed=n_completed,
            best_score=max(scores) if scores else None,
            score_interval=score_interval(scores) if scores else None,
            success_rate=(n_completed / n_experiments) if n_experiments else 0.0,
        )

    def compare(self, *, task_env: str, methods: List[str]) -> Dict[str, MethodStats]:
        """Return MethodStats for every method — the evidence table used by propose()."""
        return {method: self.stats_for(task_env=task_env, method=method) for method in methods}

    def propose(
        self,
        *,
        task_env: str,
        methods: List[str],
        budget: int,
        task_features: Optional[Dict[str, Any]] = None,
    ) -> SelectionDecision:
        """Allocate `budget` trials across `methods` using ledger evidence only.

        Rule:
        - Any method with zero completed experiments for this task_env gets
          a minimum exploration allocation first (can't compare what hasn't
          been tried).
        - Remaining budget is split by rank order of score_interval.hi95:
          the method with the best historical hi95 gets the majority share.
        """
        if task_features is not None:
            raise NotImplementedError(
                "task_features is reserved for Phase 4+ meta-feature extraction; "
                "pass None in Phase 3"
            )
        if not methods:
            raise ValueError("propose() requires at least one method")

        stats = self.compare(task_env=task_env, methods=methods)
        allocations = {method: 0 for method in methods}
        remaining = budget

        min_exploration = max(1, budget // len(methods)) if budget else 0
        untried = [m for m in methods if stats[m].n_completed == 0]
        for method in untried:
            take = min(min_exploration, remaining)
            allocations[method] += take
            remaining -= take

        tried = [m for m in methods if stats[m].n_completed > 0]
        if tried and remaining > 0:
            ranked = sorted(tried, key=lambda m: stats[m].score_interval.hi95, reverse=True)
            if len(ranked) == 1:
                allocations[ranked[0]] += remaining
            else:
                best = ranked[0]
                majority = int(round(remaining * 0.7))
                allocations[best] += majority
                rest_remaining = remaining - majority
                rest = ranked[1:]
                share = rest_remaining // len(rest)
                leftover = rest_remaining - share * len(rest)
                for i, method in enumerate(rest):
                    allocations[method] += share + (1 if i < leftover else 0)
        elif not tried and remaining > 0:
            # Nothing has history at all beyond the minimum exploration just
            # assigned above; split whatever is left evenly.
            share = remaining // len(methods)
            leftover = remaining - share * len(methods)
            for i, method in enumerate(methods):
                allocations[method] += share + (1 if i < leftover else 0)

        if tried:
            ranked_desc = sorted(tried, key=lambda m: stats[m].score_interval.hi95, reverse=True)
            parts = [
                f"{m} hi95={stats[m].score_interval.hi95:.2f} (n={stats[m].n_completed})"
                for m in ranked_desc
            ]
            rationale = "; ".join(parts)
            if untried:
                rationale += f"; exploring untried method(s): {', '.join(untried)}"
            rationale += f"; allocating {allocations}"
        else:
            rationale = (
                f"No completed experiments for any method on {task_env!r}; "
                f"allocating minimum exploration budget evenly: {allocations}"
            )

        return SelectionDecision(allocations=allocations, rationale=rationale)

    def observe(
        self,
        *,
        orchestrator: "TrialOrchestrator",
        task_env: str,
        algorithm: str,
        framework: str = "torch",
        seed: int = 0,
        solve_threshold: Optional[float] = None,
    ) -> List[Any]:
        """Ingest one completed TrialOrchestrator run into the ledger.

        TrialOrchestrator exposes no per-trial streaming hook into its
        internal _objective, so this operates on a whole completed study,
        not individual trials as they finish.
        """
        return backfill_study_to_ledger(
            orchestrator,
            self._store,
            task_env=task_env,
            algorithm=algorithm,
            framework=framework,
            seed=seed,
            solve_threshold=solve_threshold,
        )
