"""PromotionPolicy -- the proposal's 6-point promotion checklist
(section 10) as real, independently-checkable booleans, not a vibe.

Two approval modes:
- "human" (default): criterion 6 (human_reviewed) is never set by
  evaluate() itself -- only ExperimentStore.approve_strategy(), called
  by a human after inspecting criteria 1-5, can set it. This is the
  literal implementation of "passes any required human review."
- "auto": if criteria 1-5 all pass, evaluate() calls approve_strategy()
  itself with reviewer="auto:PromotionPolicy" -- a real, traceable
  self-approval (never silent) for contexts that explicitly want fully
  automatic promotion. Still never bypasses criteria 1-5 -- it only
  removes the requirement for a separate human action on top of them.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from tensor_optix.ledger.records import StrategyStatus
from tensor_optix.ledger.store import ExperimentStore

from .candidate import StrategyCandidate
from .comparator import compare_candidate


@dataclass
class PromotionVerdict:
    candidate_id: str
    approved: bool
    criteria: Dict[str, bool]
    rationale: str


@dataclass
class StrategyScope:
    """The bounded claim a strategy's promotion actually supports --
    "a strategy can be useful within a narrow task family without being
    globally superior" (proposal sec 10) as a structured, queryable
    fact rather than prose that can drift from reality."""

    task_envs: List[str] = field(default_factory=list)
    held_out_task_envs: List[str] = field(default_factory=list)
    algorithms: List[str] = field(default_factory=list)  # exact strings tested, never "all"
    frameworks: List[str] = field(default_factory=list)
    budget_range_episodes: Tuple[int, int] = (0, 0)
    n_seeds_per_task: int = 0
    excluded_agents: List[str] = field(default_factory=list)
    confound_notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "StrategyScope":
        return StrategyScope(
            task_envs=data.get("task_envs", []),
            held_out_task_envs=data.get("held_out_task_envs", []),
            algorithms=data.get("algorithms", []),
            frameworks=data.get("frameworks", []),
            budget_range_episodes=tuple(data.get("budget_range_episodes", (0, 0))),
            n_seeds_per_task=data.get("n_seeds_per_task", 0),
            excluded_agents=data.get("excluded_agents", []),
            confound_notes=data.get("confound_notes", ""),
        )


def record_candidate_scope(
    store: ExperimentStore, candidate: StrategyCandidate, *,
    train_task_envs: List[str], held_out_task_envs: List[str],
    algorithms: List[str], frameworks: List[str],
    budget_range_episodes: Tuple[int, int], n_seeds_per_task: int,
    excluded_agents: Optional[List[str]] = None, confound_notes: str = "",
) -> StrategyScope:
    """Build and persist a candidate's StrategyScope from what was
    actually run -- the caller (the orchestrating script that calls
    StrategyTrialRunner once per task_env) supplies the real task/seed/
    budget facts observed, never 'all'."""
    scope = StrategyScope(
        task_envs=list(train_task_envs), held_out_task_envs=list(held_out_task_envs),
        algorithms=list(algorithms), frameworks=list(frameworks),
        budget_range_episodes=tuple(budget_range_episodes), n_seeds_per_task=n_seeds_per_task,
        excluded_agents=list(excluded_agents or []), confound_notes=confound_notes,
    )
    store.record_strategy_scope(candidate.candidate_id, scope.to_dict())
    return scope


class PromotionPolicy:
    def __init__(
        self,
        store: ExperimentStore,
        *,
        margin: float = 0.0,
        min_seeds: int = 3,
        min_held_out_tasks: int = 1,
        max_compute_ratio: float = 1.5,
        approval_mode: str = "human",
    ):
        if approval_mode not in ("human", "auto"):
            raise ValueError(f"PromotionPolicy.approval_mode must be 'human' or 'auto', got {approval_mode!r}")
        self._store = store
        self._margin = margin
        self._min_seeds = min_seeds
        self._min_held_out_tasks = min_held_out_tasks
        self._max_compute_ratio = max_compute_ratio
        self._approval_mode = approval_mode

    def evaluate(
        self,
        candidate_id: str,
        *,
        train_task_envs: List[str],
        held_out_task_envs: List[str],
    ) -> PromotionVerdict:
        criteria: Dict[str, bool] = {}

        train_comparisons = {
            t: compare_candidate(self._store, candidate_id, t, margin=self._margin)
            for t in train_task_envs
        }
        criteria["beats_baseline_margin"] = bool(train_comparisons) and all(
            c.passes_margin_gate for c in train_comparisons.values()
        )
        criteria["repeats_across_seeds"] = bool(train_comparisons) and all(
            c.n_evaluations >= self._min_seeds for c in train_comparisons.values()
        )

        held_out_comparisons = {
            t: compare_candidate(self._store, candidate_id, t, margin=self._margin)
            for t in held_out_task_envs
        }
        criteria["generalizes_held_out"] = (
            len(held_out_task_envs) >= self._min_held_out_tasks
            and all(c.passes_margin_gate for c in held_out_comparisons.values())
        )

        criteria["meets_compute_constraints"] = self._check_compute_ratio(candidate_id, train_task_envs)
        criteria["reproducible"] = self._check_reproducibility(candidate_id, train_task_envs)

        automatable_passed = all(criteria.values())
        if self._approval_mode == "auto" and automatable_passed:
            existing_scope = self._store.get_strategy_scope(candidate_id) or {}
            self._store.approve_strategy(
                candidate_id, reviewer="auto:PromotionPolicy", scope=existing_scope,
            )

        existing = self._store.get_strategy_registry_entry(candidate_id)
        criteria["human_reviewed"] = existing is not None and existing.status == StrategyStatus.APPROVED

        approved = all(criteria.values())
        rationale = self._explain(criteria, train_comparisons, held_out_comparisons)
        return PromotionVerdict(candidate_id=candidate_id, approved=approved, criteria=criteria, rationale=rationale)

    def _check_compute_ratio(self, candidate_id: str, task_envs: List[str]) -> bool:
        evals = []
        for t in task_envs:
            evals.extend(self._store.query_strategy_evaluations(candidate_id=candidate_id, task_env=t))
        if not evals:
            return False

        for e in evals:
            cand = self._store.get_experiment(e.experiment_id)
            base = self._store.get_experiment(e.baseline_experiment_id) if e.baseline_experiment_id else None
            if cand is None or base is None:
                return False
            if cand.started_at is None or cand.finished_at is None or base.started_at is None or base.finished_at is None:
                return False
            cand_wallclock = cand.finished_at - cand.started_at
            base_wallclock = base.finished_at - base.started_at
            if base_wallclock <= 0:
                continue
            if (cand_wallclock / base_wallclock) > self._max_compute_ratio:
                return False
        return True

    def _check_reproducibility(self, candidate_id: str, task_envs: List[str]) -> bool:
        """A real re-run check is a separate, explicit operational step
        (re-executing StrategyTrialRunner and comparing the new score
        against the original run's CI) -- this method checks the
        PRECONDITION for that step: every recorded evaluation has a
        genuinely completed, scored experiment to reproduce from, not a
        skipped/failed run with nothing to re-derive."""
        for t in task_envs:
            evals = self._store.query_strategy_evaluations(candidate_id=candidate_id, task_env=t)
            if not evals:
                return False
            for e in evals:
                cand = self._store.get_experiment(e.experiment_id)
                if cand is None or cand.best_primary_score is None:
                    return False
        return True

    def _explain(self, criteria, train_comparisons, held_out_comparisons) -> str:
        failed = [name for name, passed in criteria.items() if not passed]
        if not failed:
            return "All promotion criteria satisfied."
        return f"Failed criteria: {', '.join(failed)}."
