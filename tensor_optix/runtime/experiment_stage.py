"""Experiment stage: when the Observer signals degradation or a
scheduled research cycle fires, Nanonets ranks a fixed shortlist of
grammar operations (it never invents new ops), and StrategyTrialRunner
executes whichever ops receive nonzero allocation -- identical to
benchmarks/validate_strategy_discovery.py's own pattern, just invoked
from the runtime loop instead of a one-shot script.
"""

from __future__ import annotations

import uuid
from typing import Any, Callable, Dict, List, Tuple

from tensor_optix.ledger.records import EventRecord
from tensor_optix.ledger.store import ExperimentStore
from tensor_optix.nanonets import NanonetsSelector
from tensor_optix.strategy import (
    ExplorationBoundsOp,
    LearningRateTrendOp,
    PromotionPolicy,
    PromotionVerdict,
    StrategyCandidate,
    StrategyOp,
    StrategyTrialRunner,
    compare_against_approved,
    record_candidate_scope,
)

# Fixed, small, PPO-appropriate shortlist -- both already proven
# compatible with TorchPPOAgent via required_capabilities(). Nanonets
# ranks among THESE; it never proposes an 8th op.
SHORTLIST: Dict[str, Callable[[], StrategyOp]] = {
    "lr_trend": lambda: LearningRateTrendOp(
        window=3, increase_factor=1.2, decrease_factor=0.5, min_lr=1e-5, max_lr=5e-3,
    ),
    "exploration_bounds": lambda: ExplorationBoundsOp(
        param_name="entropy_coef", lo=0.0, hi=0.1, step_fn="plateau_increase",
    ),
}


def propose_and_run(
    store: ExperimentStore,
    *,
    agent_factory: Callable[[Any], Any],
    pipeline_factory: Callable[[], Any],
    task_env: str,
    algorithm: str,
    solve_threshold: float,
    source_task_envs: List[str],
    budget: int,
    trial_steps: int,
    seeds: List[int],
    trial_timeout_s: float = 180.0,
) -> List[Tuple[StrategyCandidate, List[Dict[str, Any]]]]:
    """method here is literally op_name -- Nanonets' "method" parameter
    is the config["sampler"]-equality discriminator established in
    Phase 3/4; for this use it is set equal to each shortlist op_name,
    not a new concept."""
    selector = NanonetsSelector(store, source_task_envs=source_task_envs)
    decision = selector.propose(task_env=task_env, methods=list(SHORTLIST), budget=budget)

    outcomes: List[Tuple[StrategyCandidate, List[Dict[str, Any]]]] = []
    for op_name, n_trials in decision.allocations.items():
        if n_trials <= 0:
            continue
        op = SHORTLIST[op_name]()
        candidate = StrategyCandidate.from_op(op)
        store.create_strategy_candidate(
            candidate_id=candidate.candidate_id, op_name=candidate.op_name,
            op_version=candidate.op_version, grammar_schema_version=candidate.grammar_schema_version,
            params=candidate.params, created_by="runtime.experiment_stage",
        )
        runner = StrategyTrialRunner(
            agent_factory=agent_factory, pipeline_factory=pipeline_factory,
            trial_steps=trial_steps, ledger_store=store, task_env=task_env, algorithm=algorithm,
            framework="torch", solve_threshold=solve_threshold, trial_timeout_s=trial_timeout_s,
        )
        results = runner.run_candidate(candidate, seeds=seeds)
        outcomes.append((candidate, results))
    return outcomes


def evaluate_and_recommend(
    store: ExperimentStore,
    candidate: StrategyCandidate,
    trial_results: List[Dict[str, Any]],
    *,
    task_env: str,
    algorithm: str,
    trial_steps: int,
    seeds: List[int],
) -> PromotionVerdict:
    """Records scope, computes a PromotionVerdict via a hardcoded
    approval_mode="human" PromotionPolicy (never "auto" -- the runtime
    must never self-approve), and issues a durable, auditable
    recommendation event. Never calls store.approve_strategy."""
    record_candidate_scope(
        store, candidate, train_task_envs=[task_env], held_out_task_envs=[],
        algorithms=[algorithm], frameworks=["torch"], budget_range_episodes=(trial_steps, trial_steps),
        n_seeds_per_task=len(seeds),
    )
    compare_against_approved(store, candidate.candidate_id, task_env, margin=0.0)

    policy = PromotionPolicy(
        store, margin=0.0, min_seeds=3, min_held_out_tasks=0, max_compute_ratio=3.0,
        approval_mode="human",
    )
    verdict = policy.evaluate(candidate_id=candidate.candidate_id, train_task_envs=[task_env], held_out_task_envs=[])

    latest_experiment_id = trial_results[-1]["candidate_experiment_id"]
    issue_recommendation(store, candidate, verdict, experiment_id=latest_experiment_id)
    return verdict


def issue_recommendation(
    store: ExperimentStore, candidate: StrategyCandidate, verdict: PromotionVerdict, *, experiment_id: str,
) -> None:
    existing = store.get_events(experiment_id)
    next_seq = max((e.sequence for e in existing), default=0) + 1
    store.append_event(
        experiment_id,
        EventRecord(
            event_id=str(uuid.uuid4()), experiment_id=experiment_id, sequence=next_seq,
            event_type="recommendation_issued", source="live_runtime",
            payload={
                "candidate_id": candidate.candidate_id, "op_name": candidate.op_name,
                "criteria": verdict.criteria, "rationale": verdict.rationale,
            },
        ),
    )
