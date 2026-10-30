#!/usr/bin/env python3
"""
Phase 5 validation -- real end-to-end strategy discovery run.

Evaluates LearningRateTrendOp (the most universal, mechanically
simplest grammar operation -- see tensor_optix/strategy/grammar.py)
against real PPO training on CartPole-v1 (train) and Acrobot-v1
(held-out), producing an actual PromotionVerdict via PromotionPolicy.
Reports honestly whether the candidate clears promotion or not --
either outcome is a legitimate result, per Phase 4's own
documented-negative-result precedent.

This is a small, deliberately tight-budget probe (few steps, few
seeds) to confirm the full pipeline (grammar -> candidate ->
StrategyTrialRunner -> comparator -> PromotionPolicy -> registry)
works end to end on real training, not a statistically powered claim.

Usage:
    python benchmarks/validate_strategy_discovery.py
"""

from __future__ import annotations

import functools
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import gymnasium as gym
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from tensor_optix.algorithms.torch_ppo import TorchPPOAgent
from tensor_optix.core.types import HyperparamSet
from tensor_optix.ledger import ExperimentStore
from tensor_optix.pipeline.batch_pipeline import BatchPipeline
from tensor_optix.strategy import (
    LearningRateTrendOp,
    PromotionPolicy,
    StrategyCandidate,
    StrategyTrialRunner,
    record_candidate_scope,
)

TRIAL_STEPS = 3_000
SEEDS = [0, 1, 2]

TASKS = {
    "CartPole-v1": {"solve_threshold": 475.0},
    "Acrobot-v1": {"solve_threshold": -100.0},
}
TRAIN_TASK = "CartPole-v1"
HELD_OUT_TASK = "Acrobot-v1"


def _ppo_agent_factory(env_id: str, params: dict) -> TorchPPOAgent:
    """Module-level, picklable -- required as an mp.Process kwarg under
    Windows spawn (local closures can't be pickled)."""
    env = gym.make(env_id)
    obs_dim, n_actions = env.observation_space.shape[0], env.action_space.n
    env.close()
    actor = nn.Sequential(nn.Linear(obs_dim, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, n_actions))
    critic = nn.Sequential(nn.Linear(obs_dim, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))
    hp = {
        "learning_rate": 5e-4, "clip_ratio": 0.2, "entropy_coef": 0.01, "vf_coef": 0.5,
        "gamma": 0.99, "gae_lambda": 0.95, "n_epochs": 10, "minibatch_size": 64, "max_grad_norm": 0.5,
    }
    return TorchPPOAgent(
        actor=actor, critic=critic,
        optimizer=torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=hp["learning_rate"]),
        hyperparams=HyperparamSet(params=hp, episode_id=0), device="auto",
    )


def _ppo_pipeline_factory(env_id: str) -> BatchPipeline:
    return BatchPipeline(env=gym.make(env_id), window_size=512)


def make_ppo_agent_factory(env_id: str):
    return functools.partial(_ppo_agent_factory, env_id)


def make_pipeline_factory(env_id: str):
    return functools.partial(_ppo_pipeline_factory, env_id)


def run_task(store: ExperimentStore, candidate: StrategyCandidate, env_id: str) -> list:
    runner = StrategyTrialRunner(
        agent_factory=make_ppo_agent_factory(env_id), pipeline_factory=make_pipeline_factory(env_id),
        trial_steps=TRIAL_STEPS, ledger_store=store, task_env=env_id, algorithm="PPO",
        framework="torch", solve_threshold=TASKS[env_id]["solve_threshold"], trial_timeout_s=180.0,
    )
    results = runner.run_candidate(candidate, seeds=SEEDS)
    print(f"\n[{env_id}] results:")
    for r in results:
        print(f"  seed={r['seed']}  candidate={r['candidate_score']}  baseline={r['baseline_score']}  delta={r['outcome_delta']}")
    return results


def main() -> int:
    print("=" * 72)
    print("  Phase 5 validation: Strategy Discovery end-to-end")
    print(f"  op=LearningRateTrendOp  trial_steps={TRIAL_STEPS}  seeds={SEEDS}")
    print(f"  train_task={TRAIN_TASK}  held_out_task={HELD_OUT_TASK}")
    print("=" * 72)

    store = ExperimentStore("./benchmarks/strategy_ledger.sqlite3")

    op = LearningRateTrendOp(window=3, increase_factor=1.2, decrease_factor=0.5, min_lr=1e-5, max_lr=5e-3)
    candidate = StrategyCandidate.from_op(op)
    store.create_strategy_candidate(
        candidate_id=candidate.candidate_id, op_name=candidate.op_name,
        op_version=candidate.op_version, grammar_schema_version=candidate.grammar_schema_version,
        params=candidate.params, created_by="validate_strategy_discovery.py",
    )
    print(f"\ncandidate_id={candidate.candidate_id}  op={op}")

    run_task(store, candidate, TRAIN_TASK)
    run_task(store, candidate, HELD_OUT_TASK)

    record_candidate_scope(
        store, candidate, train_task_envs=[TRAIN_TASK], held_out_task_envs=[HELD_OUT_TASK],
        algorithms=["PPO"], frameworks=["torch"], budget_range_episodes=(TRIAL_STEPS, TRIAL_STEPS),
        n_seeds_per_task=len(SEEDS),
        confound_notes="ran with RLOptimizer's default optimizer disabled via a no-op BaseOptimizer, "
                        "since LearningRateTrendOp touches learning_rate directly",
    )

    print("\n" + "=" * 72)
    print("  PROMOTION EVALUATION")
    print("=" * 72)
    policy = PromotionPolicy(store, margin=0.0, min_seeds=3, min_held_out_tasks=1, max_compute_ratio=3.0)
    verdict = policy.evaluate(candidate_id=candidate.candidate_id, train_task_envs=[TRAIN_TASK], held_out_task_envs=[HELD_OUT_TASK])

    for name, passed in verdict.criteria.items():
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
    print(f"\n  {verdict.rationale}")

    print("\n" + "=" * 72)
    print("  SUMMARY")
    print("=" * 72)
    if verdict.approved:
        print("  RESULT: candidate cleared every automatable promotion criterion.")
        print("  Awaiting human review (criterion 6) before final approval.")
    else:
        print("  RESULT: candidate did NOT clear promotion at this budget/seed count.")
        print("  This is a documented negative/inconclusive result, not a failure --")
        print("  consistent with Phase 4's own honest-negative-result precedent.")
    print("=" * 72 + "\n")

    store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
