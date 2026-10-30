#!/usr/bin/env python3
"""
Proposal section 19 validation -- the live adaptive runtime, real
end-to-end, bounded run.

Treats a tuned CartPole-v1 + TorchPPOAgent configuration as the
"deployed model" stand-in, periodically re-evaluated in place of live
inference traffic. Runs the real observe -> detect -> propose ->
experiment -> evaluate -> record -> recommend loop for a small, fixed
number of cycles. Recommend-only: never calls store.approve_strategy,
never auto-promotes a candidate to become the deployed baseline.

This is a small, deliberately tight-budget probe, following this
project's own established precedent (Phase 4/5 validation scripts) of
proving the mechanism before any larger run.

Usage:
    python benchmarks/validate_live_runtime.py
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

from tensor_optix.adapters.pytorch.torch_evaluator import TorchEvaluator
from tensor_optix.algorithms.torch_ppo import TorchPPOAgent
from tensor_optix.core.types import HyperparamSet
from tensor_optix.ledger import ExperimentStore
from tensor_optix.ledger.records import StrategyStatus
from tensor_optix.pipeline.batch_pipeline import BatchPipeline
from tensor_optix.runtime.baseline import establish_baseline, get_current_baseline_experiment_id
from tensor_optix.runtime.live_runtime import LiveAdaptiveRuntime, RuntimeConfig

TASK_ENV = "CartPole-v1"
STUDY_ID = "live_runtime_cartpole_v1"


def _ppo_agent_factory(env_id: str, params: dict) -> TorchPPOAgent:
    """Module-level, picklable -- required for Windows mp.Process spawn."""
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


def main() -> int:
    print("=" * 72)
    print("  Section 19 validation: Live Adaptive Runtime (recommend-only)")
    print(f"  task_env={TASK_ENV}  study_id={STUDY_ID}")
    print("=" * 72)

    store = ExperimentStore("./benchmarks/live_runtime_ledger.sqlite3")

    baseline_id = get_current_baseline_experiment_id(
        store, task_env=TASK_ENV, algorithm="PPO", study_id=STUDY_ID,
    )
    if baseline_id is None:
        print("\n[establishing deployed baseline -- first run for this study_id]")
        baseline_id = establish_baseline(
            store, agent_factory=make_ppo_agent_factory(TASK_ENV), pipeline_factory=make_pipeline_factory(TASK_ENV),
            evaluator=TorchEvaluator(), task_env=TASK_ENV, algorithm="PPO", study_id=STUDY_ID,
            trial_steps=20_000, seed=0,
        )
        print(f"  baseline_experiment_id={baseline_id}")
    else:
        print(f"\n[reusing existing deployed baseline] baseline_experiment_id={baseline_id}")

    config = RuntimeConfig(
        task_env=TASK_ENV, algorithm="PPO", study_id=STUDY_ID, solve_threshold=475.0,
        eval_episodes=5, cycle_sleep_s=1.0, research_cycle_every=3,
        experiment_budget_trials=2, experiment_trial_steps=3_000, experiment_seeds=(0, 1, 2),
        max_cycles=6, source_task_envs=(),
    )
    runtime = LiveAdaptiveRuntime(
        store, config, agent_factory=make_ppo_agent_factory(TASK_ENV), pipeline_factory=make_pipeline_factory(TASK_ENV),
    )

    print("\n[running bounded live-adaptive loop]")
    observations, recommendations = runtime.run()

    print("\n" + "=" * 72)
    print("  SAFETY CHECK")
    print("=" * 72)
    any_approved = False
    for rec in recommendations:
        entry = store.get_strategy_registry_entry(rec.candidate_id)
        status = entry.status if entry else None
        print(f"  candidate={rec.candidate_id[:8]} op={rec.op_name} registry_status={status}")
        if status == StrategyStatus.APPROVED:
            any_approved = True
    print(f"  any candidate auto-approved: {any_approved} (must be False)")
    assert not any_approved, "SAFETY VIOLATION: a candidate was auto-approved by the runtime loop"

    print("\n" + "=" * 72)
    print("  SUMMARY")
    print("=" * 72)
    print(f"  cycles run: {len(observations)}")
    print(f"  recommendations issued: {len(recommendations)}")
    cleared = [r for r in recommendations if r.automatable_criteria_passed]
    print(f"  recommendations clearing all automatable criteria: {len(cleared)}")
    if cleared:
        print("  At least one candidate cleared automatable criteria -- awaiting")
        print("  separate human review (approve_strategy) before any promotion.")
    else:
        print("  No candidate cleared every automatable criterion at this tight")
        print("  budget -- a documented negative/inconclusive result, consistent")
        print("  with this project's own honest-negative-result precedent.")
    print("=" * 72 + "\n")

    store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
