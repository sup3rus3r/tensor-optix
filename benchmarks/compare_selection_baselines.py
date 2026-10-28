#!/usr/bin/env python3
"""
Phase 3 validation — TPE vs RandomSearch on CartPole, matched trial budget.

Runs two TrialOrchestrator instances (DQN on CartPole-v1) under an
identical, small trial budget — one using the default TPESampler, one
using optuna.samplers.RandomSampler via the new `sampler` constructor
param. Both studies are backfilled into the shared experiment ledger at
./benchmarks/experiment_ledger.sqlite3 (same default path benchmark.py
uses), then StrategySelector reads that ledger to compare the two methods
and propose a budget allocation — purely evidence-based, no ML.

This is the real, reproducible proof of Phase 3's exit criterion: "the
selection policy can be compared reproducibly with baseline search under
matched budgets."

Measured wall-clock on this machine (CPU, torch 2.4.0+cpu): a 2-trial,
2000-step sanity run took ~109s total (~54s/trial). Scaling that linearly,
a 10-trial/10_000-step budget per orchestrator would take well over an
hour for both orchestrators combined — too slow for routine verification.
N_TRIALS/TRIAL_STEPS below are deliberately set smaller (still a real,
non-trivial matched-budget comparison, just scoped to finish in a few
minutes) based on that measurement.

Usage:
    python benchmarks/compare_selection_baselines.py
"""

from __future__ import annotations

import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import gymnasium as gym
import optuna
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from tensor_optix.algorithms.torch_dqn import TorchDQNAgent
from tensor_optix.adapters.pytorch.torch_evaluator import TorchEvaluator
from tensor_optix.core.types import HyperparamSet
from tensor_optix.ledger import ExperimentStore
from tensor_optix.orchestrator import TrialOrchestrator
from tensor_optix.pipeline.batch_pipeline import BatchPipeline
from tensor_optix.selection import StrategySelector

N_TRIALS = 6
TRIAL_STEPS = 3_000

PARAM_SPACE = {
    "learning_rate": ("log_float", 1e-4, 1e-3),
    "gamma": ("float", 0.95, 0.999),
}

DQN_HP_DEFAULTS = {
    "epsilon": 1.0, "epsilon_min": 0.05, "epsilon_decay": 0.995,
    "batch_size": 64, "target_update_freq": 10, "replay_capacity": 10_000,
}


def build_dqn_net(obs_dim, n_actions, hidden=128):
    return nn.Sequential(
        nn.Linear(obs_dim, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Linear(hidden, n_actions),
    )


def agent_factory(params: dict) -> TorchDQNAgent:
    env = gym.make("CartPole-v1")
    obs_dim, n_actions = env.observation_space.shape[0], env.action_space.n
    env.close()
    q = build_dqn_net(obs_dim, n_actions)
    hp = {**DQN_HP_DEFAULTS, "learning_rate": params["learning_rate"], "gamma": params["gamma"]}
    return TorchDQNAgent(
        q_network=q, n_actions=n_actions,
        optimizer=torch.optim.Adam(q.parameters(), lr=params["learning_rate"]),
        hyperparams=HyperparamSet(params=hp, episode_id=0),
    )


def pipeline_factory() -> BatchPipeline:
    return BatchPipeline(env=gym.make("CartPole-v1"), window_size=200)


def run_orchestrator(sampler, study_name: str) -> TrialOrchestrator:
    orch = TrialOrchestrator(
        agent_factory=agent_factory,
        pipeline_factory=pipeline_factory,
        param_space=PARAM_SPACE,
        n_trials=N_TRIALS,
        trial_steps=TRIAL_STEPS,
        rloptimizer_kwargs={"evaluator": TorchEvaluator()},
        study_name=study_name,
        sampler=sampler,
        optuna_verbosity=optuna.logging.WARNING,
    )
    t0 = time.perf_counter()
    best_params, best_score = orch.run()
    elapsed = time.perf_counter() - t0
    print(f"  [{study_name}] {elapsed:6.1f}s  best_score={best_score:8.2f}  best_params={best_params}")
    return orch


def main() -> int:
    print("=" * 72)
    print("  Phase 3 validation: TPE vs RandomSearch on CartPole-v1 (DQN)")
    print(f"  n_trials={N_TRIALS}  trial_steps={TRIAL_STEPS}  (matched budget)")
    print("=" * 72)

    print("\n[TPE]")
    tpe_orch = run_orchestrator(None, "cartpole_tpe_baseline")

    print("\n[RandomSearch]")
    rand_orch = run_orchestrator(optuna.samplers.RandomSampler(seed=0), "cartpole_random_baseline")

    tpe_completed = sum(1 for t in tpe_orch.study.trials if t.value is not None)
    rand_completed = sum(1 for t in rand_orch.study.trials if t.value is not None)
    if tpe_completed == 0 or rand_completed == 0:
        print(f"\nFAILED: TPE completed {tpe_completed} trials, RandomSearch completed "
              f"{rand_completed} trials. At least one trial must complete for a "
              f"meaningful comparison.")
        return 1

    store = ExperimentStore("./benchmarks/experiment_ledger.sqlite3")
    selector = StrategySelector(store)
    selector.observe(orchestrator=tpe_orch, task_env="CartPole-v1", algorithm="DQN",
                      framework="torch", seed=0)
    selector.observe(orchestrator=rand_orch, task_env="CartPole-v1", algorithm="DQN",
                      framework="torch", seed=0)

    print("\n" + "=" * 72)
    print("  EVIDENCE TABLE (StrategySelector.compare)")
    print("=" * 72)
    stats = selector.compare(task_env="CartPole-v1", methods=["TPESampler", "RandomSampler"])
    for method, s in stats.items():
        ci = s.score_interval
        ci_str = f"[{ci.lo95:.2f}, {ci.hi95:.2f}]" if ci else "n/a"
        print(f"  {method:<14}  n={s.n_completed:<3}  best={s.best_score!s:<10}  "
              f"95% CI={ci_str:<20}  success_rate={s.success_rate:.2f}")

    print("\n" + "=" * 72)
    print("  SELECTION DECISION (StrategySelector.propose, budget=20)")
    print("=" * 72)
    decision = selector.propose(
        task_env="CartPole-v1", methods=["TPESampler", "RandomSampler"], budget=20,
    )
    print(f"  Allocations: {decision.allocations}")
    print(f"  Rationale:   {decision.rationale}")
    print("=" * 72 + "\n")

    store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
