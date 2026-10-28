#!/usr/bin/env python3
"""
Phase 2 validation — real concurrent trial execution, CartPole DQN.

Runs ConcurrentTrialScheduler with n_workers=2 against the same CartPole
DQN setup already proven in benchmarks/compare_selection_baselines.py
(Phase 3). Proves genuine process-level concurrency with a deterministic,
non-timing-ratio check: real started_at/finished_at wall-clock intervals
from worker result payloads must show at least one pair of trials that
genuinely overlapped. A speed ratio between sequential full runs is never
asserted here — see tests/test_improvements/test_11_distributed.py's
known-flaky test_four_actors_faster_than_one for the confirmed anti-pattern
this script deliberately avoids.

Also demonstrates the timeout/cancellation path: one trial is given an
artificially short trial_timeout_s so it lands in the ledger as TIMED_OUT,
proving failure classification end to end, not just the happy path.

Usage:
    python benchmarks/run_concurrent_scheduler_demo.py
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import gymnasium as gym
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from tensor_optix.algorithms.torch_dqn import TorchDQNAgent
from tensor_optix.core.types import HyperparamSet
from tensor_optix.ledger import ExperimentStore
from tensor_optix.pipeline.batch_pipeline import BatchPipeline
from tensor_optix.scheduler import ConcurrentTrialScheduler

N_TRIALS = 6
TRIAL_STEPS = 3_000
N_WORKERS = 2

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


def main() -> int:
    print("=" * 72)
    print("  Phase 2 validation: ConcurrentTrialScheduler on CartPole-v1 (DQN)")
    print(f"  n_trials={N_TRIALS}  trial_steps={TRIAL_STEPS}  n_workers={N_WORKERS}")
    print("=" * 72)

    store = ExperimentStore("./benchmarks/experiment_ledger.sqlite3")

    results = []
    scheduler = ConcurrentTrialScheduler(
        agent_factory=agent_factory,
        pipeline_factory=pipeline_factory,
        param_space=PARAM_SPACE,
        n_trials=N_TRIALS,
        trial_steps=TRIAL_STEPS,
        n_workers=N_WORKERS,
        ledger_store=store,
        task_env="CartPole-v1",
        algorithm="DQN",
        framework="torch",
        study_name="concurrent_scheduler_demo",
    )

    original_handle = scheduler._handle_result

    def _capturing_handle(ctx, trial_number, active, result):
        results.append(dict(result))
        return original_handle(ctx, trial_number, active, result)

    scheduler._handle_result = _capturing_handle

    best_params, best_score = scheduler.run()

    print(f"\n  Best params: {best_params}")
    print(f"  Best score:  {best_score:.2f}\n")

    print("=" * 72)
    print("  PER-TRIAL RESULTS (real worker-reported wall-clock)")
    print("=" * 72)
    for r in sorted(results, key=lambda r: r["trial_number"]):
        duration = r["finished_at"] - r["started_at"]
        print(
            f"  trial={r['trial_number']}  pid={r['pid']}  status={r['status']:<10}"
            f"  score={r['score']!s:<10}  duration={duration:6.1f}s"
        )

    def overlaps(a, b):
        return a[0] < b[1] and b[0] < a[1]

    intervals = [(r["started_at"], r["finished_at"]) for r in results]
    any_overlap = any(
        overlaps(intervals[i], intervals[j])
        for i in range(len(intervals))
        for j in range(i + 1, len(intervals))
    )

    print("\n" + "=" * 72)
    print("  CONCURRENCY PROOF (structural, not timing-ratio)")
    print("=" * 72)
    if any_overlap:
        print("  PASS: at least one pair of trials had overlapping [started_at, finished_at] "
              "intervals — trials genuinely ran in parallel.")
    else:
        print("  FAIL: No concurrent execution observed — scheduler ran trials serially.")
        store.close()
        return 1

    print("\n" + "=" * 72)
    print("  TIMEOUT/CANCELLATION DEMO (separate, short run)")
    print("=" * 72)
    timeout_store = ExperimentStore("./benchmarks/experiment_ledger.sqlite3")
    timeout_scheduler = ConcurrentTrialScheduler(
        agent_factory=agent_factory,
        pipeline_factory=pipeline_factory,
        param_space=PARAM_SPACE,
        n_trials=1,
        trial_steps=TRIAL_STEPS,
        n_workers=1,
        trial_timeout_s=1.0,  # deliberately too short to finish -> TIMED_OUT
        ledger_store=timeout_store,
        task_env="CartPole-v1",
        algorithm="DQN",
        framework="torch",
        study_name="concurrent_scheduler_demo_timeout",
    )
    timeout_scheduler.run()
    timeout_records = timeout_store.query_experiments(study_id="concurrent_scheduler_demo_timeout", limit=5)
    for r in timeout_records:
        print(f"  experiment_id={r.experiment_id}  state={r.state.value}  "
              f"evaluation_status={r.evaluation_status.value}")
    timeout_store.close()

    print("\n" + "=" * 72)
    print("  LEDGER QUERY (real rows written by the concurrent run)")
    print("=" * 72)
    records = store.query_experiments(study_id="concurrent_scheduler_demo", limit=20)
    for r in records:
        print(f"  {r.experiment_id}  state={r.state.value:<10}  "
              f"eval={r.evaluation_status.value:<12}  score={r.best_primary_score}  "
              f"ckpt_dir={r.checkpoint_dir}")
    print("=" * 72 + "\n")

    store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
