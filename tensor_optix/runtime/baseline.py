"""Establishes and looks up the "deployed baseline" -- the already-
trained agent the live adaptive runtime (proposal sec 19) treats as a
stand-in for a live, approved, deployed model.

This is a plain ExperimentRecord with a reserved config["role"] marker,
not a new ledger concept -- "which agent is currently deployed" is a
different entity from a strategy candidate's approval state
(strategy_registry), so it is deliberately NOT stored there.
"""

from __future__ import annotations

import os
import tempfile
from typing import Any, Callable, Optional

from tensor_optix.core.checkpoint_registry import CheckpointRegistry
from tensor_optix.core.types import EvalMetrics, HyperparamSet
from tensor_optix.ledger.records import EvaluationStatus, ExperimentState
from tensor_optix.ledger.store import ExperimentStore

ROLE_BASELINE = "deployed_baseline"
ROLE_BASELINE_EVAL = "deployed_baseline_eval"


def establish_baseline(
    store: ExperimentStore,
    *,
    agent_factory: Callable[[dict], Any],
    pipeline_factory: Callable[[], Any],
    evaluator,
    task_env: str,
    algorithm: str,
    study_id: str,
    trial_steps: int,
    seed: int = 0,
    checkpoint_dir: Optional[str] = None,
) -> str:
    """Train a real agent directly (no LoopController/strategy-op --
    mirrors validate_nanonets_loto.py's run_task_direct pattern) and
    record it as the current deployed_baseline. Returns the new
    experiment_id."""
    import torch

    torch.manual_seed(seed)

    agent = agent_factory({})
    pipeline = pipeline_factory()
    if hasattr(pipeline, "set_agent"):
        pipeline.set_agent(agent)
    pipeline.setup()

    ckpt_dir = checkpoint_dir or tempfile.mkdtemp(prefix="to_runtime_baseline_")
    os.makedirs(ckpt_dir, exist_ok=True)
    registry = CheckpointRegistry(ckpt_dir)

    # Train first, into a checkpoint registry that is not yet tied to any
    # ledger row -- create_experiment's config is write-once at creation,
    # so the real weights_path must be known before that call, not
    # patched in afterward.
    steps_per_episode = getattr(pipeline, "steps_per_episode", None) or getattr(pipeline, "n_steps", None)
    max_windows = max(1, trial_steps // steps_per_episode) if steps_per_episode else trial_steps

    best_score = float("-inf")
    best_snapshot = None
    for i, episode_data in enumerate(pipeline.episodes()):
        agent.learn(episode_data)
        eval_metrics = evaluator.score(episode_data, {})
        if eval_metrics.primary_score > best_score:
            best_score = eval_metrics.primary_score
            best_snapshot = registry.save(
                agent, eval_metrics=eval_metrics, hyperparams=agent.get_hyperparams(),
            )
        if i + 1 >= max_windows:
            break
    pipeline.teardown()

    weights_path = best_snapshot.weights_path if best_snapshot is not None else None

    experiment_id = f"{study_id}_baseline_seed{seed}"
    record = store.create_experiment(
        experiment_id=experiment_id, task_env=task_env, algorithm=algorithm,
        framework="torch", seed=seed, checkpoint_dir=ckpt_dir, study_id=study_id,
        config={"role": ROLE_BASELINE, "weights_path": weights_path},
    )
    experiment_id = record.experiment_id
    store.start_experiment(experiment_id)
    store.record_progress(experiment_id, best_primary_score=best_score)
    store.finalize_experiment(
        experiment_id, state=ExperimentState.COMPLETED, evaluation_status=EvaluationStatus.PRELIMINARY,
        final_metrics={"best_score": best_score},
    )

    return experiment_id


def get_current_baseline_experiment_id(
    store: ExperimentStore, *, task_env: str, algorithm: str, study_id: str,
) -> Optional[str]:
    """Live lookup, never cached -- mirrors demote_strategy's own
    'filter live on every read' discipline, since the deployed baseline
    can change at any time via promote.py's separate human-invoked path."""
    records = store.query_experiments(task_env=task_env, algorithm=algorithm, study_id=study_id, limit=100)
    baseline_records = [r for r in records if r.config.get("role") == ROLE_BASELINE]
    if not baseline_records:
        return None
    baseline_records.sort(key=lambda r: r.created_at, reverse=True)
    return baseline_records[0].experiment_id


def get_baseline_weights_path(store: ExperimentStore, experiment_id: str) -> Optional[str]:
    record = store.get_experiment(experiment_id)
    if record is None:
        return None
    return record.config.get("weights_path")


def run_reevaluation(
    store: ExperimentStore,
    *,
    baseline_experiment_id: str,
    agent_factory: Callable[[dict], Any],
    evaluator,
    task_env: str,
    algorithm: str,
    study_id: str,
    eval_episodes: int = 5,
) -> float:
    """Load the deployed baseline's weights into a fresh agent and run a
    real, inference-only rollout (no learn() calls) -- standing in for
    live inference feedback. Appends one deployed_baseline_eval
    ExperimentRecord and returns the real score."""
    weights_path = get_baseline_weights_path(store, baseline_experiment_id)
    if weights_path is None:
        raise ValueError(f"No weights_path recorded for baseline experiment {baseline_experiment_id!r}")

    agent = agent_factory({})
    agent.load_weights(weights_path)
    # A freshly built epsilon-greedy agent starts at its initial exploration
    # rate (typically 1.0 = fully random); load_weights() does not restore
    # hyperparameters. Evaluate the deployed policy greedily.
    hp = agent.get_hyperparams()
    if "epsilon" in hp.params:
        hp.params["epsilon"] = 0.0
        agent.set_hyperparams(hp)

    import gymnasium as gym

    gym_env = gym.make(task_env)
    observations, actions, rewards, terminated_flags, truncated_flags, infos = [], [], [], [], [], []
    for ep in range(eval_episodes):
        obs, _ = gym_env.reset(seed=ep)
        done = False
        while not done:
            action = agent.act(obs)
            next_obs, reward, terminated, truncated, info = gym_env.step(action)
            observations.append(obs)
            actions.append(action)
            rewards.append(reward)
            terminated_flags.append(terminated)
            truncated_flags.append(truncated)
            infos.append(info)
            obs = next_obs
            done = terminated or truncated
    gym_env.close()

    from tensor_optix.core.types import EpisodeData

    episode_data = EpisodeData(
        observations=observations, actions=actions, rewards=rewards,
        terminated=terminated_flags, truncated=truncated_flags, infos=infos, episode_id=0,
    )
    # TorchEvaluator._mean_episode_return already averages per-episode
    # return using episode_data.dones (terminated OR truncated) as the
    # boundary -- primary_score is already the real mean return, not a
    # sum needing further division.
    eval_metrics = evaluator.score(episode_data, {})
    score = eval_metrics.primary_score

    eval_experiment_id = f"{study_id}_eval_{baseline_experiment_id}_{int(__import__('time').time() * 1000)}"
    record = store.create_experiment(
        experiment_id=eval_experiment_id, task_env=task_env, algorithm=algorithm,
        framework="torch", seed=0, checkpoint_dir="", study_id=study_id,
        config={"role": ROLE_BASELINE_EVAL, "parent_experiment_id": baseline_experiment_id},
    )
    store.start_experiment(record.experiment_id)
    store.record_progress(record.experiment_id, best_primary_score=score)
    store.finalize_experiment(record.experiment_id, state=ExperimentState.COMPLETED,
                               evaluation_status=EvaluationStatus.PRELIMINARY)
    return score
