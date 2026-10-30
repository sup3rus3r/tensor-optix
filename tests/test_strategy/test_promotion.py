import time

from tensor_optix.ledger import ExperimentState, ExperimentStore
from tensor_optix.strategy.promotion import PromotionPolicy


def _seed_full_evaluation(store, candidate_id, task_env, delta, *, wallclock_ratio=1.0, seed=0):
    cand_id = f"{candidate_id}-cand-{task_env}-{seed}"
    base_id = f"{candidate_id}-base-{task_env}-{seed}"
    now = time.time()

    store.create_experiment(experiment_id=cand_id, task_env=task_env, algorithm="PPO", framework="torch", seed=seed, checkpoint_dir="", config={})
    store.start_experiment(cand_id)
    store.record_progress(cand_id, best_primary_score=100.0 + delta)
    store.finalize_experiment(cand_id, state=ExperimentState.COMPLETED)

    store.create_experiment(experiment_id=base_id, task_env=task_env, algorithm="PPO", framework="torch", seed=seed, checkpoint_dir="", config={})
    store.start_experiment(base_id)
    store.record_progress(base_id, best_primary_score=100.0)
    store.finalize_experiment(base_id, state=ExperimentState.COMPLETED)

    store.record_strategy_evaluation(
        candidate_id=candidate_id, experiment_id=cand_id, baseline_experiment_id=base_id,
        task_env=task_env, algorithm="PPO", framework="torch", seed=seed, outcome_delta=delta,
    )

    # Backfill started_at/finished_at directly (finalize_experiment only
    # sets finished_at; started_at comes from start_experiment -- both
    # already real timestamps from the calls above, no manual override
    # needed unless we want a specific ratio).
    if wallclock_ratio != 1.0:
        with store._conn:
            store._conn.execute(
                "UPDATE experiments SET started_at = ?, finished_at = ? WHERE experiment_id = ?",
                (now, now + 10.0 * wallclock_ratio, cand_id),
            )
            store._conn.execute(
                "UPDATE experiments SET started_at = ?, finished_at = ? WHERE experiment_id = ?",
                (now, now + 10.0, base_id),
            )


def test_promotion_fails_without_enough_seeds(tmp_path):
    store = ExperimentStore(str(tmp_path / "ledger.sqlite3"))
    store.create_strategy_candidate(candidate_id="c1", op_name="lr_trend", op_version=1, grammar_schema_version=1)
    _seed_full_evaluation(store, "c1", "CartPole-v1", delta=20.0, seed=0)

    policy = PromotionPolicy(store, margin=5.0, min_seeds=3)
    verdict = policy.evaluate(candidate_id="c1", train_task_envs=["CartPole-v1"], held_out_task_envs=[])

    assert verdict.approved is False
    assert verdict.criteria["repeats_across_seeds"] is False
    store.close()


def test_promotion_fails_without_held_out_generalization(tmp_path):
    store = ExperimentStore(str(tmp_path / "ledger.sqlite3"))
    store.create_strategy_candidate(candidate_id="c1", op_name="lr_trend", op_version=1, grammar_schema_version=1)
    for seed in range(3):
        _seed_full_evaluation(store, "c1", "CartPole-v1", delta=20.0, seed=seed)

    policy = PromotionPolicy(store, margin=5.0, min_seeds=3, min_held_out_tasks=1)
    verdict = policy.evaluate(candidate_id="c1", train_task_envs=["CartPole-v1"], held_out_task_envs=[])

    assert verdict.approved is False
    assert verdict.criteria["generalizes_held_out"] is False
    store.close()


def test_promotion_fails_on_excessive_compute_ratio(tmp_path):
    store = ExperimentStore(str(tmp_path / "ledger.sqlite3"))
    store.create_strategy_candidate(candidate_id="c1", op_name="lr_trend", op_version=1, grammar_schema_version=1)
    for seed in range(3):
        _seed_full_evaluation(store, "c1", "CartPole-v1", delta=20.0, wallclock_ratio=3.0, seed=seed)
    for seed in range(3):
        _seed_full_evaluation(store, "c1", "Acrobot-v1", delta=20.0, seed=seed)

    policy = PromotionPolicy(store, margin=5.0, min_seeds=3, min_held_out_tasks=1, max_compute_ratio=1.5)
    verdict = policy.evaluate(candidate_id="c1", train_task_envs=["CartPole-v1"], held_out_task_envs=["Acrobot-v1"])

    assert verdict.criteria["meets_compute_constraints"] is False
    assert verdict.approved is False
    store.close()


def test_promotion_requires_explicit_human_approval_even_when_automatable_criteria_pass(tmp_path):
    """The single most important check: PromotionPolicy.evaluate() must
    NEVER set human_reviewed=True on its own, no matter how good the
    other 5 criteria look."""
    store = ExperimentStore(str(tmp_path / "ledger.sqlite3"))
    store.create_strategy_candidate(candidate_id="c1", op_name="lr_trend", op_version=1, grammar_schema_version=1)
    for seed in range(3):
        _seed_full_evaluation(store, "c1", "CartPole-v1", delta=20.0, seed=seed)
    for seed in range(3):
        _seed_full_evaluation(store, "c1", "Acrobot-v1", delta=20.0, seed=seed)

    policy = PromotionPolicy(store, margin=5.0, min_seeds=3, min_held_out_tasks=1, max_compute_ratio=10.0)
    verdict = policy.evaluate(candidate_id="c1", train_task_envs=["CartPole-v1"], held_out_task_envs=["Acrobot-v1"])

    assert verdict.criteria["beats_baseline_margin"] is True
    assert verdict.criteria["repeats_across_seeds"] is True
    assert verdict.criteria["generalizes_held_out"] is True
    assert verdict.criteria["meets_compute_constraints"] is True
    assert verdict.criteria["reproducible"] is True
    assert verdict.criteria["human_reviewed"] is False  # still false -- no human approved it
    assert verdict.approved is False  # so overall approval must be False too

    # Now a human approves it explicitly.
    store.approve_strategy("c1", reviewer="a_real_reviewer", scope={"task_envs": ["CartPole-v1"]})
    verdict2 = policy.evaluate(candidate_id="c1", train_task_envs=["CartPole-v1"], held_out_task_envs=["Acrobot-v1"])
    assert verdict2.criteria["human_reviewed"] is True
    assert verdict2.approved is True
    store.close()


def test_promotion_fails_when_a_task_has_zero_evaluations(tmp_path):
    store = ExperimentStore(str(tmp_path / "ledger.sqlite3"))
    store.create_strategy_candidate(candidate_id="c1", op_name="lr_trend", op_version=1, grammar_schema_version=1)
    for seed in range(3):
        _seed_full_evaluation(store, "c1", "CartPole-v1", delta=20.0, seed=seed)

    policy = PromotionPolicy(store, margin=5.0, min_seeds=3)
    verdict = policy.evaluate(candidate_id="c1", train_task_envs=["CartPole-v1", "Pendulum-v1"], held_out_task_envs=[])

    assert verdict.criteria["beats_baseline_margin"] is False  # Pendulum-v1 has no evaluations
    assert verdict.approved is False
    store.close()
