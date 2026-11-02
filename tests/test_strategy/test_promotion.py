import time

import pytest

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


def test_promotion_policy_rejects_invalid_approval_mode(tmp_path):
    store = ExperimentStore(str(tmp_path / "ledger.sqlite3"))
    try:
        PromotionPolicy(store, approval_mode="bogus")
        assert False, "expected ValueError"
    except ValueError:
        pass
    store.close()


def test_promotion_human_mode_never_self_approves_even_with_everything_else_passing(tmp_path):
    """Default/explicit human mode must never set human_reviewed=True on
    its own, confirming the default is still the safe one."""
    store = ExperimentStore(str(tmp_path / "ledger.sqlite3"))
    store.create_strategy_candidate(candidate_id="c1", op_name="lr_trend", op_version=1, grammar_schema_version=1)
    for seed in range(3):
        _seed_full_evaluation(store, "c1", "CartPole-v1", delta=20.0, seed=seed)
        _seed_full_evaluation(store, "c1", "Acrobot-v1", delta=20.0, seed=seed)

    policy = PromotionPolicy(store, margin=5.0, min_seeds=3, min_held_out_tasks=1,
                              max_compute_ratio=10.0, approval_mode="human")
    verdict = policy.evaluate(candidate_id="c1", train_task_envs=["CartPole-v1"], held_out_task_envs=["Acrobot-v1"])

    assert verdict.criteria["human_reviewed"] is False
    assert verdict.approved is False
    entry = store.get_strategy_registry_entry("c1")
    assert entry is None  # no registry row was created at all
    store.close()


def test_promotion_auto_mode_self_approves_when_all_automatable_criteria_pass(tmp_path):
    store = ExperimentStore(str(tmp_path / "ledger.sqlite3"))
    store.create_strategy_candidate(candidate_id="c1", op_name="lr_trend", op_version=1, grammar_schema_version=1)
    for seed in range(3):
        _seed_full_evaluation(store, "c1", "CartPole-v1", delta=20.0, seed=seed)
        _seed_full_evaluation(store, "c1", "Acrobot-v1", delta=20.0, seed=seed)

    policy = PromotionPolicy(store, margin=5.0, min_seeds=3, min_held_out_tasks=1,
                              max_compute_ratio=10.0, approval_mode="auto", acknowledge_auto_approve_bypasses_human_review=True)
    verdict = policy.evaluate(candidate_id="c1", train_task_envs=["CartPole-v1"], held_out_task_envs=["Acrobot-v1"])

    assert verdict.criteria["human_reviewed"] is True
    assert verdict.approved is True

    entry = store.get_strategy_registry_entry("c1")
    assert entry is not None
    assert entry.status.value == "approved"
    assert entry.reviewer == "auto:PromotionPolicy"  # traceable -- never pretends a human looked at it
    store.close()


def test_promotion_auto_mode_does_not_self_approve_when_a_criterion_fails(tmp_path):
    store = ExperimentStore(str(tmp_path / "ledger.sqlite3"))
    store.create_strategy_candidate(candidate_id="c1", op_name="lr_trend", op_version=1, grammar_schema_version=1)
    for seed in range(3):
        _seed_full_evaluation(store, "c1", "CartPole-v1", delta=20.0, seed=seed)
    # No Acrobot-v1 evaluations at all -- generalizes_held_out must fail.

    policy = PromotionPolicy(store, margin=5.0, min_seeds=3, min_held_out_tasks=1, approval_mode="auto", acknowledge_auto_approve_bypasses_human_review=True)
    verdict = policy.evaluate(candidate_id="c1", train_task_envs=["CartPole-v1"], held_out_task_envs=["Acrobot-v1"])

    assert verdict.approved is False
    entry = store.get_strategy_registry_entry("c1")
    assert entry is None  # never self-approved since a real criterion failed
    store.close()


def test_promotion_auto_mode_preserves_recorded_scope_when_self_approving(tmp_path):
    from tensor_optix.strategy.promotion import record_candidate_scope
    from tensor_optix.strategy.candidate import StrategyCandidate

    store = ExperimentStore(str(tmp_path / "ledger.sqlite3"))
    candidate = StrategyCandidate(candidate_id="c1", op_name="lr_trend", op_version=1, params={})
    store.create_strategy_candidate(candidate_id="c1", op_name="lr_trend", op_version=1, grammar_schema_version=1)
    record_candidate_scope(
        store, candidate, train_task_envs=["CartPole-v1"], held_out_task_envs=["Acrobot-v1"],
        algorithms=["PPO"], frameworks=["torch"], budget_range_episodes=(10, 10), n_seeds_per_task=3,
    )
    for seed in range(3):
        _seed_full_evaluation(store, "c1", "CartPole-v1", delta=20.0, seed=seed)
        _seed_full_evaluation(store, "c1", "Acrobot-v1", delta=20.0, seed=seed)

    policy = PromotionPolicy(store, margin=5.0, min_seeds=3, min_held_out_tasks=1,
                              max_compute_ratio=10.0, approval_mode="auto", acknowledge_auto_approve_bypasses_human_review=True)
    policy.evaluate(candidate_id="c1", train_task_envs=["CartPole-v1"], held_out_task_envs=["Acrobot-v1"])

    entry = store.get_strategy_registry_entry("c1")
    assert entry.scope["task_envs"] == ["CartPole-v1"]  # scope recorded before approval survives it
    store.close()


def test_auto_mode_without_acknowledgement_is_rejected(tmp_path):
    # approval_mode="auto" alone must never be enough to self-approve --
    # a bare string is too easy to pass by accident. The constructor must
    # refuse until the bypass is explicitly acknowledged.
    store = ExperimentStore(str(tmp_path / "ledger.sqlite3"))
    with pytest.raises(ValueError, match="acknowledge_auto_approve_bypasses_human_review"):
        PromotionPolicy(store, approval_mode="auto")
    store.close()


def test_auto_mode_with_acknowledgement_false_is_still_rejected(tmp_path):
    store = ExperimentStore(str(tmp_path / "ledger.sqlite3"))
    with pytest.raises(ValueError, match="acknowledge_auto_approve_bypasses_human_review"):
        PromotionPolicy(store, approval_mode="auto", acknowledge_auto_approve_bypasses_human_review=False)
    store.close()


def test_human_mode_never_requires_acknowledgement(tmp_path):
    store = ExperimentStore(str(tmp_path / "ledger.sqlite3"))
    PromotionPolicy(store)  # default approval_mode="human" -- must not raise
    PromotionPolicy(store, approval_mode="human")
    store.close()
