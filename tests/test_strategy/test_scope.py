from tensor_optix.ledger import ExperimentStore
from tensor_optix.strategy import LearningRateTrendOp, StrategyCandidate, record_candidate_scope


def _candidate(store):
    candidate = StrategyCandidate.from_op(LearningRateTrendOp())
    store.create_strategy_candidate(
        candidate_id=candidate.candidate_id, op_name=candidate.op_name,
        op_version=candidate.op_version, grammar_schema_version=candidate.grammar_schema_version,
        params=candidate.params,
    )
    return candidate


def test_record_candidate_scope_persists_real_observed_facts(tmp_path):
    store = ExperimentStore(str(tmp_path / "ledger.sqlite3"))
    candidate = _candidate(store)

    scope = record_candidate_scope(
        store, candidate, train_task_envs=["CartPole-v1"], held_out_task_envs=["Acrobot-v1"],
        algorithms=["PPO"], frameworks=["torch"], budget_range_episodes=(10, 20), n_seeds_per_task=3,
    )

    assert scope.task_envs == ["CartPole-v1"]
    retrieved = store.get_strategy_scope(candidate.candidate_id)
    assert retrieved["task_envs"] == ["CartPole-v1"]
    assert retrieved["held_out_task_envs"] == ["Acrobot-v1"]
    store.close()


def test_record_candidate_scope_does_not_disturb_existing_approval(tmp_path):
    """Recording a scope update after approval must not silently reset
    the approval status back to proposed."""
    store = ExperimentStore(str(tmp_path / "ledger.sqlite3"))
    candidate = _candidate(store)

    store.approve_strategy(candidate.candidate_id, reviewer="someone", scope={"task_envs": ["CartPole-v1"]})
    record_candidate_scope(
        store, candidate, train_task_envs=["CartPole-v1", "Pendulum-v1"], held_out_task_envs=[],
        algorithms=["PPO", "TD3"], frameworks=["torch"], budget_range_episodes=(10, 50), n_seeds_per_task=5,
    )

    entry = store.get_strategy_registry_entry(candidate.candidate_id)
    assert entry.status.value == "approved"  # unchanged by the scope update
    assert entry.scope["task_envs"] == ["CartPole-v1", "Pendulum-v1"]  # scope itself did update
    store.close()


def test_record_candidate_scope_never_says_all(tmp_path):
    """No code path should default algorithms/task_envs to a blanket
    'all' claim -- the caller must always supply exact observed values."""
    store = ExperimentStore(str(tmp_path / "ledger.sqlite3"))
    candidate = _candidate(store)

    scope = record_candidate_scope(
        store, candidate, train_task_envs=["CartPole-v1"], held_out_task_envs=[],
        algorithms=["PPO"], frameworks=["torch"], budget_range_episodes=(10, 10), n_seeds_per_task=1,
    )
    assert "all" not in scope.algorithms
    assert "all" not in scope.task_envs
    store.close()
