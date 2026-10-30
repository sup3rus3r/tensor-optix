from tensor_optix.ledger import ExperimentState, ExperimentStore
from tensor_optix.strategy.comparator import MIN_EVALUATIONS, compare_candidate


def _seed_evaluations(store, candidate_id, task_env, deltas, algorithm="PPO", framework="torch"):
    for i, delta in enumerate(deltas):
        cand_id = f"{candidate_id}-cand-{i}"
        base_id = f"{candidate_id}-base-{i}"
        for eid in (cand_id, base_id):
            store.create_experiment(
                experiment_id=eid, task_env=task_env, algorithm=algorithm, framework=framework,
                seed=i, checkpoint_dir="", config={},
            )
            store.start_experiment(eid)
            store.finalize_experiment(eid, state=ExperimentState.COMPLETED)
        store.record_strategy_evaluation(
            candidate_id=candidate_id, experiment_id=cand_id, baseline_experiment_id=base_id,
            task_env=task_env, algorithm=algorithm, framework=framework, seed=i, outcome_delta=delta,
        )


def test_compare_candidate_rejects_single_seed_even_with_a_huge_delta(tmp_path):
    """Direct regression test for the proposal's own non-goal: no
    promotion from a single lucky seed. A delta of +1000 at n=1 must
    still fail the gate."""
    store = ExperimentStore(str(tmp_path / "ledger.sqlite3"))
    store.create_strategy_candidate(candidate_id="c1", op_name="lr_trend", op_version=1, grammar_schema_version=1)
    _seed_evaluations(store, "c1", "CartPole-v1", [1000.0])

    comparison = compare_candidate(store, "c1", "CartPole-v1", margin=0.0)
    assert comparison.n_evaluations == 1
    assert comparison.passes_margin_gate is False
    store.close()


def test_compare_candidate_rejects_near_zero_delta_even_with_enough_seeds(tmp_path):
    store = ExperimentStore(str(tmp_path / "ledger.sqlite3"))
    store.create_strategy_candidate(candidate_id="c1", op_name="lr_trend", op_version=1, grammar_schema_version=1)
    _seed_evaluations(store, "c1", "CartPole-v1", [0.1, -0.2, 0.05, -0.1, 0.0])

    comparison = compare_candidate(store, "c1", "CartPole-v1", margin=5.0)
    assert comparison.n_evaluations == MIN_EVALUATIONS + 2
    assert comparison.passes_margin_gate is False
    store.close()


def test_compare_candidate_passes_with_enough_seeds_and_a_real_margin(tmp_path):
    store = ExperimentStore(str(tmp_path / "ledger.sqlite3"))
    store.create_strategy_candidate(candidate_id="c1", op_name="lr_trend", op_version=1, grammar_schema_version=1)
    _seed_evaluations(store, "c1", "CartPole-v1", [50.0, 48.0, 55.0, 52.0])

    comparison = compare_candidate(store, "c1", "CartPole-v1", margin=10.0)
    assert comparison.n_evaluations == 4
    assert comparison.passes_margin_gate is True
    store.close()


def test_compare_candidate_handles_no_evaluations(tmp_path):
    store = ExperimentStore(str(tmp_path / "ledger.sqlite3"))
    comparison = compare_candidate(store, "never-evaluated", "CartPole-v1")
    assert comparison.n_evaluations == 0
    assert comparison.passes_margin_gate is False
    assert comparison.delta_interval is None
    store.close()


def test_compare_candidate_requires_exactly_min_evaluations_not_fewer(tmp_path):
    store = ExperimentStore(str(tmp_path / "ledger.sqlite3"))
    store.create_strategy_candidate(candidate_id="c1", op_name="lr_trend", op_version=1, grammar_schema_version=1)
    _seed_evaluations(store, "c1", "CartPole-v1", [50.0, 50.0])  # only 2, below MIN_EVALUATIONS

    comparison = compare_candidate(store, "c1", "CartPole-v1", margin=1.0)
    assert comparison.n_evaluations == 2
    assert comparison.passes_margin_gate is False
    store.close()
