import pytest

from tensor_optix.ledger import ExperimentState, ExperimentStore
from tensor_optix.ledger.records import EvaluationStatus
from tensor_optix.runtime.baseline import ROLE_BASELINE
from tensor_optix.runtime.promote import promote_candidate_to_baseline


def _seed_candidate_with_evaluation(store, candidate_id, task_env, score, weights_path, seed=0):
    store.create_strategy_candidate(candidate_id=candidate_id, op_name="lr_trend", op_version=1, grammar_schema_version=1)
    exp_id = f"{candidate_id}-exp-{seed}"
    store.create_experiment(
        experiment_id=exp_id, task_env=task_env, algorithm="PPO", framework="torch",
        seed=seed, checkpoint_dir="", config={},
    )
    store.start_experiment(exp_id)
    store.record_progress(exp_id, best_primary_score=score)
    store.finalize_experiment(exp_id, state=ExperimentState.COMPLETED,
                               evaluation_status=EvaluationStatus.PRELIMINARY,
                               final_metrics={"weights_path": weights_path})
    store.record_strategy_evaluation(
        candidate_id=candidate_id, experiment_id=exp_id, task_env=task_env,
        algorithm="PPO", framework="torch", seed=seed, outcome_delta=score,
    )
    return exp_id


def test_promote_refuses_unapproved_candidate(tmp_path):
    store = ExperimentStore(str(tmp_path / "ledger.sqlite3"))
    _seed_candidate_with_evaluation(store, "c1", "CartPole-v1", 100.0, "/fake/weights")

    with pytest.raises(ValueError, match="not approved"):
        promote_candidate_to_baseline(
            store, "c1", reviewer="r", task_env="CartPole-v1", algorithm="PPO", study_id="s1",
        )
    store.close()


def test_promote_succeeds_after_approval_and_picks_best_scoring_weights(tmp_path):
    store = ExperimentStore(str(tmp_path / "ledger.sqlite3"))
    _seed_candidate_with_evaluation(store, "c1", "CartPole-v1", 100.0, "/fake/weights/low", seed=0)
    _seed_candidate_with_evaluation(store, "c1", "CartPole-v1", 300.0, "/fake/weights/high", seed=1)
    _seed_candidate_with_evaluation(store, "c1", "CartPole-v1", 200.0, "/fake/weights/mid", seed=2)

    store.approve_strategy("c1", reviewer="r", scope={"task_envs": ["CartPole-v1"]})
    new_id = promote_candidate_to_baseline(
        store, "c1", reviewer="r", task_env="CartPole-v1", algorithm="PPO", study_id="s1",
    )

    record = store.get_experiment(new_id)
    assert record.config["role"] == ROLE_BASELINE
    assert record.config["weights_path"] == "/fake/weights/high"  # best score wins
    assert record.config["promoted_from_candidate_id"] == "c1"
    assert record.state == ExperimentState.COMPLETED
    store.close()


def test_promote_raises_when_no_weights_path_recorded(tmp_path):
    store = ExperimentStore(str(tmp_path / "ledger.sqlite3"))
    store.create_strategy_candidate(candidate_id="c1", op_name="lr_trend", op_version=1, grammar_schema_version=1)
    store.create_experiment(experiment_id="e1", task_env="CartPole-v1", algorithm="PPO", framework="torch", seed=0, checkpoint_dir="")
    store.start_experiment("e1")
    store.record_progress("e1", best_primary_score=100.0)
    store.finalize_experiment("e1", state=ExperimentState.COMPLETED)  # no weights_path in final_metrics
    store.record_strategy_evaluation(candidate_id="c1", experiment_id="e1", task_env="CartPole-v1",
                                       algorithm="PPO", framework="torch", seed=0, outcome_delta=100.0)
    store.approve_strategy("c1", reviewer="r")

    with pytest.raises(ValueError, match="No scored, completed experiment"):
        promote_candidate_to_baseline(store, "c1", reviewer="r", task_env="CartPole-v1", algorithm="PPO", study_id="s1")
    store.close()


def test_promote_actually_changes_the_active_baseline(tmp_path):
    from tensor_optix.runtime.baseline import get_current_baseline_experiment_id

    store = ExperimentStore(str(tmp_path / "ledger.sqlite3"))
    store.create_experiment(
        experiment_id="old_baseline", task_env="CartPole-v1", algorithm="PPO", framework="torch",
        seed=0, checkpoint_dir="", study_id="s1", config={"role": ROLE_BASELINE, "weights_path": "/fake/old"},
    )
    store.start_experiment("old_baseline")
    store.finalize_experiment("old_baseline", state=ExperimentState.COMPLETED)

    before = get_current_baseline_experiment_id(store, task_env="CartPole-v1", algorithm="PPO", study_id="s1")
    assert before == "old_baseline"

    _seed_candidate_with_evaluation(store, "c1", "CartPole-v1", 500.0, "/fake/new")
    store.approve_strategy("c1", reviewer="r")
    new_id = promote_candidate_to_baseline(store, "c1", reviewer="r", task_env="CartPole-v1", algorithm="PPO", study_id="s1")

    after = get_current_baseline_experiment_id(store, task_env="CartPole-v1", algorithm="PPO", study_id="s1")
    assert after == new_id
    assert after != before
    store.close()
