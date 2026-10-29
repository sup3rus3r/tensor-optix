import pytest

from tensor_optix.ledger.records import EvaluationStatus, ExperimentState
from tensor_optix.ledger.store import ExperimentStore
from tensor_optix.nanonets.features import TASK_FEATURES_SCHEMA_VERSION, extract_task_features


def _seed_task(store, task_env, algorithm, scores, generalization_gaps=None):
    generalization_gaps = generalization_gaps or [None] * len(scores)
    for i, (score, gap) in enumerate(zip(scores, generalization_gaps)):
        experiment_id = f"{task_env}-{algorithm}-{i}"
        store.create_experiment(
            experiment_id=experiment_id, task_env=task_env, algorithm=algorithm,
            framework="torch", seed=i, checkpoint_dir="",
        )
        store.start_experiment(experiment_id)
        store.record_progress(
            experiment_id, best_primary_score=score,
            generalization_gap=gap,
        )
        store.finalize_experiment(
            experiment_id, state=ExperimentState.COMPLETED,
            evaluation_status=EvaluationStatus.PRELIMINARY,
        )


def test_task_features_computes_from_ledger_history(tmp_path):
    store = ExperimentStore(str(tmp_path / "ledger.sqlite3"))
    _seed_task(store, "CartPole-v1", "DQN", [100.0, 200.0, 150.0], [0.1, 0.2, 0.15])

    features = extract_task_features(store, task_env="CartPole-v1")
    assert features.task_env == "CartPole-v1"
    assert features.algorithm == "DQN"
    assert features.n_completed == 3
    assert features.score_mean == 150.0
    assert features.score_std is not None
    assert features.generalization_gap_mean == pytest.approx(0.15)
    assert features.schema_version == TASK_FEATURES_SCHEMA_VERSION
    store.close()


def test_task_features_tolerates_missing_final_metrics_keys(tmp_path):
    store = ExperimentStore(str(tmp_path / "ledger.sqlite3"))
    _seed_task(store, "CartPole-v1", "DQN", [100.0])  # no generalization_gap supplied

    features = extract_task_features(store, task_env="CartPole-v1")
    assert features.generalization_gap_mean is None
    store.close()


def test_task_features_empty_task_returns_zero_state(tmp_path):
    store = ExperimentStore(str(tmp_path / "ledger.sqlite3"))
    features = extract_task_features(store, task_env="NeverSeen-v1")
    assert features.n_completed == 0
    assert features.score_mean is None
    assert features.algorithm == ""
    store.close()


def test_task_features_reports_majority_algorithm(tmp_path):
    store = ExperimentStore(str(tmp_path / "ledger.sqlite3"))
    _seed_task(store, "CartPole-v1", "DQN", [100.0, 110.0])
    _seed_task(store, "CartPole-v1", "PPO", [90.0])

    features = extract_task_features(store, task_env="CartPole-v1")
    assert features.algorithm == "DQN"
    assert features.n_completed == 3
    store.close()
