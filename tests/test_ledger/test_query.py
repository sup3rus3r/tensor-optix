from tensor_optix.ledger.records import ExperimentState, EvaluationStatus
from tensor_optix.ledger.store import ExperimentStore


def _seed(store, tmp_path):
    specs = [
        dict(task_env="CartPole-v1", algorithm="DQN", framework="torch", seed=0),
        dict(task_env="CartPole-v1", algorithm="PPO", framework="torch", seed=1),
        dict(task_env="Acrobot-v1", algorithm="PPO", framework="torch", seed=0),
        dict(task_env="Acrobot-v1", algorithm="PPO", framework="tf", seed=1),
    ]
    records = []
    for spec in specs:
        r = store.create_experiment(checkpoint_dir=str(tmp_path / "ckpt"), **spec)
        records.append(r)
    return records


def test_query_by_task_env(tmp_path):
    store = ExperimentStore(str(tmp_path / "ledger.sqlite3"))
    records = _seed(store, tmp_path)
    results = store.query_experiments(task_env="CartPole-v1")
    assert {r.experiment_id for r in results} == {records[0].experiment_id, records[1].experiment_id}
    store.close()


def test_query_by_algorithm(tmp_path):
    store = ExperimentStore(str(tmp_path / "ledger.sqlite3"))
    records = _seed(store, tmp_path)
    results = store.query_experiments(algorithm="DQN")
    assert {r.experiment_id for r in results} == {records[0].experiment_id}
    store.close()


def test_query_by_state(tmp_path):
    store = ExperimentStore(str(tmp_path / "ledger.sqlite3"))
    records = _seed(store, tmp_path)
    store.finalize_experiment(records[0].experiment_id, state=ExperimentState.COMPLETED)
    store.finalize_experiment(records[1].experiment_id, state=ExperimentState.FAILED)

    completed = store.query_experiments(state=ExperimentState.COMPLETED)
    failed = store.query_experiments(state=ExperimentState.FAILED)
    assert {r.experiment_id for r in completed} == {records[0].experiment_id}
    assert {r.experiment_id for r in failed} == {records[1].experiment_id}
    store.close()


def test_query_by_evaluation_status(tmp_path):
    store = ExperimentStore(str(tmp_path / "ledger.sqlite3"))
    records = _seed(store, tmp_path)
    store.finalize_experiment(
        records[0].experiment_id, state=ExperimentState.COMPLETED,
        evaluation_status=EvaluationStatus.VALIDATED,
    )
    results = store.query_experiments(evaluation_status=EvaluationStatus.VALIDATED)
    assert {r.experiment_id for r in results} == {records[0].experiment_id}
    store.close()


def test_query_by_created_at_range(tmp_path):
    store = ExperimentStore(str(tmp_path / "ledger.sqlite3"))
    records = _seed(store, tmp_path)
    all_times = [r.created_at for r in records]
    results = store.query_experiments(created_after=min(all_times), created_before=max(all_times))
    assert len(results) == len(records)
    store.close()


def test_query_combined_filters_are_and(tmp_path):
    store = ExperimentStore(str(tmp_path / "ledger.sqlite3"))
    records = _seed(store, tmp_path)
    results = store.query_experiments(task_env="Acrobot-v1", framework="tf")
    assert {r.experiment_id for r in results} == {records[3].experiment_id}
    store.close()
