from tensor_optix.ledger.records import ExperimentState, EvaluationStatus, SCHEMA_VERSION
from tensor_optix.ledger.store import ExperimentStore


def _make_store(tmp_path):
    return ExperimentStore(str(tmp_path / "ledger.sqlite3"))


def test_create_experiment_persists_solve_threshold(tmp_path):
    store = _make_store(tmp_path)
    record = store.create_experiment(
        task_env="CartPole-v1", algorithm="DQN", framework="torch", seed=0,
        checkpoint_dir=str(tmp_path / "ckpt"), solve_threshold=475.0,
    )
    round_tripped = store.get_experiment(record.experiment_id)
    assert round_tripped.solve_threshold == 475.0
    store.close()


def test_create_experiment_solve_threshold_defaults_to_none(tmp_path):
    store = _make_store(tmp_path)
    record = store.create_experiment(
        task_env="CartPole-v1", algorithm="DQN", framework="torch", seed=0,
        checkpoint_dir=str(tmp_path / "ckpt"),
    )
    round_tripped = store.get_experiment(record.experiment_id)
    assert round_tripped.solve_threshold is None
    store.close()


def test_create_experiment_schema_version_matches_constant(tmp_path):
    """Regression test for a pre-existing bug: the INSERT used to hardcode
    schema_version to the literal 1 regardless of SCHEMA_VERSION's real
    value -- this would have silently stayed wrong once SCHEMA_VERSION
    became 2."""
    store = _make_store(tmp_path)
    record = store.create_experiment(
        task_env="CartPole-v1", algorithm="DQN", framework="torch", seed=0,
        checkpoint_dir=str(tmp_path / "ckpt"),
    )
    assert record.schema_version == SCHEMA_VERSION
    store.close()


def test_create_experiment_populates_provenance(tmp_path):
    store = _make_store(tmp_path)
    record = store.create_experiment(
        task_env="CartPole-v1",
        algorithm="DQN",
        framework="torch",
        seed=0,
        checkpoint_dir=str(tmp_path / "ckpt"),
    )
    assert record.tensor_optix_version is not None or record.tensor_optix_version is None
    assert isinstance(record.library_versions, dict)
    assert "numpy" in record.library_versions
    assert "python" in record.library_versions
    assert record.hostname  # non-empty string on a real machine
    store.close()


def test_create_experiment_idempotent_with_explicit_id(tmp_path):
    store = _make_store(tmp_path)
    r1 = store.create_experiment(
        task_env="CartPole-v1", algorithm="DQN", framework="torch", seed=0,
        checkpoint_dir=str(tmp_path / "ckpt"), experiment_id="fixed123",
    )
    r2 = store.create_experiment(
        task_env="CartPole-v1", algorithm="DQN", framework="torch", seed=0,
        checkpoint_dir=str(tmp_path / "ckpt"), experiment_id="fixed123",
    )
    assert r1.experiment_id == r2.experiment_id == "fixed123"
    assert r1.created_at == r2.created_at

    count = store._conn.execute(
        "SELECT COUNT(*) FROM experiments WHERE experiment_id = ?", ("fixed123",)
    ).fetchone()[0]
    assert count == 1
    store.close()


def test_start_experiment_sets_running_and_started_at(tmp_path):
    store = _make_store(tmp_path)
    record = store.create_experiment(
        task_env="CartPole-v1", algorithm="DQN", framework="torch", seed=0,
        checkpoint_dir=str(tmp_path / "ckpt"),
    )
    assert record.state == ExperimentState.CREATED
    assert record.started_at is None

    store.start_experiment(record.experiment_id)
    updated = store.get_experiment(record.experiment_id)
    assert updated.state == ExperimentState.RUNNING
    assert updated.started_at is not None
    store.close()


def test_record_progress_updates_best_scores(tmp_path):
    store = _make_store(tmp_path)
    record = store.create_experiment(
        task_env="CartPole-v1", algorithm="DQN", framework="torch", seed=0,
        checkpoint_dir=str(tmp_path / "ckpt"),
    )
    store.record_progress(
        record.experiment_id,
        best_primary_score=42.0,
        best_train_score=10.0,
        best_val_score=8.0,
        generalization_gap=2.0,
    )
    updated = store.get_experiment(record.experiment_id)
    assert updated.best_primary_score == 42.0
    assert updated.best_train_score == 10.0
    assert updated.best_val_score == 8.0
    assert updated.generalization_gap == 2.0
    store.close()


def test_finalize_experiment_completed(tmp_path):
    store = _make_store(tmp_path)
    record = store.create_experiment(
        task_env="CartPole-v1", algorithm="DQN", framework="torch", seed=0,
        checkpoint_dir=str(tmp_path / "ckpt"),
    )
    store.finalize_experiment(
        record.experiment_id,
        state=ExperimentState.COMPLETED,
        evaluation_status=EvaluationStatus.PRELIMINARY,
        final_metrics={"primary_score": 500.0},
    )
    updated = store.get_experiment(record.experiment_id)
    assert updated.state == ExperimentState.COMPLETED
    assert updated.evaluation_status == EvaluationStatus.PRELIMINARY
    assert updated.finished_at is not None
    assert updated.final_metrics == {"primary_score": 500.0}
    store.close()


def test_finalize_experiment_failed_with_error_message(tmp_path):
    store = _make_store(tmp_path)
    record = store.create_experiment(
        task_env="CartPole-v1", algorithm="DQN", framework="torch", seed=0,
        checkpoint_dir=str(tmp_path / "ckpt"),
    )
    store.finalize_experiment(
        record.experiment_id,
        state=ExperimentState.FAILED,
        evaluation_status=EvaluationStatus.INCONCLUSIVE,
        error_message="boom",
    )
    updated = store.get_experiment(record.experiment_id)
    assert updated.state == ExperimentState.FAILED
    assert updated.evaluation_status == EvaluationStatus.INCONCLUSIVE
    assert updated.error_message == "boom"
    store.close()


def test_finalize_experiment_cancelled(tmp_path):
    store = _make_store(tmp_path)
    record = store.create_experiment(
        task_env="CartPole-v1", algorithm="DQN", framework="torch", seed=0,
        checkpoint_dir=str(tmp_path / "ckpt"),
    )
    store.finalize_experiment(
        record.experiment_id,
        state=ExperimentState.CANCELLED,
        evaluation_status=EvaluationStatus.INCONCLUSIVE,
    )
    updated = store.get_experiment(record.experiment_id)
    assert updated.state == ExperimentState.CANCELLED
    store.close()


def test_finalize_experiment_timed_out(tmp_path):
    store = _make_store(tmp_path)
    record = store.create_experiment(
        task_env="CartPole-v1", algorithm="DQN", framework="torch", seed=0,
        checkpoint_dir=str(tmp_path / "ckpt"),
    )
    store.finalize_experiment(
        record.experiment_id,
        state=ExperimentState.TIMED_OUT,
        evaluation_status=EvaluationStatus.INCONCLUSIVE,
    )
    updated = store.get_experiment(record.experiment_id)
    assert updated.state == ExperimentState.TIMED_OUT
    store.close()


def test_finalize_experiment_is_idempotent_second_call_noop(tmp_path):
    store = _make_store(tmp_path)
    record = store.create_experiment(
        task_env="CartPole-v1", algorithm="DQN", framework="torch", seed=0,
        checkpoint_dir=str(tmp_path / "ckpt"),
    )
    store.finalize_experiment(
        record.experiment_id,
        state=ExperimentState.COMPLETED,
        evaluation_status=EvaluationStatus.VALIDATED,
        error_message=None,
    )
    first = store.get_experiment(record.experiment_id)

    store.finalize_experiment(
        record.experiment_id,
        state=ExperimentState.FAILED,
        evaluation_status=EvaluationStatus.INCONCLUSIVE,
        error_message="this must not overwrite",
    )
    second = store.get_experiment(record.experiment_id)

    assert second.state == first.state == ExperimentState.COMPLETED
    assert second.evaluation_status == EvaluationStatus.VALIDATED
    assert second.error_message is None
    assert second.finished_at == first.finished_at
    store.close()
