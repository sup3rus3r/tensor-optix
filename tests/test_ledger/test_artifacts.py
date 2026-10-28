from tensor_optix.core.checkpoint_registry import CheckpointRegistry
from tensor_optix.ledger.records import ArtifactRef
from tensor_optix.ledger.store import ExperimentStore


def test_record_and_get_artifacts_against_real_checkpoint_registry(
    tmp_path, dummy_agent, sample_eval_metrics, basic_hyperparams
):
    store = ExperimentStore(str(tmp_path / "ledger.sqlite3"))
    registry = CheckpointRegistry(str(tmp_path / "ckpt"))

    record = store.create_experiment(
        task_env="CartPole-v1", algorithm="DQN", framework="torch", seed=0,
        checkpoint_dir=str(tmp_path / "ckpt"),
    )

    snapshot = registry.save(dummy_agent, sample_eval_metrics, basic_hyperparams)

    store.record_artifact(
        ArtifactRef(
            experiment_id=record.experiment_id,
            snapshot_id=snapshot.snapshot_id,
            checkpoint_dir=str(tmp_path / "ckpt"),
            snapshot_dir=str(tmp_path / "ckpt" / f"snapshot_{snapshot.snapshot_id}"),
            weights_path=snapshot.weights_path,
            primary_score=snapshot.eval_metrics.primary_score,
            episode_id=snapshot.episode_id,
        )
    )

    artifacts = store.get_artifacts(record.experiment_id)
    assert len(artifacts) == 1
    assert artifacts[0].weights_path == registry.best.weights_path
    assert artifacts[0].snapshot_id == snapshot.snapshot_id
    store.close()


def test_get_artifacts_sorted_by_score_desc(tmp_path):
    store = ExperimentStore(str(tmp_path / "ledger.sqlite3"))
    record = store.create_experiment(
        task_env="CartPole-v1", algorithm="DQN", framework="torch", seed=0,
        checkpoint_dir=str(tmp_path / "ckpt"),
    )

    for score, sid in [(1.0, "a"), (5.0, "b"), (3.0, "c")]:
        store.record_artifact(
            ArtifactRef(
                experiment_id=record.experiment_id, snapshot_id=sid,
                checkpoint_dir=str(tmp_path / "ckpt"),
                snapshot_dir=str(tmp_path / "ckpt" / f"snapshot_{sid}"),
                weights_path=str(tmp_path / "ckpt" / f"snapshot_{sid}" / "weights"),
                primary_score=score, episode_id=0,
            )
        )

    artifacts = store.get_artifacts(record.experiment_id)
    assert [a.snapshot_id for a in artifacts] == ["b", "c", "a"]
    store.close()
