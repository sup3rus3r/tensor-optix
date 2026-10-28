import uuid

from tensor_optix.ledger.records import EventRecord
from tensor_optix.ledger.store import ExperimentStore


def _make_experiment(store, tmp_path):
    return store.create_experiment(
        task_env="CartPole-v1", algorithm="DQN", framework="torch", seed=0,
        checkpoint_dir=str(tmp_path / "ckpt"),
    )


def test_append_event_basic(tmp_path):
    store = ExperimentStore(str(tmp_path / "ledger.sqlite3"))
    record = _make_experiment(store, tmp_path)

    inserted = store.append_event(
        record.experiment_id,
        EventRecord(
            event_id=str(uuid.uuid4()), experiment_id=record.experiment_id,
            sequence=1, event_type="episode_end", step=1, payload={"score": 1.0},
        ),
    )
    assert inserted is True
    events = store.get_events(record.experiment_id)
    assert len(events) == 1
    assert events[0].event_type == "episode_end"
    assert events[0].payload == {"score": 1.0}
    store.close()


def test_append_event_duplicate_event_id_is_noop(tmp_path):
    store = ExperimentStore(str(tmp_path / "ledger.sqlite3"))
    record = _make_experiment(store, tmp_path)
    eid = str(uuid.uuid4())

    first = store.append_event(
        record.experiment_id,
        EventRecord(event_id=eid, experiment_id=record.experiment_id, sequence=1,
                    event_type="episode_end", payload={"score": 1.0}),
    )
    second = store.append_event(
        record.experiment_id,
        EventRecord(event_id=eid, experiment_id=record.experiment_id, sequence=2,
                    event_type="episode_end", payload={"score": 999.0}),
    )
    assert first is True
    assert second is False

    events = store.get_events(record.experiment_id)
    assert len(events) == 1
    assert events[0].payload == {"score": 1.0}
    store.close()


def test_append_event_duplicate_sequence_is_noop(tmp_path):
    store = ExperimentStore(str(tmp_path / "ledger.sqlite3"))
    record = _make_experiment(store, tmp_path)

    first = store.append_event(
        record.experiment_id,
        EventRecord(event_id=str(uuid.uuid4()), experiment_id=record.experiment_id,
                    sequence=5, event_type="episode_end"),
    )
    second = store.append_event(
        record.experiment_id,
        EventRecord(event_id=str(uuid.uuid4()), experiment_id=record.experiment_id,
                    sequence=5, event_type="improvement"),
    )
    assert first is True
    assert second is False

    events = store.get_events(record.experiment_id)
    assert len(events) == 1
    assert events[0].event_type == "episode_end"
    store.close()


def test_get_events_preserves_sequence_order(tmp_path):
    store = ExperimentStore(str(tmp_path / "ledger.sqlite3"))
    record = _make_experiment(store, tmp_path)

    for sequence in [3, 1, 2]:
        store.append_event(
            record.experiment_id,
            EventRecord(event_id=str(uuid.uuid4()), experiment_id=record.experiment_id,
                        sequence=sequence, event_type="episode_end", step=sequence),
        )

    events = store.get_events(record.experiment_id)
    assert [e.sequence for e in events] == [1, 2, 3]
    assert [e.step for e in events] == [1, 2, 3]
    store.close()
