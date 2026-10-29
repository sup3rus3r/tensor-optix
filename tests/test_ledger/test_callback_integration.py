import os

from tensor_optix.core.loop_controller import LoopController
from tensor_optix.core.checkpoint_registry import CheckpointRegistry
from tensor_optix.core.backoff_scheduler import BackoffScheduler
from tensor_optix.optimizers.backoff_optimizer import BackoffOptimizer
from tensor_optix.ledger.callback import ExperimentLedgerCallback
from tensor_optix.ledger.records import ExperimentState, EvaluationStatus
from tensor_optix.ledger.store import ExperimentStore


def make_controller(agent, evaluator, pipeline, tmp_path, max_episodes=5, callbacks=None):
    registry = CheckpointRegistry(str(tmp_path / "checkpoints"), max_snapshots=5)
    scheduler = BackoffScheduler(base_interval=1, plateau_threshold=3, dormant_threshold=6)
    optimizer = BackoffOptimizer()
    return LoopController(
        agent=agent,
        evaluator=evaluator,
        optimizer=optimizer,
        pipeline=pipeline,
        checkpoint_registry=registry,
        backoff_scheduler=scheduler,
        max_episodes=max_episodes,
        callbacks=callbacks or [],
    )


def test_ledger_callback_persists_solve_threshold(dummy_agent, dummy_evaluator, dummy_pipeline, tmp_path):
    store = ExperimentStore(str(tmp_path / "ledger.sqlite3"))
    ledger_cb = ExperimentLedgerCallback(
        store, task_env="CartPole-v1", algorithm="DummyAgent", framework="torch",
        seed=0, checkpoint_dir=str(tmp_path / "checkpoints"), solve_threshold=475.0,
    )
    controller = make_controller(dummy_agent, dummy_evaluator, dummy_pipeline, tmp_path, max_episodes=2, callbacks=[ledger_cb])
    controller.run()

    record = store.get_experiment(ledger_cb.experiment_id)
    assert record.solve_threshold == 475.0
    store.close()


def test_ledger_callback_records_full_run(dummy_agent, dummy_evaluator, tmp_path):
    from conftest import DummyPipeline
    # Strictly increasing rewards guarantee on_improvement fires in the main
    # loop (cold start alone never fires it — confirmed in loop_controller.py).
    pipeline = DummyPipeline(rewards_sequence=[[float(i)] * 3 for i in range(1, 20)])

    store = ExperimentStore(str(tmp_path / "ledger.sqlite3"))
    checkpoint_dir = str(tmp_path / "checkpoints")
    ledger_cb = ExperimentLedgerCallback(
        store,
        task_env="CartPole-v1",
        algorithm="DummyAgent",
        framework="torch",
        seed=0,
        checkpoint_dir=checkpoint_dir,
    )

    controller = make_controller(
        dummy_agent, dummy_evaluator, pipeline, tmp_path,
        max_episodes=5, callbacks=[ledger_cb],
    )
    controller.run()

    record = store.get_experiment(ledger_cb.experiment_id)
    assert record is not None
    assert record.state == ExperimentState.COMPLETED
    assert record.evaluation_status == EvaluationStatus.PRELIMINARY
    assert record.started_at is not None
    assert record.finished_at is not None
    assert record.started_at <= record.finished_at

    events = store.get_events(ledger_cb.experiment_id)
    event_types = [e.event_type for e in events]
    assert "loop_start" in event_types
    assert "loop_stop" in event_types
    assert "improvement" in event_types  # guaranteed by the strictly increasing reward sequence

    artifacts = store.get_artifacts(ledger_cb.experiment_id)
    assert len(artifacts) >= 1
    assert os.path.exists(artifacts[0].weights_path)
    store.close()


def test_ledger_records_episode_end_events_in_order(dummy_agent, dummy_evaluator, tmp_path):
    from conftest import DummyPipeline
    pipeline = DummyPipeline(rewards_sequence=[[float(i)] * 3 for i in range(1, 20)])

    store = ExperimentStore(str(tmp_path / "ledger.sqlite3"))
    ledger_cb = ExperimentLedgerCallback(
        store, task_env="CartPole-v1", algorithm="DummyAgent", framework="torch",
        seed=0, checkpoint_dir=str(tmp_path / "checkpoints"),
    )
    controller = make_controller(dummy_agent, dummy_evaluator, pipeline, tmp_path, max_episodes=8, callbacks=[ledger_cb])
    controller.run()

    events = store.get_events(ledger_cb.experiment_id)
    sequences = [e.sequence for e in events]
    assert sequences == sorted(sequences)
    assert len(sequences) == len(set(sequences))

    episode_end_steps = [e.step for e in events if e.event_type == "episode_end"]
    assert episode_end_steps == sorted(episode_end_steps)
    store.close()


def test_ledger_write_failure_does_not_crash_training(dummy_agent, dummy_evaluator, dummy_pipeline, tmp_path):
    store = ExperimentStore(str(tmp_path / "ledger.sqlite3"))

    class BrokenLedgerCallback(ExperimentLedgerCallback):
        def on_episode_end(self, episode_id, eval_metrics):
            raise RuntimeError("simulated ledger write failure")

    broken_cb = BrokenLedgerCallback(
        store, task_env="CartPole-v1", algorithm="DummyAgent", framework="torch",
        seed=0, checkpoint_dir=str(tmp_path / "checkpoints"),
    )
    controller = make_controller(
        dummy_agent, dummy_evaluator, dummy_pipeline, tmp_path,
        max_episodes=5, callbacks=[broken_cb],
    )

    # Should not raise despite every on_episode_end call failing.
    controller.run()

    assert dummy_agent.learn_calls >= 1
    assert controller.best_snapshot is not None
    assert controller.best_snapshot.eval_metrics.primary_score > 0
    store.close()


def test_mark_failed_path_produces_failed_state(tmp_path):
    store = ExperimentStore(str(tmp_path / "ledger.sqlite3"))
    ledger_cb = ExperimentLedgerCallback(
        store, task_env="CartPole-v1", algorithm="DummyAgent", framework="torch",
        seed=0, checkpoint_dir=str(tmp_path / "checkpoints"),
    )
    ledger_cb.on_loop_start()
    ledger_cb.mark_failed(RuntimeError("boom"))

    record = store.get_experiment(ledger_cb.experiment_id)
    assert record.state == ExperimentState.FAILED
    assert record.evaluation_status == EvaluationStatus.INCONCLUSIVE
    assert record.error_message == "boom"

    # on_loop_stop firing afterwards must not overwrite the FAILED state.
    ledger_cb.on_loop_stop()
    record_after = store.get_experiment(ledger_cb.experiment_id)
    assert record_after.state == ExperimentState.FAILED
    assert record_after.error_message == "boom"
    store.close()
