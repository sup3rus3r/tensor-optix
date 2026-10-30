from tensor_optix.core.types import LoopState
from tensor_optix.ledger import ExperimentStore
from tensor_optix.runtime.observer import DeploymentObserver


def test_observer_tracks_improving_sequence_as_active():
    store = ExperimentStore(":memory:")
    obs = DeploymentObserver(store, task_env="T", algorithm="A", study_id="s")
    results = [obs.observe(s) for s in [10.0, 20.0, 30.0, 40.0, 50.0]]
    assert all(r.state == LoopState.ACTIVE for r in results)
    assert all(r.degradation_fired is False for r in results)
    store.close()


def test_observer_detects_genuine_drop_as_degradation():
    store = ExperimentStore(":memory:")
    obs = DeploymentObserver(store, task_env="T", algorithm="A", study_id="s")
    for s in [10.0, 20.0, 30.0, 40.0, 50.0, 55.0]:
        obs.observe(s)
    result = obs.observe(1.0)  # huge drop from best
    assert result.degradation_fired is True
    store.close()


def test_observer_flat_sequence_eventually_reaches_non_active_state():
    store = ExperimentStore(":memory:")
    obs = DeploymentObserver(store, task_env="T", algorithm="A", study_id="s")
    result = None
    for _ in range(15):
        result = obs.observe(50.0)
    assert result.state in (LoopState.COOLING, LoopState.DORMANT)
    store.close()


def test_score_history_reads_only_baseline_eval_rows():
    store = ExperimentStore(":memory:")
    store.create_experiment(
        experiment_id="e1", task_env="T", algorithm="A", framework="torch", seed=0,
        checkpoint_dir="", study_id="s", config={"role": "deployed_baseline_eval"},
    )
    store.record_progress("e1", best_primary_score=42.0)
    store.create_experiment(
        experiment_id="e2", task_env="T", algorithm="A", framework="torch", seed=0,
        checkpoint_dir="", study_id="s", config={"role": "deployed_baseline"},  # NOT an eval row
    )
    store.record_progress("e2", best_primary_score=999.0)

    obs = DeploymentObserver(store, task_env="T", algorithm="A", study_id="s")
    history = obs.score_history()
    assert history == [42.0]  # the deployed_baseline row must be excluded
    store.close()


def test_replay_history_reconstructs_state_from_ledger():
    store = ExperimentStore(":memory:")
    for i, score in enumerate([10.0, 20.0, 30.0, 40.0, 50.0]):
        eid = f"e{i}"
        store.create_experiment(
            experiment_id=eid, task_env="T", algorithm="A", framework="torch", seed=0,
            checkpoint_dir="", study_id="s", config={"role": "deployed_baseline_eval"},
        )
        store.record_progress(eid, best_primary_score=score)

    obs = DeploymentObserver(store, task_env="T", algorithm="A", study_id="s")
    obs.replay_history()
    assert obs._cycle == 5
    store.close()
