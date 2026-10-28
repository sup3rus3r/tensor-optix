from tensor_optix.core.types import HyperparamSet
from tensor_optix.ledger.records import ExperimentState
from tensor_optix.ledger.store import ExperimentStore
from tensor_optix.scheduler.pool import ConcurrentTrialScheduler


def _agent_factory(params):
    from conftest import DummyAgent
    return DummyAgent(hyperparams=HyperparamSet(params=params, episode_id=0))


def _pipeline_factory():
    from conftest import DummyPipeline
    return DummyPipeline(rewards_sequence=[[1.0, 1.0, 1.0]])


def test_concurrent_run_writes_distinct_ledger_records(tmp_path):
    store = ExperimentStore(str(tmp_path / "ledger.sqlite3"))
    scheduler = ConcurrentTrialScheduler(
        agent_factory=_agent_factory,
        pipeline_factory=_pipeline_factory,
        param_space={"learning_rate": ("float", 1e-4, 1e-3)},
        n_trials=4,
        trial_steps=2,
        n_workers=2,
        poll_interval_s=0.05,
        ledger_store=store,
        task_env="DummyEnv",
        algorithm="Dummy",
        study_name="integration_test",
    )
    scheduler.run()

    records = store.query_experiments(study_id="integration_test", limit=10)
    assert len(records) == 4
    experiment_ids = {r.experiment_id for r in records}
    assert len(experiment_ids) == 4  # all distinct
    checkpoint_dirs = {r.checkpoint_dir for r in records}
    assert len(checkpoint_dirs) == 4  # isolation: distinct checkpoint dirs
    assert all(r.state == ExperimentState.COMPLETED for r in records)
    store.close()


def test_run_returns_best_params_and_score_contract(tmp_path):
    scheduler = ConcurrentTrialScheduler(
        agent_factory=_agent_factory,
        pipeline_factory=_pipeline_factory,
        param_space={"learning_rate": ("float", 1e-4, 1e-3)},
        n_trials=2,
        trial_steps=2,
        n_workers=2,
        poll_interval_s=0.05,
    )
    best_params, best_score = scheduler.run()
    assert isinstance(best_params, dict)
    assert isinstance(best_score, float)


def test_overlapping_start_finish_timestamps_prove_concurrency(tmp_path):
    """Deterministic, non-timing-ratio proof that trials genuinely ran in
    parallel: checks real started_at/finished_at wall-clock timestamps
    recorded by the worker's result payload (relayed via the ledger's
    started_at/finished_at columns is not directly exposed, so this reads
    the worker results captured by the scheduler during the run instead)."""
    results = []
    scheduler = ConcurrentTrialScheduler(
        agent_factory=_agent_factory,
        pipeline_factory=_pipeline_factory,
        param_space={"learning_rate": ("float", 1e-4, 1e-3)},
        n_trials=4,
        trial_steps=2,
        n_workers=2,
        poll_interval_s=0.05,
    )

    # Monkeypatch _handle_result to capture raw worker result payloads
    # (which carry real started_at/finished_at) without changing production
    # behavior.
    original_handle = scheduler._handle_result

    def _capturing_handle(ctx, trial_number, active, result):
        results.append(result)
        return original_handle(ctx, trial_number, active, result)

    scheduler._handle_result = _capturing_handle
    scheduler.run()

    assert len(results) == 4
    intervals = [(r["started_at"], r["finished_at"]) for r in results if r.get("started_at")]
    assert len(intervals) == 4

    def overlaps(a, b):
        return a[0] < b[1] and b[0] < a[1]

    any_overlap = any(
        overlaps(intervals[i], intervals[j])
        for i in range(len(intervals))
        for j in range(i + 1, len(intervals))
    )
    assert any_overlap, "No concurrent execution observed — scheduler ran trials serially"
