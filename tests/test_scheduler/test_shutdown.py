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


def _broken_agent_factory(params):
    raise RuntimeError("deliberate trial failure")


def test_all_worker_processes_reaped_after_run(tmp_path):
    scheduler = ConcurrentTrialScheduler(
        agent_factory=_agent_factory,
        pipeline_factory=_pipeline_factory,
        param_space={"learning_rate": ("float", 1e-4, 1e-3)},
        n_trials=3,
        trial_steps=2,
        n_workers=2,
        poll_interval_s=0.05,
    )
    scheduler.run()
    assert len(scheduler._active_procs) == 0
    assert len(scheduler._active) == 0


def test_clean_failure_is_marked_failed_not_retried_forever(tmp_path):
    """A clean Python exception should exhaust retries quickly (not retry
    forever) and land as FAILED in the ledger — proves the retry policy's
    non-transient classification is actually wired into the real run loop,
    not just the pure retry.classify() unit test."""
    store = ExperimentStore(str(tmp_path / "ledger.sqlite3"))
    scheduler = ConcurrentTrialScheduler(
        agent_factory=_broken_agent_factory,
        pipeline_factory=_pipeline_factory,
        param_space={"learning_rate": ("float", 1e-4, 1e-3)},
        n_trials=1,
        trial_steps=2,
        n_workers=1,
        max_retries=2,
        poll_interval_s=0.05,
        ledger_store=store,
        task_env="DummyEnv",
        algorithm="Dummy",
        study_name="shutdown_failure_test",
    )
    scheduler.run()

    records = store.query_experiments(study_id="shutdown_failure_test", limit=10)
    assert len(records) == 1
    assert records[0].state == ExperimentState.FAILED
    assert records[0].error_message == "deliberate trial failure"
    store.close()
    assert len(scheduler._active_procs) == 0


def test_shutdown_worker_inspects_exitcode(tmp_path):
    """_shutdown_worker must return the real exitcode so a crashed worker
    is distinguishable from a hung one, per the plan's explicit requirement
    (async_learner's proven pattern never inspected exitcode -- this fixes
    that gap)."""
    scheduler = ConcurrentTrialScheduler(
        agent_factory=_agent_factory,
        pipeline_factory=_pipeline_factory,
        param_space={"learning_rate": ("float", 1e-4, 1e-3)},
        n_trials=1,
        trial_steps=2,
        n_workers=1,
        poll_interval_s=0.05,
    )
    import multiprocessing as mp
    from tests.test_scheduler._noop_target import noop_process_target

    ctx = mp.get_context("spawn") if __import__("sys").platform != "linux" else mp.get_context("fork")

    proc = ctx.Process(target=noop_process_target, daemon=True)
    proc.start()
    proc.join(timeout=10)
    exitcode, reaped = scheduler._shutdown_worker(proc)
    assert exitcode == 0
    assert reaped is True


# NOTE: test_kill_escalation_after_terminate_timeout from the plan is
# intentionally not implemented. On Windows, Process.terminate() maps to
# TerminateProcess(), which the target process cannot catch or ignore
# (unlike POSIX SIGTERM) -- confirmed by direct experimentation. A
# genuinely terminate()-resistant worker is not constructible on this
# platform, so the kill() escalation path, while present in
# _shutdown_worker for defense-in-depth and POSIX portability, is
# unreachable dead code on Windows and not meaningfully testable here.
