import queue

from tensor_optix.core.types import HyperparamSet
from tensor_optix.ledger.records import EvaluationStatus, ExperimentState
from tensor_optix.ledger.store import ExperimentStore
from tensor_optix.scheduler.pool import ConcurrentTrialScheduler, _ActiveTrial


def _agent_factory(params):
    from conftest import DummyAgent
    return DummyAgent(hyperparams=HyperparamSet(params=params, episode_id=0))


def _fast_pipeline_factory():
    from conftest import DummyPipeline
    return DummyPipeline(rewards_sequence=[[1.0, 1.0, 1.0]])


def _hanging_agent_factory(params):
    """An agent whose learn() sleeps forever — simulates a genuinely hung
    trial for the timeout path, without relying on timing-ratio luck."""
    import time as _time
    from conftest import DummyAgent

    class HangingAgent(DummyAgent):
        def learn(self, episode_data):
            _time.sleep(60)
            return super().learn(episode_data)

    return HangingAgent(hyperparams=HyperparamSet(params=params, episode_id=0))


def test_hung_trial_is_terminated_and_marked_timed_out(tmp_path):
    store = ExperimentStore(str(tmp_path / "ledger.sqlite3"))
    scheduler = ConcurrentTrialScheduler(
        agent_factory=_hanging_agent_factory,
        pipeline_factory=_fast_pipeline_factory,
        param_space={"learning_rate": ("float", 1e-4, 1e-3)},
        n_trials=1,
        trial_steps=2,
        n_workers=1,
        trial_timeout_s=2.0,
        poll_interval_s=0.1,
        ledger_store=store,
        task_env="DummyEnv",
        algorithm="Dummy",
        study_name="timeout_test",
    )
    scheduler.run()

    records = store.query_experiments(study_id="timeout_test", limit=10)
    assert len(records) == 1
    assert records[0].state == ExperimentState.TIMED_OUT
    store.close()


def test_pruned_trial_cancel_event_is_set(tmp_path):
    """Forces MedianPruner to actually fire (n_startup_trials=1,
    n_warmup_steps=0 makes pruning deterministic on the second trial once
    the first trial's score establishes a median) and asserts the
    scheduler's _poll_reports sets the correct worker's cancel_event in
    response, without needing a real multi-minute training run."""
    scheduler = ConcurrentTrialScheduler(
        agent_factory=_agent_factory,
        pipeline_factory=_fast_pipeline_factory,
        param_space={"learning_rate": ("float", 1e-4, 1e-3)},
        n_trials=1,
        trial_steps=2,
        n_workers=1,
        n_startup_trials=1,
        pruner_warmup_steps=0,
        pruner_interval=1,
        poll_interval_s=0.05,
    )

    # Seed the study with one high-scoring completed trial so the pruner
    # has a real median to compare against.
    from tensor_optix.orchestrator import _sample_params

    seed_trial = scheduler.study.ask()
    _sample_params(seed_trial, scheduler._param_space)
    seed_trial.report(1000.0, step=0)
    scheduler.study.tell(seed_trial, 1000.0)

    # Now construct a second trial, report a low score, and verify the
    # scheduler would prune it via the exact code path _poll_reports uses.
    trial = scheduler.study.ask()
    _sample_params(trial, scheduler._param_space)
    trial.report(1.0, step=0)
    assert bool(trial.should_prune()) == True

    cancel_event = __import__("threading").Event()
    fake_active = _ActiveTrial(
        trial=trial, process=None, cancel_event=cancel_event,
        result_queue=None, report_queue=_FakeReportQueue([(trial.number, 0, 1.0)]),
        params={}, experiment_id=None,
    )
    scheduler._active[trial.number] = fake_active
    scheduler._poll_reports()

    assert cancel_event.is_set() == True
    del scheduler._active[trial.number]


class _FakeReportQueue:
    def __init__(self, items):
        self._items = list(items)

    def get_nowait(self):
        if not self._items:
            raise queue.Empty
        return self._items.pop(0)
