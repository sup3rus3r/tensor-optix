from tensor_optix.scheduler.pool import ConcurrentTrialScheduler
from tensor_optix.core.types import HyperparamSet


def _agent_factory(params):
    from conftest import DummyAgent
    return DummyAgent(hyperparams=HyperparamSet(params=params, episode_id=0))


def _pipeline_factory():
    from conftest import DummyPipeline
    return DummyPipeline(rewards_sequence=[[1.0, 1.0, 1.0]])


def test_run_returns_best_params_and_score_contract(tmp_path):
    scheduler = ConcurrentTrialScheduler(
        agent_factory=_agent_factory,
        pipeline_factory=_pipeline_factory,
        param_space={"learning_rate": ("float", 1e-4, 1e-3)},
        n_trials=3,
        trial_steps=2,
        n_workers=2,
        poll_interval_s=0.05,
    )
    best_params, best_score = scheduler.run()
    assert isinstance(best_params, dict)
    assert "learning_rate" in best_params
    assert isinstance(best_score, float)


def test_never_more_than_n_workers_alive_at_once(tmp_path):
    n_workers = 2
    scheduler = ConcurrentTrialScheduler(
        agent_factory=_agent_factory,
        pipeline_factory=_pipeline_factory,
        param_space={"learning_rate": ("float", 1e-4, 1e-3)},
        n_trials=6,
        trial_steps=2,
        n_workers=n_workers,
        poll_interval_s=0.05,
    )
    scheduler.run()

    # max_concurrent_observed is a high-water mark recorded by the dispatch
    # loop itself (incremented only when len(self._active) is checked
    # immediately after starting a worker process) -- a deterministic,
    # non-timing proof the bounded-pool invariant held throughout the run.
    assert 0 < scheduler.max_concurrent_observed <= n_workers
    assert len(scheduler._active) == 0
    assert len(scheduler._active_procs) == 0


def test_excess_trials_wait_for_free_slot(tmp_path):
    scheduler = ConcurrentTrialScheduler(
        agent_factory=_agent_factory,
        pipeline_factory=_pipeline_factory,
        param_space={"learning_rate": ("float", 1e-4, 1e-3)},
        n_trials=5,
        trial_steps=2,
        n_workers=2,
        poll_interval_s=0.05,
    )
    scheduler.run()
    assert len(scheduler.study.trials) >= 5
    assert len(scheduler._active) == 0
    assert len(scheduler._active_procs) == 0
