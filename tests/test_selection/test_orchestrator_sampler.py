import optuna
import pytest

from tensor_optix.orchestrator import TrialOrchestrator
from tensor_optix.core.types import HyperparamSet


def _make_orchestrator(dummy_evaluator, dummy_pipeline_factory, **kwargs):
    def agent_factory(params):
        from conftest import DummyAgent
        return DummyAgent(hyperparams=HyperparamSet(params=params, episode_id=0))

    return TrialOrchestrator(
        agent_factory=agent_factory,
        pipeline_factory=dummy_pipeline_factory,
        param_space={"learning_rate": ("float", 1e-4, 1e-3)},
        n_trials=2,
        trial_steps=2,
        rloptimizer_kwargs={"evaluator": dummy_evaluator},
        optuna_verbosity=optuna.logging.ERROR,
        **kwargs,
    )


@pytest.fixture
def dummy_pipeline_factory():
    from conftest import DummyPipeline
    return lambda: DummyPipeline(rewards_sequence=[[1.0, 1.0, 1.0]])


def test_default_sampler_is_tpe(dummy_evaluator, dummy_pipeline_factory):
    orch = _make_orchestrator(dummy_evaluator, dummy_pipeline_factory)
    assert type(orch.sampler).__name__ == "TPESampler"
    assert type(orch.study.sampler).__name__ == "TPESampler"


def test_custom_sampler_is_used(dummy_evaluator, dummy_pipeline_factory):
    sampler = optuna.samplers.RandomSampler(seed=0)
    orch = _make_orchestrator(dummy_evaluator, dummy_pipeline_factory, sampler=sampler)
    assert type(orch.sampler).__name__ == "RandomSampler"
    assert orch.study.sampler is sampler


def test_random_sampler_runs_end_to_end(dummy_evaluator, dummy_pipeline_factory):
    orch = _make_orchestrator(
        dummy_evaluator, dummy_pipeline_factory,
        sampler=optuna.samplers.RandomSampler(seed=0),
    )
    best_params, best_score = orch.run()
    assert isinstance(best_params, dict)
    assert "learning_rate" in best_params
    assert isinstance(best_score, float)


def test_existing_default_behavior_unchanged(dummy_evaluator, dummy_pipeline_factory):
    orch = _make_orchestrator(dummy_evaluator, dummy_pipeline_factory)
    result = orch.run()
    assert isinstance(result, tuple)
    best_params, best_score = result
    assert isinstance(best_params, dict)
    assert isinstance(best_score, float)
