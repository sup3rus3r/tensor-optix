import optuna
import pytest

from tensor_optix.orchestrator import TrialOrchestrator
from tensor_optix.core.types import HyperparamSet
from tensor_optix.ledger.records import EvaluationStatus, ExperimentState
from tensor_optix.ledger.store import ExperimentStore
from tensor_optix.selection.backfill import backfill_study_to_ledger


def _make_orchestrator(dummy_evaluator, rewards_sequence, n_trials=3, trial_steps=2, sampler=None):
    def agent_factory(params):
        from conftest import DummyAgent
        return DummyAgent(hyperparams=HyperparamSet(params=params, episode_id=0))

    def pipeline_factory():
        from conftest import DummyPipeline
        return DummyPipeline(rewards_sequence=rewards_sequence)

    kwargs = {}
    if sampler is not None:
        kwargs["sampler"] = sampler

    return TrialOrchestrator(
        agent_factory=agent_factory,
        pipeline_factory=pipeline_factory,
        param_space={"learning_rate": ("float", 1e-4, 1e-3)},
        n_trials=n_trials,
        trial_steps=trial_steps,
        rloptimizer_kwargs={"evaluator": dummy_evaluator},
        optuna_verbosity=optuna.logging.ERROR,
        **kwargs,
    )


def test_backfill_creates_one_record_per_trial(tmp_path, dummy_evaluator):
    orch = _make_orchestrator(dummy_evaluator, [[1.0, 1.0, 1.0]], n_trials=3)
    orch.run()

    store = ExperimentStore(str(tmp_path / "ledger.sqlite3"))
    records = backfill_study_to_ledger(
        orch, store, task_env="DummyEnv", algorithm="Dummy",
    )
    assert len(records) == 3
    assert all(r.evaluation_status == EvaluationStatus.PRELIMINARY for r in records)
    store.close()


def test_backfill_maps_trial_value_to_best_primary_score(tmp_path, dummy_evaluator):
    orch = _make_orchestrator(dummy_evaluator, [[1.0, 1.0, 1.0]], n_trials=2)
    orch.run()

    store = ExperimentStore(str(tmp_path / "ledger.sqlite3"))
    records = backfill_study_to_ledger(orch, store, task_env="DummyEnv", algorithm="Dummy")

    by_trial_number = {r.config["trial_number"]: r for r in records}
    for trial in orch.study.trials:
        record = by_trial_number[trial.number]
        assert record.best_primary_score == trial.value
    store.close()


def test_backfill_is_idempotent(tmp_path, dummy_evaluator):
    orch = _make_orchestrator(dummy_evaluator, [[1.0, 1.0, 1.0]], n_trials=3)
    orch.run()

    store = ExperimentStore(str(tmp_path / "ledger.sqlite3"))
    backfill_study_to_ledger(orch, store, task_env="DummyEnv", algorithm="Dummy")
    first_count = len(store.query_experiments(limit=100))

    backfill_study_to_ledger(orch, store, task_env="DummyEnv", algorithm="Dummy")
    second_count = len(store.query_experiments(limit=100))

    assert first_count == second_count == 3
    store.close()


def test_backfill_labels_sampler_in_config(tmp_path, dummy_evaluator):
    orch = _make_orchestrator(
        dummy_evaluator, [[1.0, 1.0, 1.0]], n_trials=2,
        sampler=optuna.samplers.RandomSampler(seed=0),
    )
    orch.run()

    store = ExperimentStore(str(tmp_path / "ledger.sqlite3"))
    records = backfill_study_to_ledger(orch, store, task_env="DummyEnv", algorithm="Dummy")
    assert all(r.config["sampler"] == "RandomSampler" for r in records)
    store.close()


def test_backfill_persists_solve_threshold(tmp_path, dummy_evaluator):
    orch = _make_orchestrator(dummy_evaluator, [[1.0, 1.0, 1.0]], n_trials=2)
    orch.run()

    store = ExperimentStore(str(tmp_path / "ledger.sqlite3"))
    records = backfill_study_to_ledger(
        orch, store, task_env="DummyEnv", algorithm="Dummy", solve_threshold=475.0,
    )
    assert all(r.solve_threshold == 475.0 for r in records)
    store.close()


def test_backfill_maps_pruned_state(tmp_path, dummy_evaluator):
    def agent_factory(params):
        from conftest import DummyAgent
        return DummyAgent(hyperparams=HyperparamSet(params=params, episode_id=0))

    def pipeline_factory():
        from conftest import DummyPipeline
        return DummyPipeline(rewards_sequence=[[1.0, 1.0, 1.0]])

    orch = TrialOrchestrator(
        agent_factory=agent_factory,
        pipeline_factory=pipeline_factory,
        param_space={"learning_rate": ("float", 1e-4, 1e-3)},
        n_trials=1,
        trial_steps=2,
        rloptimizer_kwargs={"evaluator": dummy_evaluator},
        optuna_verbosity=optuna.logging.ERROR,
    )
    # Force a PRUNED trial directly on the underlying study, independent of
    # whether MedianPruner would prune this tiny real run (it may not with
    # so few trials) -- this guarantees deterministic PRUNED-state coverage.
    def pruned_objective(trial):
        raise optuna.TrialPruned()

    orch._study.optimize(pruned_objective, n_trials=1, catch=())

    store = ExperimentStore(str(tmp_path / "ledger.sqlite3"))
    records = backfill_study_to_ledger(orch, store, task_env="DummyEnv", algorithm="Dummy")
    assert len(records) == 1
    assert records[0].state == ExperimentState.CANCELLED
    store.close()
