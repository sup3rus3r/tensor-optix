import optuna
import pytest

from tensor_optix.core.types import HyperparamSet
from tensor_optix.ledger.records import EvaluationStatus, ExperimentState
from tensor_optix.ledger.store import ExperimentStore
from tensor_optix.orchestrator import TrialOrchestrator
from tensor_optix.selection.selector import StrategySelector


def _seed_method(store, task_env, method, scores):
    for i, score in enumerate(scores):
        experiment_id = f"{method}-{i}"
        store.create_experiment(
            experiment_id=experiment_id,
            task_env=task_env,
            algorithm="DummyAlgo",
            framework="torch",
            seed=i,
            checkpoint_dir="",
            config={"sampler": method},
        )
        store.start_experiment(experiment_id)
        store.record_progress(experiment_id, best_primary_score=score)
        store.finalize_experiment(
            experiment_id, state=ExperimentState.COMPLETED,
            evaluation_status=EvaluationStatus.PRELIMINARY,
        )


def test_stats_for_computes_best_score_and_interval(tmp_path):
    store = ExperimentStore(str(tmp_path / "ledger.sqlite3"))
    _seed_method(store, "CartPole-v1", "MethodA", [100.0, 200.0, 300.0])

    stats = StrategySelector(store).stats_for(task_env="CartPole-v1", method="MethodA")
    assert stats.n_experiments == 3
    assert stats.n_completed == 3
    assert stats.best_score == 300.0
    assert stats.score_interval is not None
    assert stats.success_rate == 1.0
    store.close()


def test_stats_for_empty_history_returns_zero_stats(tmp_path):
    store = ExperimentStore(str(tmp_path / "ledger.sqlite3"))
    stats = StrategySelector(store).stats_for(task_env="CartPole-v1", method="Unseen")
    assert stats.n_experiments == 0
    assert stats.n_completed == 0
    assert stats.best_score is None
    assert stats.score_interval is None
    assert stats.success_rate == 0.0
    store.close()


def test_compare_returns_stats_for_all_methods(tmp_path):
    store = ExperimentStore(str(tmp_path / "ledger.sqlite3"))
    _seed_method(store, "CartPole-v1", "MethodA", [100.0])
    _seed_method(store, "CartPole-v1", "MethodB", [50.0])

    result = StrategySelector(store).compare(task_env="CartPole-v1", methods=["MethodA", "MethodB"])
    assert set(result.keys()) == {"MethodA", "MethodB"}
    store.close()


def test_propose_allocates_minimum_exploration_to_untried_method(tmp_path):
    store = ExperimentStore(str(tmp_path / "ledger.sqlite3"))
    _seed_method(store, "CartPole-v1", "MethodA", [100.0, 150.0, 200.0])
    # MethodB has no history at all.

    decision = StrategySelector(store).propose(
        task_env="CartPole-v1", methods=["MethodA", "MethodB"], budget=10,
    )
    assert decision.allocations["MethodB"] > 0
    store.close()


def test_propose_favors_higher_hi95_method(tmp_path):
    store = ExperimentStore(str(tmp_path / "ledger.sqlite3"))
    _seed_method(store, "CartPole-v1", "MethodA", [400.0, 420.0, 410.0, 430.0])
    _seed_method(store, "CartPole-v1", "MethodB", [100.0, 110.0, 90.0, 105.0])

    decision = StrategySelector(store).propose(
        task_env="CartPole-v1", methods=["MethodA", "MethodB"], budget=10,
    )
    assert decision.allocations["MethodA"] > decision.allocations["MethodB"]
    assert "MethodA" in decision.rationale
    assert "MethodB" in decision.rationale
    assert "hi95" in decision.rationale
    store.close()


def test_propose_rejects_task_features_in_phase3(tmp_path):
    store = ExperimentStore(str(tmp_path / "ledger.sqlite3"))
    with pytest.raises(NotImplementedError):
        StrategySelector(store).propose(
            task_env="CartPole-v1", methods=["MethodA"], budget=5,
            task_features={"reward_sparsity": 0.5},
        )
    store.close()


def test_observe_delegates_to_backfill(tmp_path, dummy_evaluator):
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
        n_trials=2,
        trial_steps=2,
        rloptimizer_kwargs={"evaluator": dummy_evaluator},
        optuna_verbosity=optuna.logging.ERROR,
    )
    orch.run()

    store = ExperimentStore(str(tmp_path / "ledger.sqlite3"))
    records = StrategySelector(store).observe(
        orchestrator=orch, task_env="DummyEnv", algorithm="Dummy",
    )
    assert len(records) == 2
    assert len(store.query_experiments(task_env="DummyEnv", limit=100)) == 2
    store.close()
