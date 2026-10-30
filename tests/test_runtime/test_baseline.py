import os

from tensor_optix.core.types import HyperparamSet
from tensor_optix.ledger import ExperimentState, ExperimentStore
from tensor_optix.runtime.baseline import (
    ROLE_BASELINE,
    establish_baseline,
    get_baseline_weights_path,
    get_current_baseline_experiment_id,
)
from tests.conftest import DummyAgent, DummyEvaluator, DummyPipeline


def _dummy_agent_factory(params):
    return DummyAgent()


def _dummy_pipeline_factory():
    return DummyPipeline(rewards_sequence=[[1.0, 2.0, 3.0]])


def test_establish_baseline_creates_exactly_one_marked_row(tmp_path):
    store = ExperimentStore(str(tmp_path / "ledger.sqlite3"))
    exp_id = establish_baseline(
        store, agent_factory=_dummy_agent_factory, pipeline_factory=_dummy_pipeline_factory,
        evaluator=DummyEvaluator(), task_env="DummyTask", algorithm="Dummy", study_id="study1",
        trial_steps=3, checkpoint_dir=str(tmp_path / "ckpt"),
    )
    record = store.get_experiment(exp_id)
    assert record is not None
    assert record.config.get("role") == ROLE_BASELINE
    assert record.state == ExperimentState.COMPLETED
    assert record.best_primary_score is not None
    store.close()


def test_establish_baseline_weights_path_exists_and_loads(tmp_path):
    store = ExperimentStore(str(tmp_path / "ledger.sqlite3"))
    exp_id = establish_baseline(
        store, agent_factory=_dummy_agent_factory, pipeline_factory=_dummy_pipeline_factory,
        evaluator=DummyEvaluator(), task_env="DummyTask", algorithm="Dummy", study_id="study2",
        trial_steps=3, checkpoint_dir=str(tmp_path / "ckpt"),
    )
    weights_path = get_baseline_weights_path(store, exp_id)
    assert weights_path is not None
    assert os.path.exists(weights_path)

    fresh_agent = _dummy_agent_factory({})
    fresh_agent.load_weights(weights_path)  # must not raise
    store.close()


def test_get_current_baseline_returns_most_recent(tmp_path):
    store = ExperimentStore(str(tmp_path / "ledger.sqlite3"))
    first_id = establish_baseline(
        store, agent_factory=_dummy_agent_factory, pipeline_factory=_dummy_pipeline_factory,
        evaluator=DummyEvaluator(), task_env="DummyTask", algorithm="Dummy", study_id="study3",
        trial_steps=3, checkpoint_dir=str(tmp_path / "ckpt1"), seed=0,
    )
    second_id = establish_baseline(
        store, agent_factory=_dummy_agent_factory, pipeline_factory=_dummy_pipeline_factory,
        evaluator=DummyEvaluator(), task_env="DummyTask", algorithm="Dummy", study_id="study3",
        trial_steps=3, checkpoint_dir=str(tmp_path / "ckpt2"), seed=1,
    )
    assert first_id != second_id

    current = get_current_baseline_experiment_id(store, task_env="DummyTask", algorithm="Dummy", study_id="study3")
    assert current == second_id  # most recently created
    store.close()


def test_get_current_baseline_returns_none_when_absent(tmp_path):
    store = ExperimentStore(str(tmp_path / "ledger.sqlite3"))
    current = get_current_baseline_experiment_id(store, task_env="NeverSeen", algorithm="Dummy", study_id="study4")
    assert current is None
    store.close()
