import os
import pytest
import tempfile
from tensor_optix.core.checkpoint_registry import CheckpointRegistry
from tensor_optix.core.types import EvalMetrics, HyperparamSet


@pytest.fixture
def tmp_registry(tmp_path):
    return CheckpointRegistry(str(tmp_path / "checkpoints"), max_snapshots=3)


@pytest.fixture
def dummy_agent_for_registry(tmp_path):
    from conftest import DummyAgent
    return DummyAgent()


def test_save_creates_snapshot(tmp_registry, dummy_agent_for_registry):
    metrics = EvalMetrics(primary_score=10.0, metrics={}, episode_id=0)
    hp = HyperparamSet(params={"lr": 1e-3}, episode_id=0)
    snapshot = tmp_registry.save(dummy_agent_for_registry, metrics, hp)
    assert snapshot.snapshot_id is not None
    assert snapshot.eval_metrics.primary_score == 10.0
    assert tmp_registry.best is not None
    assert tmp_registry.best.eval_metrics.primary_score == 10.0


def test_save_creates_files_on_disk(tmp_registry, dummy_agent_for_registry):
    metrics = EvalMetrics(primary_score=5.0, metrics={}, episode_id=1)
    hp = HyperparamSet(params={"lr": 1e-3}, episode_id=1)
    snapshot = tmp_registry.save(dummy_agent_for_registry, metrics, hp)
    assert os.path.exists(snapshot.weights_path)


def test_load_best_restores_agent(tmp_registry, dummy_agent_for_registry):
    metrics = EvalMetrics(primary_score=7.0, metrics={}, episode_id=0)
    hp = HyperparamSet(params={"lr": 1e-3}, episode_id=0)
    tmp_registry.save(dummy_agent_for_registry, metrics, hp)
    result = tmp_registry.load_best(dummy_agent_for_registry)
    assert result is not None
    assert result.eval_metrics.primary_score == 7.0


def test_load_best_returns_none_when_empty(tmp_path):
    registry = CheckpointRegistry(str(tmp_path / "empty"), max_snapshots=3)
    from conftest import DummyAgent
    result = registry.load_best(DummyAgent())
    assert result is None


def test_prune_keeps_max_snapshots(tmp_path):
    registry = CheckpointRegistry(str(tmp_path / "prune_test"), max_snapshots=2)
    from conftest import DummyAgent
    agent = DummyAgent()
    for i in range(4):
        metrics = EvalMetrics(primary_score=float(i), metrics={}, episode_id=i)
        hp = HyperparamSet(params={"lr": 1e-3}, episode_id=i)
        registry.save(agent, metrics, hp)

    manifest = registry._load_manifest()
    assert len(manifest) == 2


def test_manifest_persists_across_instances(tmp_path):
    checkpoint_dir = str(tmp_path / "persist_test")
    from conftest import DummyAgent
    agent = DummyAgent()

    r1 = CheckpointRegistry(checkpoint_dir, max_snapshots=5)
    metrics = EvalMetrics(primary_score=42.0, metrics={}, episode_id=0)
    hp = HyperparamSet(params={"lr": 1e-3}, episode_id=0)
    r1.save(agent, metrics, hp)

    r2 = CheckpointRegistry(checkpoint_dir, max_snapshots=5)
    result = r2.load_best(agent)
    assert result is not None
    assert result.eval_metrics.primary_score == 42.0
