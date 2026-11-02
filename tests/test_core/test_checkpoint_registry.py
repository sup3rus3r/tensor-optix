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


# ---------------------------------------------------------------------------
# resumable_components (weights-only vs. full-state checkpoints)
# ---------------------------------------------------------------------------

class ResumableAgent:
    """Duck-typed agent (not a BaseAgent subclass) that opts into full
    resumability, to prove CheckpointRegistry doesn't assume BaseAgent."""

    def __init__(self):
        self._weights = {"w": 1.0}
        self.full_state_saved_to = None
        self.full_state_loaded_from = None

    def save_weights(self, path):
        import json, os
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "weights.json"), "w") as f:
            json.dump(self._weights, f)

    def load_weights(self, path):
        pass

    @property
    def resumable_components(self):
        return frozenset({"optimizer", "rng"})

    def save_full_state(self, path):
        self.save_weights(path)
        self.full_state_saved_to = path

    def load_full_state(self, path):
        self.full_state_loaded_from = path


def test_weights_only_agent_records_empty_resumable_components(tmp_registry, dummy_agent_for_registry):
    # DummyAgent subclasses BaseAgent and never overrides resumable_components
    # -> must default to empty (weights-only), never silently claim more.
    metrics = EvalMetrics(primary_score=1.0, metrics={}, episode_id=0)
    hp = HyperparamSet(params={}, episode_id=0)
    snapshot = tmp_registry.save(dummy_agent_for_registry, metrics, hp)
    assert snapshot.resumable_components == frozenset()


def test_full_state_agent_records_resumable_components(tmp_path):
    registry = CheckpointRegistry(str(tmp_path / "ckpt"), max_snapshots=5)
    agent = ResumableAgent()
    metrics = EvalMetrics(primary_score=1.0, metrics={}, episode_id=0)
    hp = HyperparamSet(params={}, episode_id=0)

    snapshot = registry.save(agent, metrics, hp)

    assert snapshot.resumable_components == frozenset({"optimizer", "rng"})
    assert agent.full_state_saved_to == snapshot.weights_path

    with open(os.path.join(snapshot.weights_path.rsplit("weights", 1)[0], "metadata.json")) as f:
        import json
        meta = json.load(f)
    assert sorted(meta["resumable_components"]) == ["optimizer", "rng"]


def test_load_best_calls_load_full_state_when_available(tmp_path):
    registry = CheckpointRegistry(str(tmp_path / "ckpt"), max_snapshots=5)
    agent = ResumableAgent()
    metrics = EvalMetrics(primary_score=1.0, metrics={}, episode_id=0)
    hp = HyperparamSet(params={}, episode_id=0)
    registry.save(agent, metrics, hp)

    registry.load_best(agent)
    assert agent.full_state_loaded_from == registry.best.weights_path


def test_resumable_components_survive_manifest_reload(tmp_path):
    checkpoint_dir = str(tmp_path / "persist_resumable")
    agent = ResumableAgent()
    r1 = CheckpointRegistry(checkpoint_dir, max_snapshots=5)
    metrics = EvalMetrics(primary_score=1.0, metrics={}, episode_id=0)
    hp = HyperparamSet(params={}, episode_id=0)
    r1.save(agent, metrics, hp)

    r2 = CheckpointRegistry(checkpoint_dir, max_snapshots=5)
    snapshot = r2.load_best(agent)
    assert snapshot.resumable_components == frozenset({"optimizer", "rng"})


def test_old_metadata_without_resumable_components_defaults_to_empty(tmp_path):
    # Simulates a checkpoint saved before this field existed -- must load
    # as weights-only (empty), not crash on the missing key.
    checkpoint_dir = str(tmp_path / "legacy")
    agent = ResumableAgent()
    registry = CheckpointRegistry(checkpoint_dir, max_snapshots=5)
    metrics = EvalMetrics(primary_score=1.0, metrics={}, episode_id=0)
    hp = HyperparamSet(params={}, episode_id=0)
    snapshot = registry.save(agent, metrics, hp)

    import json
    meta_path = os.path.join(checkpoint_dir, f"snapshot_{snapshot.snapshot_id}", "metadata.json")
    with open(meta_path) as f:
        meta = json.load(f)
    del meta["resumable_components"]
    with open(meta_path, "w") as f:
        json.dump(meta, f)

    r2 = CheckpointRegistry(checkpoint_dir, max_snapshots=5)
    reloaded = r2.load_best(agent)
    assert reloaded.resumable_components == frozenset()
