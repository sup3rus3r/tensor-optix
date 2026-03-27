import pytest
from tensor_optix.core.policy_manager import PolicyManager, PolicyManagerCallback
from tensor_optix.core.checkpoint_registry import CheckpointRegistry
from tensor_optix.core.types import EvalMetrics, HyperparamSet


def make_registry(tmp_path):
    return CheckpointRegistry(str(tmp_path / "checkpoints"), max_snapshots=5)


def make_metrics(score):
    return EvalMetrics(primary_score=score, metrics={}, episode_id=0)


def make_hp():
    return HyperparamSet(params={"lr": 1e-3}, episode_id=0)


# -----------------------------------------------------------------------
# budget_exhausted / spawns_remaining properties
# -----------------------------------------------------------------------

def test_no_budget_never_exhausted():
    pm = PolicyManager.__new__(PolicyManager)
    pm._registry = None
    pm._ensemble = []
    pm._score_history = {}
    pm._score_window = 10
    pm._spawn_count = 0
    pm._prune_count = 0
    pm._current_regime = None
    pm._max_spawns = None
    pm._max_ensemble_size = None

    assert pm.budget_exhausted is False
    assert pm.spawns_remaining is None


def test_budget_not_exhausted_before_limit():
    pm = PolicyManager.__new__(PolicyManager)
    pm._max_spawns = 3
    pm._max_ensemble_size = None
    pm._spawn_count = 2
    assert pm.budget_exhausted is False
    assert pm.spawns_remaining == 1


def test_budget_exhausted_at_limit():
    pm = PolicyManager.__new__(PolicyManager)
    pm._max_spawns = 3
    pm._max_ensemble_size = None
    pm._spawn_count = 3
    assert pm.budget_exhausted is True
    assert pm.spawns_remaining == 0


def test_budget_exhausted_beyond_limit():
    pm = PolicyManager.__new__(PolicyManager)
    pm._max_spawns = 2
    pm._max_ensemble_size = None
    pm._spawn_count = 5  # shouldn't happen but be safe
    assert pm.budget_exhausted is True
    assert pm.spawns_remaining == 0


# -----------------------------------------------------------------------
# spawn_variant counts against budget
# -----------------------------------------------------------------------

def test_spawn_increments_count(tmp_path):
    from tests.conftest import DummyAgent
    registry = make_registry(tmp_path)
    agent = DummyAgent()
    registry.save(agent, make_metrics(10.0), make_hp())

    pm = PolicyManager(registry, max_spawns=2)
    assert pm.spawns_remaining == 2

    pm.spawn_variant(DummyAgent())
    assert pm._spawn_count == 1
    assert pm.spawns_remaining == 1
    assert pm.budget_exhausted is False

    pm.spawn_variant(DummyAgent())
    assert pm._spawn_count == 2
    assert pm.spawns_remaining == 0
    assert pm.budget_exhausted is True


# -----------------------------------------------------------------------
# status() includes budget fields
# -----------------------------------------------------------------------

def test_status_includes_budget_fields():
    pm = PolicyManager.__new__(PolicyManager)
    pm._registry = None
    pm._ensemble = []
    pm._score_history = {}
    pm._score_window = 10
    pm._spawn_count = 1
    pm._prune_count = 0
    pm._current_regime = None
    pm._max_spawns = 3
    pm._max_ensemble_size = None

    s = pm.status()
    assert s["max_spawns"] == 3
    assert s["spawns_remaining"] == 2
    assert s["budget_exhausted"] is False


def test_status_budget_exhausted_true():
    pm = PolicyManager.__new__(PolicyManager)
    pm._registry = None
    pm._ensemble = []
    pm._score_history = {}
    pm._score_window = 10
    pm._spawn_count = 3
    pm._prune_count = 0
    pm._current_regime = None
    pm._max_spawns = 3
    pm._max_ensemble_size = None

    assert pm.status()["budget_exhausted"] is True


# -----------------------------------------------------------------------
# PolicyManagerCallback.set_stop_fn + termination
# -----------------------------------------------------------------------

def test_set_stop_fn_called_when_budget_exhausted(tmp_path):
    from tests.conftest import DummyAgent
    registry = make_registry(tmp_path)
    agent = DummyAgent()
    registry.save(agent, make_metrics(100.0), make_hp())

    pm = PolicyManager(registry, max_spawns=1)
    pm.spawn_variant(DummyAgent())  # exhaust budget
    assert pm.budget_exhausted is True

    stop_calls = []
    cb = pm.as_callback(agent)
    cb.set_stop_fn(lambda: stop_calls.append(1))

    cb.on_episode_end(1, make_metrics(50.0))
    cb.on_dormant(1)

    assert len(stop_calls) == 1


def test_stop_fn_not_called_when_budget_not_exhausted(tmp_path):
    from tests.conftest import DummyAgent
    registry = make_registry(tmp_path)
    agent = DummyAgent()
    registry.save(agent, make_metrics(100.0), make_hp())

    pm = PolicyManager(registry, max_spawns=3)
    pm.spawn_variant(DummyAgent())  # 1 of 3 used

    stop_calls = []
    cb = pm.as_callback(agent)
    cb.set_stop_fn(lambda: stop_calls.append(1))

    cb.on_episode_end(1, make_metrics(50.0))
    cb.on_dormant(1)

    assert len(stop_calls) == 0


def test_stop_fn_not_called_when_no_budget(tmp_path):
    from tests.conftest import DummyAgent
    registry = make_registry(tmp_path)
    agent = DummyAgent()
    registry.save(agent, make_metrics(100.0), make_hp())

    pm = PolicyManager(registry)  # no max_spawns
    stop_calls = []
    cb = pm.as_callback(agent)
    cb.set_stop_fn(lambda: stop_calls.append(1))

    cb.on_episode_end(1, make_metrics(50.0))
    cb.on_dormant(1)

    assert len(stop_calls) == 0


def test_budget_exhausted_fires_stop_even_without_score(tmp_path):
    from tests.conftest import DummyAgent
    registry = make_registry(tmp_path)
    agent = DummyAgent()

    pm = PolicyManager(registry, max_spawns=0)  # already exhausted
    stop_calls = []
    cb = pm.as_callback(agent)
    cb.set_stop_fn(lambda: stop_calls.append(1))

    # No on_episode_end — _last_score is None
    cb.on_dormant(1)

    assert len(stop_calls) == 1
