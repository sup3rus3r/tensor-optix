import pytest
import numpy as np
import tempfile
from tensor_optix.core.policy_manager import PolicyManager, PolicyManagerCallback
from tensor_optix.core.checkpoint_registry import CheckpointRegistry
from tensor_optix.core.types import EvalMetrics, HyperparamSet


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def make_registry(tmp_path):
    return CheckpointRegistry(str(tmp_path / "checkpoints"), max_snapshots=5)


def make_metrics(score, episode_id=0):
    return EvalMetrics(primary_score=score, metrics={}, episode_id=episode_id)


def make_hp():
    return HyperparamSet(params={"lr": 1e-3}, episode_id=0)


# -----------------------------------------------------------------------
# PolicyManager.evolve()
# -----------------------------------------------------------------------

def test_evolve_returns_false_when_registry_empty(tmp_path):
    from conftest import DummyAgent
    registry = make_registry(tmp_path)
    pm = PolicyManager(registry)
    result = pm.evolve(DummyAgent(), current_score=5.0)
    assert result is False


def test_evolve_rolls_back_when_current_below_best(tmp_path):
    from conftest import DummyAgent
    registry = make_registry(tmp_path)
    agent = DummyAgent()

    # Save a high-score checkpoint
    registry.save(agent, make_metrics(100.0), make_hp())

    # Mark agent weights as changed
    agent._weights["w"] = 999.0

    pm = PolicyManager(registry)
    rolled_back = pm.evolve(agent, current_score=50.0)

    assert rolled_back is True
    # Weights restored from checkpoint (DummyAgent saves {"w": 1.0})
    assert agent._weights.get("w") == 1.0


def test_evolve_no_rollback_when_current_at_best(tmp_path):
    from conftest import DummyAgent
    registry = make_registry(tmp_path)
    agent = DummyAgent()

    registry.save(agent, make_metrics(50.0), make_hp())
    agent._weights["w"] = 999.0

    pm = PolicyManager(registry)
    rolled_back = pm.evolve(agent, current_score=50.0)

    assert rolled_back is False
    assert agent._weights["w"] == 999.0  # unchanged


def test_evolve_no_rollback_when_current_above_best(tmp_path):
    from conftest import DummyAgent
    registry = make_registry(tmp_path)
    agent = DummyAgent()

    registry.save(agent, make_metrics(50.0), make_hp())
    agent._weights["w"] = 999.0

    pm = PolicyManager(registry)
    rolled_back = pm.evolve(agent, current_score=75.0)

    assert rolled_back is False


# -----------------------------------------------------------------------
# PolicyManager.ensemble_action()
# -----------------------------------------------------------------------

class FixedActionAgent:
    """Test double that always returns a fixed action."""
    def __init__(self, action):
        self._action = action

    def act(self, obs):
        return np.array(self._action, dtype=float)


def test_ensemble_single_agent_delegates_directly():
    registry = CheckpointRegistry.__new__(CheckpointRegistry)
    registry._best = None
    pm = PolicyManager.__new__(PolicyManager)
    pm._registry = registry
    pm._ensemble = []

    pm.add_agent(FixedActionAgent([3.0]), weight=1.0)
    result = pm.ensemble_action(np.zeros(4))
    np.testing.assert_allclose(result, [3.0])


def test_ensemble_two_agents_equal_weights():
    pm = PolicyManager.__new__(PolicyManager)
    pm._registry = None
    pm._ensemble = []

    pm.add_agent(FixedActionAgent([0.0]), weight=1.0)
    pm.add_agent(FixedActionAgent([4.0]), weight=1.0)

    result = pm.ensemble_action(np.zeros(4))
    np.testing.assert_allclose(result, [2.0])  # (0 + 4) / 2


def test_ensemble_two_agents_unequal_weights():
    pm = PolicyManager.__new__(PolicyManager)
    pm._registry = None
    pm._ensemble = []

    pm.add_agent(FixedActionAgent([0.0]), weight=1.0)
    pm.add_agent(FixedActionAgent([6.0]), weight=2.0)

    result = pm.ensemble_action(np.zeros(4))
    # (0*1 + 6*2) / (1+2) = 12/3 = 4.0
    np.testing.assert_allclose(result, [4.0])


def test_ensemble_raises_with_no_agents():
    pm = PolicyManager.__new__(PolicyManager)
    pm._registry = None
    pm._ensemble = []

    with pytest.raises(RuntimeError):
        pm.ensemble_action(np.zeros(4))


# -----------------------------------------------------------------------
# PolicyManager.update_weights()
# -----------------------------------------------------------------------

def test_update_weights_positive_scores():
    pm = PolicyManager.__new__(PolicyManager)
    pm._registry = None
    pm._ensemble = []

    pm.add_agent(FixedActionAgent([0.0]), weight=1.0)
    pm.add_agent(FixedActionAgent([0.0]), weight=1.0)

    pm.update_weights({0: 10.0, 1: 5.0})

    weights = [w for _, w in pm._ensemble]
    assert weights[0] > weights[1]  # higher score → higher weight


def test_update_weights_negative_scores_stay_positive():
    pm = PolicyManager.__new__(PolicyManager)
    pm._registry = None
    pm._ensemble = []

    pm.add_agent(FixedActionAgent([0.0]), weight=1.0)
    pm.add_agent(FixedActionAgent([0.0]), weight=1.0)

    pm.update_weights({0: -5.0, 1: -10.0})

    weights = [w for _, w in pm._ensemble]
    assert all(w > 0 for w in weights)
    assert weights[0] > weights[1]


def test_update_weights_partial_update():
    pm = PolicyManager.__new__(PolicyManager)
    pm._registry = None
    pm._ensemble = []

    pm.add_agent(FixedActionAgent([0.0]), weight=2.0)
    pm.add_agent(FixedActionAgent([0.0]), weight=3.0)

    pm.update_weights({0: 10.0})  # only update first

    _, w0 = pm._ensemble[0]
    _, w1 = pm._ensemble[1]
    assert w1 == 3.0  # second unchanged


# -----------------------------------------------------------------------
# PolicyManager.ranked_snapshots
# -----------------------------------------------------------------------

def test_ranked_snapshots_sorted_by_score(tmp_path):
    from conftest import DummyAgent
    registry = make_registry(tmp_path)
    agent = DummyAgent()

    registry.save(agent, make_metrics(5.0, episode_id=0), make_hp())
    registry.save(agent, make_metrics(20.0, episode_id=1), make_hp())
    registry.save(agent, make_metrics(10.0, episode_id=2), make_hp())

    pm = PolicyManager(registry)
    ranked = pm.ranked_snapshots

    scores = [e["primary_score"] for e in ranked]
    assert scores == sorted(scores, reverse=True)


# -----------------------------------------------------------------------
# PolicyManagerCallback
# -----------------------------------------------------------------------

def test_callback_triggers_evolve_on_dormant(tmp_path):
    from conftest import DummyAgent
    registry = make_registry(tmp_path)
    agent = DummyAgent()

    registry.save(agent, make_metrics(100.0), make_hp())
    agent._weights["w"] = 999.0

    pm = PolicyManager(registry)
    cb = pm.as_callback(agent)

    # Simulate episode_end with low score
    cb.on_episode_end(episode_id=5, eval_metrics=make_metrics(30.0))
    # Simulate dormant trigger
    cb.on_dormant(episode_id=5)

    # Agent should have been rolled back
    assert agent._weights.get("w") == 1.0


def test_callback_no_evolve_without_score(tmp_path):
    from conftest import DummyAgent
    registry = make_registry(tmp_path)
    agent = DummyAgent()
    registry.save(agent, make_metrics(100.0), make_hp())
    agent._weights["w"] = 999.0

    pm = PolicyManager(registry)
    cb = pm.as_callback(agent)

    # No on_episode_end call — _last_score is None
    cb.on_dormant(episode_id=1)

    # No rollback should have occurred
    assert agent._weights["w"] == 999.0
