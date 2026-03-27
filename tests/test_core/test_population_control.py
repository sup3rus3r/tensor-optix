import pytest
import numpy as np
from tensor_optix.core.policy_manager import PolicyManager
from tensor_optix.core.types import EvalMetrics


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

class FixedAgent:
    def __init__(self, action=0.0):
        self._action = action

    def act(self, obs):
        return np.array([self._action], dtype=float)


def make_pm(*weights):
    """Create a PolicyManager with N agents at given weights."""
    pm = PolicyManager.__new__(PolicyManager)
    pm._registry = None
    pm._ensemble = []
    pm._score_history = {}
    pm._score_window = 10
    pm._spawn_count = 0
    pm._prune_count = 0
    pm._current_regime = None
    for w in weights:
        pm._ensemble.append((FixedAgent(), w))
    return pm


def make_metrics(scores):
    return [EvalMetrics(primary_score=s, metrics={}, episode_id=i) for i, s in enumerate(scores)]


# -----------------------------------------------------------------------
# prune()
# -----------------------------------------------------------------------

def test_prune_removes_lowest_weight_agent():
    pm = make_pm(5.0, 1.0, 3.0)
    removed = pm.prune(bottom_k=1)
    assert len(removed) == 1
    assert len(pm._ensemble) == 2
    weights = [w for _, w in pm._ensemble]
    assert 1.0 not in weights


def test_prune_bottom_k_two():
    pm = make_pm(5.0, 1.0, 3.0, 0.5)
    pm.prune(bottom_k=2)
    assert len(pm._ensemble) == 2
    weights = [w for _, w in pm._ensemble]
    assert sorted(weights, reverse=True) == [5.0, 3.0]


def test_prune_returns_removed_agents():
    agents = [FixedAgent(i) for i in range(3)]
    pm = PolicyManager.__new__(PolicyManager)
    pm._registry = None
    pm._ensemble = [(agents[0], 1.0), (agents[1], 0.1), (agents[2], 3.0)]
    pm._score_history = {}
    pm._score_window = 10
    pm._spawn_count = 0
    pm._prune_count = 0
    pm._current_regime = None

    removed = pm.prune(bottom_k=1)
    assert removed == [agents[1]]


def test_prune_noop_when_too_small():
    pm = make_pm(1.0)
    removed = pm.prune(bottom_k=1)
    assert removed == []
    assert len(pm._ensemble) == 1


def test_prune_remaps_score_history():
    pm = make_pm(5.0, 1.0, 3.0)
    from collections import deque
    pm._score_history[0] = deque([10.0], maxlen=10)
    pm._score_history[1] = deque([2.0], maxlen=10)  # will be pruned
    pm._score_history[2] = deque([8.0], maxlen=10)

    pm.prune(bottom_k=1)

    # After pruning index 1, old index 2 becomes new index 1
    assert 0 in pm._score_history
    assert 1 in pm._score_history
    assert list(pm._score_history[0]) == [10.0]
    assert list(pm._score_history[1]) == [8.0]


def test_prune_increments_prune_count():
    pm = make_pm(1.0, 2.0, 3.0)
    pm.prune(bottom_k=2)
    assert pm._prune_count == 2


# -----------------------------------------------------------------------
# boost()
# -----------------------------------------------------------------------

def test_boost_multiplies_agent_weight():
    agents = [FixedAgent(), FixedAgent()]
    pm = PolicyManager.__new__(PolicyManager)
    pm._registry = None
    pm._ensemble = [(agents[0], 1.0), (agents[1], 1.0)]
    pm._score_history = {}
    pm._score_window = 10
    pm._spawn_count = 0
    pm._prune_count = 0
    pm._current_regime = None

    pm.boost(agents[0], factor=3.0)

    assert pm._ensemble[0][1] == pytest.approx(3.0)
    assert pm._ensemble[1][1] == pytest.approx(1.0)


def test_boost_only_affects_target_agent():
    agents = [FixedAgent(), FixedAgent(), FixedAgent()]
    pm = PolicyManager.__new__(PolicyManager)
    pm._registry = None
    pm._ensemble = [(a, 1.0) for a in agents]
    pm._score_history = {}
    pm._score_window = 10
    pm._spawn_count = 0
    pm._prune_count = 0
    pm._current_regime = None

    pm.boost(agents[1], factor=4.0)

    assert pm._ensemble[0][1] == pytest.approx(1.0)
    assert pm._ensemble[1][1] == pytest.approx(4.0)
    assert pm._ensemble[2][1] == pytest.approx(1.0)


def test_boost_unknown_agent_is_noop():
    pm = make_pm(1.0, 2.0)
    stranger = FixedAgent()
    pm.boost(stranger, factor=5.0)  # should not raise
    weights = [w for _, w in pm._ensemble]
    assert weights == [1.0, 2.0]


# -----------------------------------------------------------------------
# set_regime()
# -----------------------------------------------------------------------

def test_set_regime_stores_regime():
    pm = make_pm(1.0)
    pm.set_regime("volatile")
    assert pm._current_regime == "volatile"


def test_set_regime_updates_on_change():
    pm = make_pm(1.0)
    pm.set_regime("ranging")
    pm.set_regime("trending")
    assert pm._current_regime == "trending"


# -----------------------------------------------------------------------
# adaptive_noise_scale()
# -----------------------------------------------------------------------

def test_adaptive_noise_returns_max_on_plateau():
    pm = make_pm(1.0)
    flat = make_metrics([10.0, 10.0, 10.0, 10.0, 10.0])
    scale = pm.adaptive_noise_scale(flat, min_scale=0.001, max_scale=0.1)
    assert scale == pytest.approx(0.1)


def test_adaptive_noise_returns_low_on_strong_improvement():
    pm = make_pm(1.0)
    improving = make_metrics([10.0, 15.0, 20.0, 25.0, 30.0])
    scale = pm.adaptive_noise_scale(improving, min_scale=0.001, max_scale=0.1)
    assert scale < 0.05


def test_adaptive_noise_returns_max_on_insufficient_history():
    pm = make_pm(1.0)
    scale = pm.adaptive_noise_scale(make_metrics([10.0, 11.0]), max_scale=0.1)
    assert scale == pytest.approx(0.1)


def test_adaptive_noise_stays_within_bounds():
    pm = make_pm(1.0)
    declining = make_metrics([30.0, 20.0, 10.0, 5.0, 1.0])
    scale = pm.adaptive_noise_scale(declining, min_scale=0.001, max_scale=0.1)
    assert 0.001 <= scale <= 0.1


# -----------------------------------------------------------------------
# status()
# -----------------------------------------------------------------------

def test_status_reflects_ensemble_size():
    pm = make_pm(1.0, 2.0, 3.0)
    s = pm.status()
    assert s["ensemble_size"] == 3


def test_status_agents_have_expected_keys():
    pm = make_pm(1.5)
    agent_info = pm.status()["agents"][0]
    assert "index" in agent_info
    assert "weight" in agent_info
    assert "mean_score" in agent_info
    assert "recent_scores" in agent_info


def test_status_reflects_regime():
    pm = make_pm(1.0)
    pm.set_regime("trending")
    assert pm.status()["regime"] == "trending"


def test_status_tracks_spawn_and_prune_counts():
    pm = make_pm(1.0, 2.0, 3.0)
    pm._spawn_count = 3
    pm.prune(bottom_k=1)
    s = pm.status()
    assert s["spawn_count"] == 3
    assert s["prune_count"] == 1


def test_status_mean_score_none_without_history():
    pm = make_pm(1.0)
    assert pm.status()["agents"][0]["mean_score"] is None


def test_status_mean_score_from_history():
    pm = make_pm(1.0)
    from collections import deque
    pm._score_history[0] = deque([4.0, 6.0], maxlen=10)
    assert pm.status()["agents"][0]["mean_score"] == pytest.approx(5.0)
