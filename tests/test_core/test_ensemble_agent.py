import pytest
import numpy as np
from tensor_optix.core.ensemble_agent import EnsembleAgent
from tensor_optix.core.policy_manager import PolicyManager
from tensor_optix.core.types import EpisodeData, HyperparamSet, EvalMetrics


class FixedActionAgent:
    """Returns a fixed action."""
    def __init__(self, action):
        self._action = action
        self._hyperparams = HyperparamSet(params={"lr": 1e-3}, episode_id=0)
        self.learn_calls = 0
        self.saved_path = None
        self.loaded_path = None

    def act(self, obs):
        return np.array(self._action, dtype=float)

    def learn(self, episode_data):
        self.learn_calls += 1
        return {"loss": 0.0}

    def get_hyperparams(self):
        return self._hyperparams.copy()

    def set_hyperparams(self, hp):
        self._hyperparams = hp.copy()

    def save_weights(self, path):
        self.saved_path = path

    def load_weights(self, path):
        self.loaded_path = path


def make_pm_with_agents(*actions):
    """Create a PolicyManager loaded with FixedActionAgents."""
    pm = PolicyManager.__new__(PolicyManager)
    pm._registry = None
    pm._ensemble = []
    agents = []
    for action in actions:
        a = FixedActionAgent(action)
        pm.add_agent(a, weight=1.0)
        agents.append(a)
    return pm, agents


def make_episode():
    return EpisodeData(
        observations=np.zeros((3, 4)),
        actions=np.zeros(3, dtype=int),
        rewards=[1.0, 1.0, 1.0],
        terminated=[False, False, True],
        truncated=[False, False, False],
        infos=[{}, {}, {}],
        episode_id=0,
    )


# -----------------------------------------------------------------------
# act() delegates to PolicyManager.ensemble_action()
# -----------------------------------------------------------------------

def test_act_single_agent():
    pm, [agent] = make_pm_with_agents([7.0])
    ensemble = EnsembleAgent(pm, primary_agent=agent)
    result = ensemble.act(np.zeros(4))
    np.testing.assert_allclose(result, [7.0])


def test_act_two_agents_averaged():
    pm, [a, b] = make_pm_with_agents([0.0], [10.0])
    ensemble = EnsembleAgent(pm, primary_agent=a)
    result = ensemble.act(np.zeros(4))
    np.testing.assert_allclose(result, [5.0])


# -----------------------------------------------------------------------
# learn / hyperparams / weights delegate to primary_agent
# -----------------------------------------------------------------------

def test_learn_delegates_to_primary():
    pm, [a, b] = make_pm_with_agents([0.0], [1.0])
    ensemble = EnsembleAgent(pm, primary_agent=a)
    ep = make_episode()
    ensemble.learn(ep)
    assert a.learn_calls == 1
    assert b.learn_calls == 0


def test_get_hyperparams_from_primary():
    pm, [a] = make_pm_with_agents([0.0])
    ensemble = EnsembleAgent(pm, primary_agent=a)
    hp = ensemble.get_hyperparams()
    assert hp.params["lr"] == 1e-3


def test_set_hyperparams_on_primary():
    pm, [a] = make_pm_with_agents([0.0])
    ensemble = EnsembleAgent(pm, primary_agent=a)
    new_hp = HyperparamSet(params={"lr": 5e-4}, episode_id=1)
    ensemble.set_hyperparams(new_hp)
    assert a._hyperparams.params["lr"] == 5e-4


def test_save_weights_delegates_to_primary():
    pm, [a] = make_pm_with_agents([0.0])
    ensemble = EnsembleAgent(pm, primary_agent=a)
    ensemble.save_weights("/some/path")
    assert a.saved_path == "/some/path"


def test_load_weights_delegates_to_primary():
    pm, [a] = make_pm_with_agents([0.0])
    ensemble = EnsembleAgent(pm, primary_agent=a)
    ensemble.load_weights("/some/path")
    assert a.loaded_path == "/some/path"


# -----------------------------------------------------------------------
# policy_manager property
# -----------------------------------------------------------------------

def test_policy_manager_property():
    pm, [a] = make_pm_with_agents([0.0])
    ensemble = EnsembleAgent(pm, primary_agent=a)
    assert ensemble.policy_manager is pm
