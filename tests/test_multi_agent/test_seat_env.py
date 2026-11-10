import numpy as np
import pytest

pettingzoo = pytest.importorskip("pettingzoo")
from pettingzoo.classic import rps_v2

from tensor_optix.multi_agent import SeatEnvView
from tensor_optix.factory import make_agent


@pytest.fixture
def pz_env():
    env = rps_v2.parallel_env()
    env.reset()
    yield env
    env.close()


def test_discrete_observation_becomes_one_hot_box(pz_env):
    aid = pz_env.possible_agents[0]
    view = SeatEnvView(pz_env, aid)

    # rps_v2's observation is Discrete(4): opponent's last move or "none yet"
    assert view.observation_space.shape == (4,)
    assert view.observation_space.low.min() == 0.0
    assert view.observation_space.high.max() == 1.0


def test_encode_observation_produces_correct_one_hot(pz_env):
    aid = pz_env.possible_agents[0]
    view = SeatEnvView(pz_env, aid)

    encoded = view.encode_observation(2)
    assert encoded.shape == (4,)
    assert encoded[2] == 1.0
    assert encoded.sum() == 1.0


def test_non_discrete_observation_passes_through_unchanged():
    class FakePZEnv:
        def observation_space(self, agent_id):
            from gymnasium.spaces import Box
            return Box(-1, 1, (8,))

        def action_space(self, agent_id):
            from gymnasium.spaces import Discrete
            return Discrete(3)

    view = SeatEnvView(FakePZEnv(), "agent_0")
    raw = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], dtype=np.float32)
    encoded = view.encode_observation(raw)
    assert encoded is raw


def test_make_agent_builds_working_agent_from_seat_view(pz_env):
    aid = pz_env.possible_agents[0]
    view = SeatEnvView(pz_env, aid)
    agent = make_agent("PPO", view, framework="torch")

    obs = view.encode_observation(0)
    action = agent.act(obs)
    assert isinstance(action, int)
    assert 0 <= action < 3
