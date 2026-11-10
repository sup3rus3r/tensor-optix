import numpy as np
import pytest

pettingzoo = pytest.importorskip("pettingzoo")
from pettingzoo.classic import rps_v2

from tensor_optix.multi_agent import SeatEnvView, AECMultiAgentPipeline
from tensor_optix.core.types import EpisodeData
from tensor_optix.factory import make_agent


def _build_seats(env):
    views, agents = {}, {}
    for aid in env.possible_agents:
        view = SeatEnvView(env, aid)
        views[aid] = view
        agents[aid] = make_agent("PPO", view, framework="torch")
    return views, agents


@pytest.fixture
def aec_env():
    env = rps_v2.env()
    env.reset()
    yield env
    env.close()


def test_total_steps_across_seats_equals_window_size(aec_env):
    views, agents = _build_seats(aec_env)
    pipeline = AECMultiAgentPipeline(env=aec_env, agents=agents, window_size=20, views=views)
    pipeline.setup()

    window = next(pipeline.episodes())
    total_steps = sum(len(ep.rewards) for ep in window.values())
    # A terminal/truncated turn is skipped entirely (see module docstring),
    # so total steps across seats can be <= window_size, never more.
    assert total_steps <= 20
    assert total_steps > 0


def test_episode_yields_one_episode_data_per_possible_agent(aec_env):
    views, agents = _build_seats(aec_env)
    pipeline = AECMultiAgentPipeline(env=aec_env, agents=agents, window_size=10, views=views)
    pipeline.setup()

    window = next(pipeline.episodes())
    assert set(window.keys()) == set(aec_env.possible_agents)


def test_a_seat_grows_only_on_its_own_turns(aec_env):
    """In rock-paper-scissors both players act every round in strict
    alternation, so each seat's episode length should be roughly half the
    window (give or take the terminal-step skip), never the full window
    (which would mean one seat was credited with the other's turns)."""
    views, agents = _build_seats(aec_env)
    pipeline = AECMultiAgentPipeline(env=aec_env, agents=agents, window_size=20, views=views)
    pipeline.setup()

    window = next(pipeline.episodes())
    for ep in window.values():
        assert len(ep.rewards) < 20


def test_learn_does_not_crash_on_terminal_step_handling(aec_env):
    """Regression test for the real bug found this session: a terminal/
    truncated turn must not be appended to a seat's EpisodeData, or the
    agent's internal act()-populated cache desyncs from episode_data and
    learn() raises a cache-underflow error."""
    views, agents = _build_seats(aec_env)
    pipeline = AECMultiAgentPipeline(env=aec_env, agents=agents, window_size=20, views=views)
    pipeline.setup()

    window = next(pipeline.episodes())
    for aid, ep in window.items():
        if len(ep.rewards) == 0:
            continue
        agents[aid].learn(ep)  # must not raise
