import numpy as np
import pytest

pettingzoo = pytest.importorskip("pettingzoo")
from pettingzoo.classic import rps_v2

from tensor_optix.multi_agent import SeatEnvView, ParallelMultiAgentPipeline
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
def parallel_env():
    env = rps_v2.parallel_env()
    env.reset()
    yield env
    env.close()


def test_episode_yields_one_episode_data_per_seat(parallel_env):
    views, agents = _build_seats(parallel_env)
    pipeline = ParallelMultiAgentPipeline(env=parallel_env, agents=agents, window_size=10, views=views)
    pipeline.setup()

    window = next(pipeline.episodes())
    assert set(window.keys()) == set(parallel_env.possible_agents)
    for ep in window.values():
        assert isinstance(ep, EpisodeData)


def test_window_size_matches_rounds_per_seat(parallel_env):
    views, agents = _build_seats(parallel_env)
    pipeline = ParallelMultiAgentPipeline(env=parallel_env, agents=agents, window_size=15, views=views)
    pipeline.setup()

    window = next(pipeline.episodes())
    # Parallel API: every active seat acts every round, so each seat's
    # EpisodeData length equals window_size (rounds), not window_size
    # split across seats as in AEC.
    for ep in window.values():
        assert len(ep.rewards) == 15


def test_observations_are_one_hot_encoded(parallel_env):
    views, agents = _build_seats(parallel_env)
    pipeline = ParallelMultiAgentPipeline(env=parallel_env, agents=agents, window_size=5, views=views)
    pipeline.setup()

    window = next(pipeline.episodes())
    for ep in window.values():
        assert ep.observations.shape == (5, 4)  # one-hot Discrete(4)
        # every row sums to 1 (a valid one-hot vector)
        assert np.allclose(ep.observations.sum(axis=1), 1.0)


def test_multiple_windows_are_continuous(parallel_env):
    views, agents = _build_seats(parallel_env)
    pipeline = ParallelMultiAgentPipeline(env=parallel_env, agents=agents, window_size=10, views=views)
    pipeline.setup()

    gen = pipeline.episodes()
    w1 = next(gen)
    w2 = next(gen)
    for aid in w1:
        assert len(w1[aid].rewards) == 10
        assert len(w2[aid].rewards) == 10
        assert w2[aid].episode_id == w1[aid].episode_id + 1
