import numpy as np
import pytest

pettingzoo = pytest.importorskip("pettingzoo")
from pettingzoo.classic import rps_v2

from tensor_optix.multi_agent import SeatEnvView, ParallelMultiAgentPipeline, AECMultiAgentPipeline, MultiAgentRunner
from tensor_optix.factory import make_agent


def _build_seats(env):
    views, agents = {}, {}
    for aid in env.possible_agents:
        view = SeatEnvView(env, aid)
        views[aid] = view
        agents[aid] = make_agent("PPO", view, framework="torch")
    return views, agents


def _snapshot(agents):
    return {
        aid: {k: v.clone() for k, v in agents[aid]._actor.state_dict().items()}
        for aid in agents
    }


def _any_changed(before, after, aid):
    return any(not np.allclose(before[aid][k].numpy(), after[aid][k].numpy()) for k in before[aid])


def test_runner_trains_every_seat_on_parallel_env():
    env = rps_v2.parallel_env()
    env.reset()
    views, agents = _build_seats(env)
    before = _snapshot(agents)

    pipeline = ParallelMultiAgentPipeline(env=env, agents=agents, window_size=20, views=views)
    runner = MultiAgentRunner(pipeline=pipeline, agents=agents)
    runner.run(max_windows=3)

    after = _snapshot(agents)
    for aid in agents:
        assert _any_changed(before, after, aid)
    env.close()


def test_runner_trains_every_seat_on_aec_env():
    env = rps_v2.env()
    env.reset()
    views, agents = _build_seats(env)
    before = _snapshot(agents)

    pipeline = AECMultiAgentPipeline(env=env, agents=agents, window_size=20, views=views)
    runner = MultiAgentRunner(pipeline=pipeline, agents=agents)
    runner.run(max_windows=3)

    after = _snapshot(agents)
    for aid in agents:
        assert _any_changed(before, after, aid)
    env.close()


def test_runner_respects_max_windows():
    env = rps_v2.parallel_env()
    env.reset()
    views, agents = _build_seats(env)

    call_count = {"n": 0}
    pipeline = ParallelMultiAgentPipeline(env=env, agents=agents, window_size=5, views=views)
    original_episodes = pipeline.episodes

    def counting_episodes():
        for window in original_episodes():
            call_count["n"] += 1
            yield window

    pipeline.episodes = counting_episodes
    runner = MultiAgentRunner(pipeline=pipeline, agents=agents)
    runner.run(max_windows=2)

    assert call_count["n"] == 2
    env.close()
