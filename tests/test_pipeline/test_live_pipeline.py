import time
import pytest
import numpy as np
from tensor_optix.pipeline.live_pipeline import LivePipeline
from tensor_optix.core.types import EpisodeData


class FakeDataSource:
    """Yields a fixed number of steps then stops."""

    def __init__(self, n_steps=20, reward=1.0):
        self._n_steps = n_steps
        self._reward = reward

    def stream(self):
        obs = np.zeros(4, dtype=np.float32)
        for i in range(self._n_steps):
            terminated = (i == self._n_steps - 1)
            yield obs, self._reward, terminated, False, {}


class ConstantActionAgent:
    def act(self, observation):
        return np.array([0])


@pytest.fixture
def live_pipeline():
    source = FakeDataSource(n_steps=30, reward=1.0)
    pipeline = LivePipeline(
        data_source=source,
        agent=ConstantActionAgent(),
        episode_boundary_fn=LivePipeline.every_n_steps(10),
        reconnect_on_disconnect=False,
    )
    return pipeline


def test_live_pipeline_yields_episodes(live_pipeline):
    live_pipeline.setup()
    gen = live_pipeline.episodes()
    episode = next(gen)
    assert isinstance(episode, EpisodeData)
    assert len(episode.rewards) > 0
    live_pipeline.teardown()


def test_episode_boundary_every_n_steps():
    source = FakeDataSource(n_steps=50)
    pipeline = LivePipeline(
        data_source=source,
        agent=ConstantActionAgent(),
        episode_boundary_fn=LivePipeline.every_n_steps(5),
        reconnect_on_disconnect=False,
    )
    pipeline.setup()
    gen = pipeline.episodes()
    episode = next(gen)
    assert len(episode.rewards) <= 5
    pipeline.teardown()


def test_episode_boundary_every_n_seconds():
    boundary = LivePipeline.every_n_seconds(0.05)
    assert boundary(0, 0.06, None) is True
    assert boundary(0, 0.04, None) is False


def test_on_done_signal_boundary():
    boundary = LivePipeline.on_done_signal()
    assert boundary(100, 100.0, None) is False


def test_is_live_is_true(live_pipeline):
    assert live_pipeline.is_live is True


def test_teardown_stops_producer(live_pipeline):
    live_pipeline.setup()
    time.sleep(0.05)
    live_pipeline.teardown()
    # Producer thread should not be alive after teardown
    if live_pipeline._producer_thread is not None:
        assert not live_pipeline._producer_thread.is_alive()


def test_set_agent_updates_agent(live_pipeline):
    class AltAgent:
        def act(self, obs):
            return np.array([1])

    live_pipeline.set_agent(AltAgent())
    assert isinstance(live_pipeline._agent, AltAgent)
