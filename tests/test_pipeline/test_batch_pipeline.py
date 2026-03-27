import pytest
import numpy as np
import gymnasium as gym
from tensor_optix.pipeline.batch_pipeline import BatchPipeline
from tensor_optix.core.types import EpisodeData


class ConstantActionAgent:
    def act(self, observation):
        return 0


@pytest.fixture
def cartpole_env():
    env = gym.make("CartPole-v1")
    yield env
    env.close()


@pytest.fixture
def batch_pipeline(cartpole_env):
    pipeline = BatchPipeline(env=cartpole_env, agent=ConstantActionAgent(), window_size=50)
    pipeline.setup()
    return pipeline


def test_episode_yields_episode_data(batch_pipeline):
    gen = batch_pipeline.episodes()
    window = next(gen)
    assert isinstance(window, EpisodeData)
    assert len(window.rewards) == 50


def test_window_size_respected():
    env = gym.make("CartPole-v1")
    pipeline = BatchPipeline(env=env, agent=ConstantActionAgent(), window_size=30)
    pipeline.setup()
    gen = pipeline.episodes()
    window = next(gen)
    assert len(window.rewards) == 30
    pipeline.teardown()


def test_observations_shape(batch_pipeline):
    gen = batch_pipeline.episodes()
    window = next(gen)
    assert window.observations.shape == (50, 4)


def test_multiple_windows_are_continuous(batch_pipeline):
    gen = batch_pipeline.episodes()
    w1 = next(gen)
    w2 = next(gen)
    assert len(w1.rewards) == 50
    assert len(w2.rewards) == 50
    assert w2.episode_id == w1.episode_id + 1


def test_env_resets_mid_window():
    # CartPole with constant action 0 will terminate quickly
    env = gym.make("CartPole-v1")
    pipeline = BatchPipeline(env=env, agent=ConstantActionAgent(), window_size=100)
    pipeline.setup()
    gen = pipeline.episodes()
    window = next(gen)
    # Should still produce exactly 100 steps (resets internally)
    assert len(window.rewards) == 100
    pipeline.teardown()


def test_is_live_is_false(batch_pipeline):
    assert batch_pipeline.is_live is False


def test_set_agent(cartpole_env):
    pipeline = BatchPipeline(env=cartpole_env, window_size=10)
    pipeline.set_agent(ConstantActionAgent())
    pipeline.setup()
    gen = pipeline.episodes()
    window = next(gen)
    assert len(window.rewards) == 10
