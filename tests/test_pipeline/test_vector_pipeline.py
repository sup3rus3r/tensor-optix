import pytest
import numpy as np
import gymnasium as gym
from tensor_optix.pipeline.vector_pipeline import VectorBatchPipeline
from tensor_optix.core.types import EpisodeData


class ConstantActionAgent:
    def act(self, observation):
        return 0


def _cartpole_fns(n):
    return [lambda: gym.make("CartPole-v1") for _ in range(n)]


@pytest.fixture
def vector_pipeline():
    pipeline = VectorBatchPipeline(
        env_fns=_cartpole_fns(3), agent=ConstantActionAgent(), window_size=10,
    )
    pipeline.setup()
    yield pipeline
    pipeline.teardown()


def test_episode_yields_episode_data(vector_pipeline):
    window = next(vector_pipeline.episodes())
    assert isinstance(window, EpisodeData)
    assert len(window.rewards) == 10 * 3  # window_size * n_envs


def test_metadata_records_n_envs(vector_pipeline):
    window = next(vector_pipeline.episodes())
    assert window.metadata["n_envs"] == 3


def test_final_obs_is_a_list_with_one_entry_per_env(vector_pipeline):
    window = next(vector_pipeline.episodes())
    assert isinstance(window.final_obs, list)
    assert len(window.final_obs) == 3


def test_final_obs_entry_is_none_for_env_whose_last_step_was_terminal():
    # CartPole with constant action 0 tips over quickly; a short window makes
    # it very likely at least one env's last step in the window is terminal.
    pipeline = VectorBatchPipeline(
        env_fns=_cartpole_fns(2), agent=ConstantActionAgent(), window_size=60,
    )
    pipeline.setup()
    window = next(pipeline.episodes())
    pipeline.teardown()

    n_envs = window.metadata["n_envs"]
    for i in range(n_envs):
        last_done = window.terminated[-n_envs + i] or window.truncated[-n_envs + i]
        if last_done:
            assert window.final_obs[i] is None
        else:
            assert window.final_obs[i] is not None


def test_rewards_interleaved_per_env_at_matching_stride(vector_pipeline):
    """Index t*n_envs + i must correspond to env i's timestep t -- this is
    the exact layout compute_gae's n_envs parameter assumes."""
    window = next(vector_pipeline.episodes())
    n_envs = window.metadata["n_envs"]
    # Every one of the n_envs interleaved slices should have window_size entries.
    for i in range(n_envs):
        assert len(window.rewards[i::n_envs]) == 10
