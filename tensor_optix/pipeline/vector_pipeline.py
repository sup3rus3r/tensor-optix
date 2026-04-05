import numpy as np
from typing import Any, Callable, Generator, List, Optional

from tensor_optix.core.base_pipeline import BasePipeline
from tensor_optix.core.types import EpisodeData


class VectorBatchPipeline(BasePipeline):
    """
    Parallel environment pipeline using gymnasium.vector.

    Runs N environments simultaneously and collects rollouts from all of them
    in lockstep, dramatically increasing sample throughput compared to a single
    BatchPipeline. The yielded EpisodeData contains observations and actions
    from ALL envs interleaved: shape [window_size * n_envs, ...].

    Usage:
        import gymnasium as gym
        from tensor_optix.pipeline.vector_pipeline import VectorBatchPipeline

        env_fns = [lambda: gym.make("CartPole-v1")] * 8   # 8 parallel envs

        pipeline = VectorBatchPipeline(env_fns=env_fns, window_size=256)

        # Or use AsyncVectorEnv for CPU-bound envs (true parallelism):
        pipeline = VectorBatchPipeline(
            env_fns=env_fns,
            window_size=256,
            async_envs=True,
        )

    The PPO agent's act() will be called with a BATCHED observation of shape
    [n_envs, obs_dim] at each parallel step. Agents (TFPPOAgent etc.) handle
    single observations; for vector envs you should subclass and override act()
    to accept and return batched arrays, OR use window_size * n_envs steps with
    per-env act() calls (the default behavior here: we iterate per-env).

    window_size: steps collected PER ENV per yielded EpisodeData.
                 Total steps per yield = window_size * n_envs.
    async_envs:  use AsyncVectorEnv (subprocess) instead of SyncVectorEnv.

    Note: each env resets automatically when it terminates/truncates. Episode
    boundaries are tracked per-env and merged into the flat yielded arrays.
    """

    def __init__(
        self,
        env_fns: List[Callable],
        agent=None,
        window_size: int = 256,
        async_envs: bool = False,
    ):
        self._env_fns    = env_fns
        self._n_envs     = len(env_fns)
        self._agent      = agent
        self._window_size = window_size
        self._async_envs  = async_envs
        self._vec_env     = None
        self._window_counter = 0

    def set_agent(self, agent) -> None:
        self._agent = agent

    def setup(self) -> None:
        import gymnasium as gym
        if self._async_envs:
            self._vec_env = gym.vector.AsyncVectorEnv(self._env_fns)
        else:
            self._vec_env = gym.vector.SyncVectorEnv(self._env_fns)

    def episodes(self) -> Generator[EpisodeData, None, None]:
        """
        Yield EpisodeData windows collected from all parallel envs.

        At each step, act() is called once PER ENV with each env's current obs
        (single obs, not batched). Results are concatenated across envs and
        steps into flat arrays of length window_size * n_envs.
        """
        obs_vec, _ = self._vec_env.reset()   # [n_envs, obs_dim]

        while True:
            all_obs       = []
            all_actions   = []
            all_rewards   = []
            all_terminated = []
            all_truncated  = []
            all_infos     = []

            for _ in range(self._window_size):
                # Collect one action per env using the agent's act()
                step_actions = np.array([
                    self._agent.act(obs_vec[i]) for i in range(self._n_envs)
                ])

                next_obs_vec, reward_vec, term_vec, trunc_vec, info_vec = \
                    self._vec_env.step(step_actions)

                for i in range(self._n_envs):
                    all_obs.append(obs_vec[i])
                    all_actions.append(step_actions[i])
                    all_rewards.append(float(reward_vec[i]))
                    all_terminated.append(bool(term_vec[i]))
                    all_truncated.append(bool(trunc_vec[i]))
                    all_infos.append(info_vec[i] if isinstance(info_vec, (list, tuple))
                                     else {})

                obs_vec = next_obs_vec

            yield EpisodeData(
                observations=np.array(all_obs),
                actions=np.array(all_actions),
                rewards=all_rewards,
                terminated=all_terminated,
                truncated=all_truncated,
                infos=all_infos,
                episode_id=self._window_counter,
            )
            self._window_counter += 1

    def teardown(self) -> None:
        if self._vec_env is not None:
            self._vec_env.close()

    @property
    def is_live(self) -> bool:
        return False

    @property
    def n_envs(self) -> int:
        return self._n_envs
