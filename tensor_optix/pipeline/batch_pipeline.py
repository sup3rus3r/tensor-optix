import time
import numpy as np
from typing import Any, Generator, Optional
from tensor_optix.core.base_pipeline import BasePipeline, EpisodeBoundaryFn
from tensor_optix.core.types import EpisodeData


class BatchPipeline(BasePipeline):
    """
    Continuous stepping pipeline against a Gymnasium-compatible environment.

    Does NOT reset the env per window. Steps continuously and yields fixed-size
    windows of experience. The env resets automatically only when terminated
    or truncated — training never stops between windows.

    window_size: number of steps per yielded EpisodeData window.
                 Default 200. This is the unit of training, not an env episode.

    DORMANT state = model is trained (can't improve further).
    The loop controller owns when to stop — not the env's done flag.

    Uses Gymnasium API:
        env.reset() -> (obs, info)
        env.step(action) -> (obs, reward, terminated, truncated, info)
    """

    def __init__(
        self,
        env: Any,
        agent=None,
        window_size: int = 200,
    ):
        self._env = env
        self._agent = agent
        self._window_size = window_size
        self._window_counter = 0
        self._obs = None
        self._needs_reset = True

    def set_agent(self, agent) -> None:
        self._agent = agent

    def setup(self) -> None:
        self._needs_reset = True

    def episodes(self) -> Generator[EpisodeData, None, None]:
        while True:
            if self._needs_reset:
                self._obs, _ = self._env.reset()
                self._needs_reset = False

            observations = []
            actions = []
            rewards = []
            terminated_flags = []
            truncated_flags = []
            infos = []

            for _ in range(self._window_size):
                obs = self._obs
                observations.append(obs)
                action = self._agent.act(obs)
                actions.append(action)

                next_obs, reward, terminated, truncated, info = self._env.step(action)
                rewards.append(float(reward))
                terminated_flags.append(bool(terminated))
                truncated_flags.append(bool(truncated))
                infos.append(info)

                if terminated or truncated:
                    # Env done mid-window — reset and continue filling the window
                    self._obs, _ = self._env.reset()
                else:
                    self._obs = next_obs

            yield EpisodeData(
                observations=np.array(observations),
                actions=np.array(actions),
                rewards=rewards,
                terminated=terminated_flags,
                truncated=truncated_flags,
                infos=infos,
                episode_id=self._window_counter,
            )
            self._window_counter += 1

    def teardown(self) -> None:
        self._env.close()

    @property
    def is_live(self) -> bool:
        return False
