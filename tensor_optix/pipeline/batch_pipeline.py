import logging
import numpy as np

logger = logging.getLogger(__name__)
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

    EpisodeData fields populated by this pipeline:
        episode_starts: list of step indices where a new episode begins within
                        the window. Index 0 is always included. Use these instead
                        of scanning dones manually.
        final_obs:      the observation immediately after the last step. None when
                        the window ended at a terminal state. On-policy agents
                        (TFPPOAgent, TorchPPOAgent) use this to bootstrap V(s_T)
                        correctly when the window ends mid-episode.

    Warning — gym.Env method name collision:
        gymnasium wraps env instances in a VectorEnv (or similar) that exposes
        close(), step(), reset(), render(), and seed() as its own methods. If your
        env class defines an *attribute* with any of those names, it will shadow
        the wrapper's method and cause confusing AttributeErrors or silent
        misbehaviour at teardown or during collection. Rename any conflicting env
        attributes before passing the env to BatchPipeline.
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
            episode_start_indices = [0]

            for step_idx in range(self._window_size):
                obs = self._obs
                observations.append(obs)
                action = self._agent.act(obs)
                actions.append(action)

                try:
                    next_obs, reward, terminated, truncated, info = self._env.step(action)
                except Exception as e:
                    # Physics engines (e.g. Box2D in BipedalWalker) can raise
                    # AssertionErrors on degenerate states (zero-length lidar rays,
                    # unstable joint configurations). Treat as episode termination
                    # and reset — the window continues with a fresh episode.
                    logger.warning("env.step() raised %s: %s — treating as termination", type(e).__name__, e)
                    terminated = True
                    truncated = False
                    reward = 0.0
                    info = {}
                    next_obs = obs  # placeholder, overwritten by reset below

                rewards.append(float(reward))
                terminated_flags.append(bool(terminated))
                truncated_flags.append(bool(truncated))
                infos.append(info)

                if terminated or truncated:
                    # Env done mid-window — reset and continue filling the window
                    self._obs, _ = self._env.reset()
                    if step_idx + 1 < self._window_size:
                        episode_start_indices.append(step_idx + 1)
                else:
                    self._obs = next_obs

            last_done = terminated_flags[-1] or truncated_flags[-1]
            yield EpisodeData(
                observations=np.array(observations),
                actions=np.array(actions),
                rewards=rewards,
                terminated=terminated_flags,
                truncated=truncated_flags,
                infos=infos,
                episode_id=self._window_counter,
                episode_starts=episode_start_indices,
                final_obs=None if last_done else self._obs,
            )
            self._window_counter += 1

    def teardown(self) -> None:
        self._env.close()

    @property
    def is_live(self) -> bool:
        return False
