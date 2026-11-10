"""
tensor_optix.multi_agent.parallel_pipeline — ParallelMultiAgentPipeline:
drives a PettingZoo ParallelEnv, fanning out one EpisodeData per seat.

Mirrors tensor_optix.pipeline.batch_pipeline.BatchPipeline's windowing
structure exactly, just repeated per agent_id instead of once. Each
yielded value is a Dict[AgentID, EpisodeData] -- one bona fide, ordinary
EpisodeData per seat, so every existing BaseAgent's learn() works
unmodified on it.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Generator, Hashable, Optional

import numpy as np

from tensor_optix.core.types import EpisodeData

logger = logging.getLogger(__name__)


class ParallelMultiAgentPipeline:
    """
    Drives a PettingZoo ParallelEnv for window_size rounds per window.

    Each round, every currently-active agent_id (env.agents) acts via its
    own agent's act(), all actions are submitted in one env.step(actions)
    call, and the round's observation/action/reward/done is appended to
    that seat's running lists -- exactly how BatchPipeline accumulates its
    single stream, just once per seat per round.

    A seat that leaves mid-window (eliminated, env.agents shrinks) simply
    stops accumulating further rounds; its partial-window EpisodeData is
    still yielded with whatever length it reached, so one seat's early
    elimination doesn't block or truncate the whole window for the others.
    A seat that never appears in a window at all (e.g. already eliminated
    before the window started) yields an EpisodeData with T=0 -- callers
    must skip calling .learn() on an empty EpisodeData for that seat (see
    MultiAgentRunner.run(), which already does this).

    window_size: rounds collected per yielded window. Each round is one
                 env.step() call (not one step per seat -- Parallel steps
                 all active seats simultaneously).
    """

    def __init__(
        self,
        env: Any,
        agents: Dict[Hashable, Any],
        window_size: int = 200,
        views: Optional[Dict[Hashable, Any]] = None,
    ) -> None:
        self._env = env
        self._agents = agents
        self._window_size = window_size
        self._window_counter = 0
        self._obs: Optional[Dict[Hashable, Any]] = None
        self._needs_reset = True
        # Optional Dict[AgentID, SeatEnvView] -- when given, each seat's raw
        # observation is passed through views[aid].encode_observation()
        # before reaching that seat's agent.act() or EpisodeData. Needed
        # for a Discrete observation space (see seat_env.py's module
        # docstring); a no-op passthrough for Box/Dict observations.
        self._views = views or {}

    def set_agents(self, agents: Dict[Hashable, Any]) -> None:
        self._agents = agents

    def _encode(self, aid: Hashable, obs: Any) -> Any:
        view = self._views.get(aid)
        return view.encode_observation(obs) if view is not None else obs

    def setup(self) -> None:
        self._needs_reset = True

    def episodes(self) -> Generator[Dict[Hashable, EpisodeData], None, None]:
        while True:
            if self._needs_reset:
                self._obs, _ = self._env.reset()
                self._needs_reset = False

            per_seat: Dict[Hashable, dict] = {
                aid: {
                    "observations": [], "actions": [], "rewards": [],
                    "terminated": [], "truncated": [], "infos": [],
                }
                for aid in self._env.possible_agents
            }

            for _round_idx in range(self._window_size):
                active = list(self._env.agents)
                if not active:
                    # Episode ended for every seat mid-window -- reset and
                    # continue filling the window with a fresh episode,
                    # same "don't stop between windows" philosophy as
                    # BatchPipeline.
                    self._obs, _ = self._env.reset()
                    active = list(self._env.agents)
                    if not active:
                        break  # degenerate env with zero agents on reset; stop this window

                actions = {}
                for aid in active:
                    obs = self._encode(aid, self._obs[aid])
                    action = self._agents[aid].act(obs)
                    actions[aid] = action
                    per_seat[aid]["observations"].append(obs)
                    per_seat[aid]["actions"].append(action)

                next_obs, rewards, terminations, truncations, infos = self._env.step(actions)

                for aid in active:
                    per_seat[aid]["rewards"].append(float(rewards.get(aid, 0.0)))
                    per_seat[aid]["terminated"].append(bool(terminations.get(aid, False)))
                    per_seat[aid]["truncated"].append(bool(truncations.get(aid, False)))
                    per_seat[aid]["infos"].append(infos.get(aid, {}))

                self._obs = next_obs

            window: Dict[Hashable, EpisodeData] = {}
            for aid, data in per_seat.items():
                last_done = (
                    bool(data["terminated"][-1] or data["truncated"][-1])
                    if data["terminated"] else True
                )
                window[aid] = EpisodeData(
                    observations=np.array(data["observations"]) if data["observations"] else np.array([]),
                    actions=np.array(data["actions"]) if data["actions"] else np.array([]),
                    rewards=data["rewards"],
                    terminated=data["terminated"],
                    truncated=data["truncated"],
                    infos=data["infos"],
                    episode_id=self._window_counter,
                    final_obs=None if last_done else self._encode(aid, self._obs.get(aid)) if aid in self._obs else None,
                )
            yield window
            self._window_counter += 1

    def teardown(self) -> None:
        self._env.close()

    @property
    def is_live(self) -> bool:
        return False
