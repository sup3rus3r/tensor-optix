"""
tensor_optix.multi_agent.aec_pipeline — AECMultiAgentPipeline: drives a
PettingZoo AECEnv (turn-based, one agent acts per env.step() call).

Mirrors BatchPipeline's windowing structure, but a window here is
window_size TOTAL steps across all seats, not window_size per seat --
only one agent acts per env.step() call in AEC, so a window's steps are
naturally split across however many agents are active, same as a real
game's turn order. A seat's EpisodeData only grows on its own turns.

Known limitation: a seat's final terminal/truncated turn (PettingZoo
requires action=None there, so no act() call is made) is NOT included in
that seat's EpisodeData -- every existing BaseAgent's learn() assumes its
internal act()-populated cache has at least as many entries as
episode_data.rewards, and appending an un-acted-on step would desync the
two. This means a seat's final reward for an episode that ends within a
window is not part of its training signal for that window. Not solved in
this pass; flagged rather than silently dropped.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Generator, Hashable, Optional

import numpy as np

from tensor_optix.core.types import EpisodeData

logger = logging.getLogger(__name__)


class AECMultiAgentPipeline:
    """
    Drives a PettingZoo AECEnv for window_size total steps per window
    (not window_size per seat -- see module docstring).

    Each step: read env.last() for the current agent_selection, call that
    one agent's act(), step the env. Only that seat's running lists grow
    this step; other seats get no new entry until their own next turn.

    A seat that never gets a turn within a given window yields an
    EpisodeData with T=0 for that window -- callers must skip calling
    .learn() on an empty EpisodeData for that seat (MultiAgentRunner
    already does this).

    window_size: total env.step() calls collected per yielded window,
                 split across whichever seats are active during it.
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
        self._needs_reset = True
        self._agent_iter = None
        # See ParallelMultiAgentPipeline's constructor docstring for why
        # this exists -- same Discrete-observation one-hot encoding need.
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
                self._env.reset()
                self._agent_iter = iter(self._env.agent_iter())
                self._needs_reset = False

            per_seat: Dict[Hashable, dict] = {
                aid: {
                    "observations": [], "actions": [], "rewards": [],
                    "terminated": [], "truncated": [], "infos": [],
                }
                for aid in self._env.possible_agents
            }

            steps_taken = 0
            while steps_taken < self._window_size:
                try:
                    aid = next(self._agent_iter)
                except StopIteration:
                    # Episode ended for every seat -- reset and keep
                    # filling the window with a fresh episode, same
                    # "don't stop between windows" philosophy as
                    # BatchPipeline/ParallelMultiAgentPipeline.
                    self._env.reset()
                    self._agent_iter = iter(self._env.agent_iter())
                    try:
                        aid = next(self._agent_iter)
                    except StopIteration:
                        break  # degenerate env with zero agents; stop this window

                raw_observation, reward, termination, truncation, info = self._env.last()

                if termination or truncation:
                    # PettingZoo requires action=None for a terminal/
                    # truncated agent's turn -- no act() call is made, so
                    # this step must NOT be appended to the seat's running
                    # lists. Every existing BaseAgent's learn() assumes its
                    # internal act()-populated cache (built only by real
                    # act() calls) has at least as many entries as
                    # episode_data.rewards; appending an un-acted-on step
                    # here would silently desync the two and crash deep
                    # inside learn() ("cache underflow"). This seat's final
                    # terminal reward is therefore not included in its
                    # training signal for this window -- a real, visible
                    # limitation of this first version, not solved by
                    # reaching into each agent class's private cache.
                    self._env.step(None)
                    steps_taken += 1
                    continue

                observation = self._encode(aid, raw_observation)
                action = self._agents[aid].act(observation)

                per_seat[aid]["observations"].append(observation)
                per_seat[aid]["actions"].append(action)
                per_seat[aid]["rewards"].append(float(reward))
                per_seat[aid]["terminated"].append(bool(termination))
                per_seat[aid]["truncated"].append(bool(truncation))
                per_seat[aid]["infos"].append(info)

                self._env.step(action)
                steps_taken += 1

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
                    final_obs=None if last_done else None,
                    # AEC has no per-seat "next observation immediately
                    # after the window" the way Parallel's step() return
                    # does -- a seat's next own observation only arrives
                    # on its next turn, which may be in a future window.
                    # final_obs is left None (bootstrap-from-zero) rather
                    # than fabricated; a GAE bootstrap correction for AEC
                    # mid-episode windows is a known limitation, not
                    # solved in this pass.
                )
            yield window
            self._window_counter += 1

    def teardown(self) -> None:
        self._env.close()

    @property
    def is_live(self) -> bool:
        return False
