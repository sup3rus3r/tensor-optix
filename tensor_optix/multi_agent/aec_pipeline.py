"""
tensor_optix.multi_agent.aec_pipeline — AECMultiAgentPipeline: drives a
PettingZoo AECEnv (turn-based, one agent acts per env.step() call).

Mirrors BatchPipeline's windowing structure, but a window here is
window_size TOTAL steps across all seats, not window_size per seat --
only one agent acts per env.step() call in AEC, so a window's steps are
naturally split across however many agents are active, same as a real
game's turn order. A seat's EpisodeData only grows on its own turns.

Reward alignment: env.last() returns the reward a seat accumulated since
its previous action, so each reward (and the terminal/truncated flag seen
on the seat's final, action=None turn) is credited back to that seat's
previously recorded step. reward[k] is therefore the outcome of action[k],
and a seat's last acted step in an episode carries the terminal reward
with terminated/truncated=True.

Known limitation: the outcome of a seat's LAST action in a window only
arrives on its next turn, which belongs to the next window. That one
reward per seat per window boundary is not credited (the step keeps
reward 0.0 and no done flag).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Generator, Hashable, Optional

import numpy as np

from tensor_optix.core.types import EpisodeData
from tensor_optix.core.reward_shape import EPISODE_RETURNS_KEY, EpisodeReturnTracker

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
        # One cross-window episode-return tracker per seat.
        self._return_trackers: Dict[Hashable, EpisodeReturnTracker] = {}
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

            # Per seat: index (within this window) of its most recent acted
            # step whose outcome has not been observed yet, or None.
            pending: Dict[Hashable, Optional[int]] = {}

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
                    pending.clear()
                    try:
                        aid = next(self._agent_iter)
                    except StopIteration:
                        break  # degenerate env with zero agents; stop this window

                raw_observation, reward, termination, truncation, info = self._env.last()

                # In AEC, env.last() reports the reward this seat accumulated
                # SINCE ITS PREVIOUS ACTION — i.e. the outcome of the step it
                # already recorded, not of the action it is about to take.
                # Credit it (and any terminal/truncated flag) back to that
                # recorded step. This keeps reward[k] aligned with action[k]
                # and delivers the terminal reward + done flag without
                # appending an un-acted-on step (which would desync agents'
                # act()-populated caches).
                prev_idx = pending.get(aid)
                if prev_idx is not None:
                    per_seat[aid]["rewards"][prev_idx] += float(reward)
                    if termination:
                        per_seat[aid]["terminated"][prev_idx] = True
                    if truncation:
                        per_seat[aid]["truncated"][prev_idx] = True

                if termination or truncation:
                    # PettingZoo requires action=None for a terminal/
                    # truncated agent's turn -- no act() call is made, so
                    # no new step is appended for this seat.
                    pending[aid] = None
                    self._env.step(None)
                    steps_taken += 1
                    continue

                observation = self._encode(aid, raw_observation)
                action = self._agents[aid].act(observation)

                per_seat[aid]["observations"].append(observation)
                per_seat[aid]["actions"].append(action)
                # Outcome is unknown until this seat's next turn — filled in
                # above when that turn arrives.
                per_seat[aid]["rewards"].append(0.0)
                per_seat[aid]["terminated"].append(False)
                per_seat[aid]["truncated"].append(False)
                per_seat[aid]["infos"].append(info)
                pending[aid] = len(per_seat[aid]["rewards"]) - 1

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
                    metadata={
                        EPISODE_RETURNS_KEY: self._return_trackers.setdefault(
                            aid, EpisodeReturnTracker()
                        ).update(
                            data["rewards"],
                            [t or tr for t, tr in zip(data["terminated"], data["truncated"])],
                        ),
                    },
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
