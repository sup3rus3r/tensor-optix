"""
tensor_optix.multi_agent.seat_env — SeatEnvView: a Gym-env-shaped wrapper
around one agent_id's slice of a PettingZoo env.

make_agent() (tensor_optix/factory.py) reads env.observation_space and
env.action_space as plain attributes. PettingZoo's AECEnv/ParallelEnv
expose these as METHODS taking an agent id instead
(env.observation_space(agent) / env.action_space(agent)), so make_agent()
cannot be called directly on a PettingZoo env. SeatEnvView resolves one
seat's spaces once at construction and exposes them as the plain
attributes make_agent() expects -- the only integration point with
make_agent(); factory.py itself needs no changes.

make_agent() also rejects gymnasium.spaces.Discrete observation spaces
outright (it only accepts Box and Dict) -- a real, separate gap in
make_agent() itself, confirmed not already fixed by the earlier non-flat-
observation-space upgrade (which added Box-image and Dict support, never
Discrete). Discrete observations are common in simple/classic games
(e.g. PettingZoo's rps_v2, whose observation is a single Discrete(4)
value: the opponent's last move or "none yet"). Rather than touch
factory.py (explicitly out of scope for this plan), SeatEnvView presents
a Discrete observation space to make_agent() as a one-hot-encoded flat
Box(n,) instead, and encode_observation() converts real observations to
match on the fly. This keeps factory.py and the encoder system completely
untouched; a general Discrete-observation fix to make_agent() itself, if
wanted, is a separate, later piece of work.
"""
from __future__ import annotations

from typing import Any, Hashable

import numpy as np
from gymnasium.spaces import Box, Discrete


class SeatEnvView:
    """
    A Gym-env-shaped view of one agent_id's observation/action space in a
    PettingZoo (AEC or Parallel) env, so make_agent() can build that
    seat's agent unmodified:

        view = SeatEnvView(pz_env, agent_id)
        agent = make_agent(view, algorithm="PPO")
        action = agent.act(view.encode_observation(raw_obs))

    Spaces are resolved once here via the PettingZoo env's
    observation_space(agent)/action_space(agent) methods and cached as
    plain attributes -- PettingZoo's own docs state both methods must
    return the same value for the same agent id on every call, so
    resolving once at construction is safe and avoids re-querying the env
    on every make_agent() internal access.

    A Discrete observation space is presented to make_agent() as a
    one-hot Box(n,) (self.observation_space), and encode_observation()
    converts a raw Discrete sample (a bare int) to the matching one-hot
    vector -- see module docstring for why this lives here rather than
    in factory.py. Any other observation space type (Box, Dict) is passed
    straight through unchanged; encode_observation() is a no-op for those.
    """

    def __init__(self, pz_env: Any, agent_id: Hashable) -> None:
        self.agent_id = agent_id
        raw_obs_space = pz_env.observation_space(agent_id)
        self.action_space = pz_env.action_space(agent_id)

        self._is_discrete_obs = isinstance(raw_obs_space, Discrete)
        if self._is_discrete_obs:
            self._discrete_n = int(raw_obs_space.n)
            self.observation_space = Box(low=0.0, high=1.0, shape=(self._discrete_n,), dtype=np.float32)
        else:
            self.observation_space = raw_obs_space

    def encode_observation(self, raw_observation: Any) -> Any:
        """Convert one raw observation from the underlying PettingZoo env
        into whatever self.observation_space promises -- a one-hot vector
        for a Discrete space, unchanged for anything else."""
        if not self._is_discrete_obs:
            return raw_observation
        one_hot = np.zeros(self._discrete_n, dtype=np.float32)
        one_hot[int(raw_observation)] = 1.0
        return one_hot
