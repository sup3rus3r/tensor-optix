"""
tensor_optix.multi_agent.multi_agent_episode — the multi-agent episode
data shape.

A multi-agent window is Dict[AgentID, EpisodeData]: one ordinary,
bona-fide EpisodeData per seat. There is no separate dataclass wrapping
this -- a plain dict is the whole contract, deliberately, so every
existing BaseAgent's learn(episode_data) works completely unmodified on
one seat's entry with zero changes to EpisodeData itself. This module
exists only to name the type alias in one place.
"""
from __future__ import annotations

from typing import Dict, Hashable

from tensor_optix.core.types import EpisodeData

MultiAgentEpisodeData = Dict[Hashable, EpisodeData]
