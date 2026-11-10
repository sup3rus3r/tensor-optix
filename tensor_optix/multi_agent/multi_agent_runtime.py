"""
tensor_optix.multi_agent.multi_agent_runtime — MultiAgentRunner: the
smallest possible multi-agent loop.

Deliberately NOT a multi-agent LoopController. No convergence detection,
no checkpointing, no rollback across seats -- those are real, separate
design questions (what does DORMANT mean when 3 of 5 seats are still
improving and 2 have plateaued?) not solved here. A caller wanting the
RL core's full convergence/checkpoint machinery for one seat wires up an
ordinary single-agent LoopController around that seat's agent themselves,
exactly as they would for any BaseAgent today.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Hashable, Optional

logger = logging.getLogger(__name__)


class MultiAgentRunner:
    """
    Owns N agents (one per seat) and one multi-agent pipeline
    (ParallelMultiAgentPipeline or AECMultiAgentPipeline). Drives each
    seat's own agent.learn() call per window, independently.

    run() calls agents[aid].learn(episode_data) for every aid whose
    EpisodeData this window has at least one step (T > 0) -- an empty
    EpisodeData (a seat that got no turns this window, possible in the
    AEC pipeline, or that was never active in a Parallel window) is
    skipped rather than passed to learn(), since every existing agent's
    learn() assumes at least one step of data.
    """

    def __init__(self, pipeline: Any, agents: Dict[Hashable, Any]) -> None:
        self._pipeline = pipeline
        self._agents = agents

    def run(self, max_windows: Optional[int] = None) -> None:
        self._pipeline.setup()
        try:
            windows_run = 0
            for window in self._pipeline.episodes():
                for aid, episode_data in window.items():
                    if len(episode_data.rewards) == 0:
                        continue
                    self._agents[aid].learn(episode_data)

                windows_run += 1
                if max_windows is not None and windows_run >= max_windows:
                    break
        finally:
            self._pipeline.teardown()
