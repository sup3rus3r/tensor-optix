"""
tensor_optix.multi_agent — multi-agent environment support (PettingZoo
AEC + Parallel APIs).

No changes are made anywhere else in tensor-optix to support this.
make_agent() builds one completely ordinary, already-tested BaseAgent
instance per seat via SeatEnvView -- a thin Gym-shaped wrapper around one
agent_id's slice of a PettingZoo env. EpisodeData, BasePipeline, and
LoopController are untouched; the pipelines here produce
Dict[AgentID, EpisodeData] (one real EpisodeData per seat) instead of a
new multi-agent EpisodeData variant, so every existing agent class's
learn() is reused completely unmodified.

MultiAgentRunner drives every seat's agent.learn() per window and applies
per-seat convergence detection (BackoffScheduler) and best-snapshot
checkpointing (CheckpointRegistry); the run stops when all seats (or any
seat, configurable) have converged, and restores each seat's best weights.

Out of scope for this package: shared/self-play policies across seats,
online hyperparameter tuning and degradation rollback per seat, and
vectorizing across multiple copies of a multi-agent environment.
"""
from tensor_optix.multi_agent.seat_env import SeatEnvView
from tensor_optix.multi_agent.parallel_pipeline import ParallelMultiAgentPipeline
from tensor_optix.multi_agent.aec_pipeline import AECMultiAgentPipeline
from tensor_optix.multi_agent.multi_agent_runtime import (
    MultiAgentCallback,
    MultiAgentRunResult,
    MultiAgentRunner,
    SeatResult,
)

__all__ = [
    "SeatEnvView",
    "ParallelMultiAgentPipeline",
    "AECMultiAgentPipeline",
    "MultiAgentRunner",
    "MultiAgentCallback",
    "MultiAgentRunResult",
    "SeatResult",
]
