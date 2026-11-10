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

MultiAgentRunner is a deliberately minimal loop: it calls each seat's
agent.learn() per window and nothing else. No convergence detection, no
checkpointing, no rollback across seats. Wire a seat's agent into an
ordinary single-agent LoopController yourself if you want the RL core's
full machinery for that seat.

Out of scope for this package (confirmed with the user): shared/self-play
policies across seats, coordinated multi-agent convergence detection,
and vectorizing across multiple copies of a multi-agent environment.
"""
from tensor_optix.multi_agent.seat_env import SeatEnvView
from tensor_optix.multi_agent.parallel_pipeline import ParallelMultiAgentPipeline
from tensor_optix.multi_agent.aec_pipeline import AECMultiAgentPipeline
from tensor_optix.multi_agent.multi_agent_runtime import MultiAgentRunner

__all__ = [
    "SeatEnvView",
    "ParallelMultiAgentPipeline",
    "AECMultiAgentPipeline",
    "MultiAgentRunner",
]
