from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional
import time


class LoopState(Enum):
    ACTIVE = auto()
    COOLING = auto()
    DORMANT = auto()
    WATCHDOG = auto()


@dataclass
class EpisodeData:
    """Raw interaction data produced by one episode."""
    observations: Any          # np.ndarray or tf.Tensor, shape [T, obs_dim]
    actions: Any               # np.ndarray or tf.Tensor, shape [T, act_dim] or [T]
    rewards: List[float]       # per-step rewards
    terminated: List[bool]     # Gymnasium terminated flags
    truncated: List[bool]      # Gymnasium truncated flags
    infos: List[Dict]          # per-step info dicts from env
    episode_id: int
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def dones(self) -> List[bool]:
        """Convenience: True when episode ended for any reason."""
        return [t or tr for t, tr in zip(self.terminated, self.truncated)]

    @property
    def total_reward(self) -> float:
        return sum(self.rewards)

    @property
    def length(self) -> int:
        return len(self.rewards)


@dataclass
class EvalMetrics:
    """
    Scored output from BaseEvaluator.score().
    primary_score is the single comparable scalar — higher is always better.
    metrics holds the full breakdown for logging/debugging.
    """
    primary_score: float
    metrics: Dict[str, float]
    episode_id: int
    timestamp: float = field(default_factory=time.time)

    def beats(self, other: "EvalMetrics", margin: float = 0.0) -> bool:
        """True if this score beats other by at least margin."""
        return self.primary_score > other.primary_score + margin


@dataclass
class HyperparamSet:
    """
    A snapshot of hyperparameters at a point in time.

    params is a completely open dict. Core never reads, writes, or assumes
    any specific key names. Keys and value types are defined entirely by
    the agent implementation and the optimizer.

    Examples:
        PPO:   {"learning_rate": 3e-4, "clip_ratio": 0.2, "entropy_coeff": 0.01, "gamma": 0.99}
        DQN:   {"learning_rate": 1e-3, "epsilon": 0.1, "gamma": 0.99, "target_update_freq": 100}
        CMA-ES: {"sigma": 0.3, "population_size": 16}
    """
    params: Dict[str, Any]
    episode_id: int
    timestamp: float = field(default_factory=time.time)

    def copy(self) -> "HyperparamSet":
        return HyperparamSet(
            params=dict(self.params),
            episode_id=self.episode_id,
            timestamp=self.timestamp,
        )


@dataclass
class PolicySnapshot:
    """
    A complete checkpoint: weights + hyperparams + eval score.
    Stored by CheckpointRegistry whenever a new best is achieved.
    """
    snapshot_id: str
    eval_metrics: EvalMetrics
    hyperparams: HyperparamSet
    weights_path: str
    episode_id: int
    timestamp: float = field(default_factory=time.time)
