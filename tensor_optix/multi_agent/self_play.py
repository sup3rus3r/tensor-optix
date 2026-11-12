"""
tensor_optix.multi_agent.self_play — frozen-opponent pool and the
single-learner self-play pipeline.

Two ways to use past versions of a policy as opponents:

  MultiAgentRunner(self_play=SelfPlayConfig(...))
      Every seat keeps learning its own policy. Each window, a seat may be
      "frozen": played by a past snapshot of itself drawn from that seat's
      pool. The live seats train against a mix of current and past opponents.

  SelfPlayPipeline(multi_agent_pipeline, learner_seat=...)
      ONE learner. The other seats are part of the environment, played by
      past snapshots of the learner. It is an ordinary single-agent
      BasePipeline, so RLOptimizer / LoopController drive it with all of
      their machinery (tuning, rollback, spawning, checkpoint confirmation).

Why a pool: two agents that only ever play each other's current version tend
to chase each other in circles — forgetting how to beat an old strategy
costs nothing. Sampling opponents from past snapshots makes that forgetting
cost reward.
"""
from __future__ import annotations

import copy
import logging
import os
import random
import shutil
import tempfile
import types
from dataclasses import dataclass
from typing import Any, Callable, Dict, Generator, Hashable, List, Optional

from tensor_optix.core.base_pipeline import BasePipeline
from tensor_optix.core.types import EpisodeData

logger = logging.getLogger(__name__)


@dataclass
class SelfPlayConfig:
    """
    pool_size:        snapshots kept per seat (oldest evicted first).
    snapshot_every:   save a snapshot of each learning policy every N windows.
    opponent_prob:    MultiAgentRunner only — per window, probability that a
                      seat is played by a past snapshot instead of its live
                      agent. At least one seat always stays live.
    pool_dir:         where snapshots are written. None = a temporary
                      directory, removed when the run ends.
    opponent_factory: callable(agent_id) -> a fresh agent of the same
                      architecture, used as the stand-in that snapshot
                      weights are loaded into. None = copy.deepcopy of the
                      live agent (works for the built-in Torch agents).
    seed:             seed for the pool's own sampling RNG.
    """
    pool_size: int = 10
    snapshot_every: int = 20
    opponent_prob: float = 0.5
    pool_dir: Optional[str] = None
    opponent_factory: Optional[Callable[[Hashable], Any]] = None
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        if self.pool_size < 1:
            raise ValueError("SelfPlayConfig.pool_size must be >= 1")
        if self.snapshot_every < 1:
            raise ValueError("SelfPlayConfig.snapshot_every must be >= 1")
        if not 0.0 <= self.opponent_prob <= 1.0:
            raise ValueError("SelfPlayConfig.opponent_prob must be in [0, 1]")


class OpponentPool:
    """
    Per-key FIFO pool of weight snapshots on disk.

    Only weights are stored; an opponent is materialised by loading a
    snapshot into a single stand-in agent per key, so memory stays flat no
    matter how large the pool is.
    """

    def __init__(self, pool_size: int = 10, pool_dir: Optional[str] = None,
                 seed: Optional[int] = None) -> None:
        self._pool_size = max(1, int(pool_size))
        self._owns_dir = pool_dir is None
        self._dir = pool_dir or tempfile.mkdtemp(prefix="to_selfplay_")
        os.makedirs(self._dir, exist_ok=True)
        self._snapshots: Dict[Hashable, List[str]] = {}
        self._counter = 0
        self._rng = random.Random(seed)

    def add(self, key: Hashable, agent: Any) -> str:
        """Save agent's current weights as the newest snapshot for key."""
        self._counter += 1
        snap_dir = os.path.join(self._dir, _safe_name(key), f"snap_{self._counter:06d}")
        os.makedirs(snap_dir, exist_ok=True)
        path = os.path.join(snap_dir, "weights")
        agent.save_weights(path)
        entries = self._snapshots.setdefault(key, [])
        entries.append(path)
        while len(entries) > self._pool_size:
            old = entries.pop(0)
            shutil.rmtree(os.path.dirname(old), ignore_errors=True)
        return path

    def size(self, key: Hashable) -> int:
        return len(self._snapshots.get(key, []))

    def sizes(self) -> Dict[Hashable, int]:
        return {k: len(v) for k, v in self._snapshots.items()}

    def sample(self, key: Hashable) -> Optional[str]:
        """Uniformly sample one snapshot path for key (None if empty)."""
        entries = self._snapshots.get(key)
        if not entries:
            return None
        return self._rng.choice(entries)

    def load_into(self, key: Hashable, stand_in: Any, live_agent: Any = None) -> bool:
        """
        Load a sampled snapshot into stand_in. Hyperparameters are synced
        from live_agent first (a stand-in built at the start of training
        would otherwise keep e.g. its initial, fully-random epsilon).
        """
        path = self.sample(key)
        if path is None:
            return False
        if live_agent is not None and hasattr(live_agent, "get_hyperparams"):
            try:
                stand_in.set_hyperparams(live_agent.get_hyperparams())
            except Exception:  # hyperparameter sync is best-effort
                pass
        stand_in.load_weights(path)
        return True

    @property
    def rng(self) -> random.Random:
        return self._rng

    def cleanup(self) -> None:
        """Remove the pool directory if this pool created it."""
        if self._owns_dir:
            shutil.rmtree(self._dir, ignore_errors=True)


def make_stand_in(agent: Any, agent_id: Hashable,
                  factory: Optional[Callable[[Hashable], Any]] = None) -> Any:
    """Build the frozen stand-in agent that snapshot weights are loaded into."""
    if factory is not None:
        return factory(agent_id)
    try:
        # Agents commonly keep a reference to their framework module
        # (self._torch = torch). Modules cannot be deep-copied and must not
        # be: share them by pre-seeding the memo.
        memo = {
            id(value): value
            for value in vars(agent).values()
            if isinstance(value, types.ModuleType)
        } if hasattr(agent, "__dict__") else {}
        stand_in = copy.deepcopy(agent, memo)
    except Exception as exc:
        raise TypeError(
            f"Self-play needs a second copy of seat {agent_id!r}'s agent to play "
            f"frozen snapshots, and copy.deepcopy() failed for "
            f"{type(agent).__name__} ({exc}). Pass "
            f"SelfPlayConfig(opponent_factory=lambda agent_id: <build a fresh agent>)."
        ) from exc
    discard_rollout(stand_in)
    return stand_in


def discard_rollout(agent: Any) -> None:
    """A stand-in acts but never learns; drop whatever act() cached."""
    reset = getattr(agent, "reset_cache", None)
    if callable(reset):
        reset()


class SelfPlayPipeline(BasePipeline):
    """
    Single-learner self-play: turns a multi-agent pipeline into an ordinary
    single-agent pipeline.

    The learner plays `learner_seat`. Every other seat is played by a
    stand-in loaded, each window, with a snapshot drawn from a pool of the
    learner's own past weights. The pool starts with the learner's initial
    weights and gains a snapshot every `snapshot_every` windows.

    Requirements: the opponent seats must be playable by the learner's
    policy — same observation and action spaces (symmetric games such as
    rock-paper-scissors, Connect Four, Pong).

    Usage:
        inner = AECMultiAgentPipeline(env, agents={}, window_size=2048, views=views)
        pipeline = SelfPlayPipeline(inner, learner_seat="player_0",
                                    config=SelfPlayConfig(snapshot_every=10))
        RLOptimizer(agent=agent, pipeline=pipeline, ...).run()

    Windows in which the learner's seat took no step are skipped. With the
    AEC pipeline, window_size counts steps across ALL seats, so the learner
    gets roughly window_size / n_seats steps per window.
    """

    def __init__(
        self,
        pipeline: Any,
        learner_seat: Hashable,
        agent: Any = None,
        config: Optional[SelfPlayConfig] = None,
    ) -> None:
        if not hasattr(pipeline, "set_agents"):
            raise TypeError("SelfPlayPipeline needs a multi-agent pipeline exposing set_agents().")
        self._inner = pipeline
        self._learner_seat = learner_seat
        self._agent = agent
        self._cfg = config or SelfPlayConfig()
        self._pool: Optional[OpponentPool] = None
        self._stand_ins: Dict[Hashable, Any] = {}
        self._windows = 0

    def set_agent(self, agent) -> None:
        self._agent = agent

    @property
    def pool(self) -> Optional[OpponentPool]:
        return self._pool

    @property
    def steps_per_episode(self) -> Optional[int]:
        return getattr(self._inner, "_window_size", None)

    @property
    def is_live(self) -> bool:
        return False

    def setup(self) -> None:
        if self._agent is None:
            raise RuntimeError("SelfPlayPipeline.setup(): no learner agent set.")
        self._inner.setup()
        if self._pool is None:
            self._pool = OpponentPool(self._cfg.pool_size, self._cfg.pool_dir, self._cfg.seed)
            self._pool.add(self._learner_seat, self._agent)   # opponents start as the initial policy

    def _opponent_seats(self) -> List[Hashable]:
        env = getattr(self._inner, "_env", None)
        seats = list(getattr(env, "possible_agents", []))
        if self._learner_seat not in seats:
            raise ValueError(
                f"learner_seat {self._learner_seat!r} is not one of the env's seats {seats}."
            )
        return [s for s in seats if s != self._learner_seat]

    def _refresh_opponents(self) -> None:
        mapping = {self._learner_seat: self._agent}
        for seat in self._opponent_seats():
            stand_in = self._stand_ins.get(seat)
            if stand_in is None:
                stand_in = make_stand_in(self._agent, seat, self._cfg.opponent_factory)
                self._stand_ins[seat] = stand_in
            discard_rollout(stand_in)
            self._pool.load_into(self._learner_seat, stand_in, self._agent)
            mapping[seat] = stand_in
        self._inner.set_agents(mapping)

    def episodes(self) -> Generator[EpisodeData, None, None]:
        self._refresh_opponents()
        for window in self._inner.episodes():
            data = window.get(self._learner_seat)
            self._windows += 1
            if self._windows % self._cfg.snapshot_every == 0:
                # The learner's weights as of the START of this window's
                # learn() are what just played; snapshot them.
                self._pool.add(self._learner_seat, self._agent)
            self._refresh_opponents()
            if data is None or len(data.rewards) == 0:
                continue
            yield data

    def teardown(self) -> None:
        try:
            self._inner.teardown()
        finally:
            for stand_in in self._stand_ins.values():
                discard_rollout(stand_in)
            if self._pool is not None:
                self._pool.cleanup()
                self._pool = None


def _safe_name(key: Hashable) -> str:
    text = str(key)
    return "".join(c if (c.isalnum() or c in "-_.") else "_" for c in text) or "seat"
