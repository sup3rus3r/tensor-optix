"""
HERReplayBuffer — Hindsight Experience Replay for goal-conditioned RL.

Reference: Andrychowicz et al. 2017 — "Hindsight Experience Replay"
           https://arxiv.org/abs/1707.01495

Design
------
HER wraps any PrioritizedReplayBuffer and adds a relabeling layer:

    store_episode(obs, acts, rews, next_obs, dones, achieved_goals, compute_reward)

For each transition t in an episode, it stores:

    1. The original transition  (obs[t], a[t], r[t], obs[t+1], done[t])
    2. k relabeled transitions  (obs_g'[t], a[t], r_g'[t], obs_g'[t+1], done[t])
       where g' is sampled from the episode according to `strategy` and
       r_g' = compute_reward(achieved_goals[t], g', {}).

obs_g'[t] = concat([env_obs_part(obs[t]), g'])  — the desired goal component
             of the observation is replaced with g'.

Relabeling strategies
---------------------
future  (default, best empirically):
    g' ~ Uniform({achieved_goals[t'], t' ∈ [t, T-1]})
    Uses the achieved goal of a future state in the same episode.
    On average k * T additional transitions per episode.

final:
    g' = achieved_goals[T-1]  (the final achieved state of the episode)
    Minimal variance; always uses the episode's terminal state.

episode:
    g' ~ Uniform({achieved_goals[i], i ∈ [0, T-1]})
    Samples uniformly from all achieved goals in the episode.

Integration
-----------
HERReplayBuffer presents the same sampling API as PrioritizedReplayBuffer:
    sample(batch_size)  → (obs, acts, rews, next_obs, dones, weights, idx, n)
    update_priorities(indices, errors)
    __len__

Agents do NOT call push() per step when using HER.  Instead, the pipeline
collects full episodes and calls store_episode() once per episode.

Usage::

    from tensor_optix.core.replay_buffer import PrioritizedReplayBuffer
    from tensor_optix.core.her_buffer import HERReplayBuffer

    inner = PrioritizedReplayBuffer(capacity=1_000_000, alpha=0.0)
    her   = HERReplayBuffer(inner, k=4, strategy="future")

    # At end of each episode:
    her.store_episode(
        obs_list, act_list, rew_list, next_obs_list, done_list,
        achieved_goals,   # shape [T, goal_dim] — achieved goal AFTER each step
        compute_reward,   # env.compute_reward(achieved, desired, info) -> float
    )

    # Inside agent._update_step():
    obs_b, act_b, rew_b, next_b, done_b, w, idx, n = her.sample(batch_size)
"""

from __future__ import annotations

from typing import Callable, List, Sequence

import numpy as np

from tensor_optix.core.replay_buffer import PrioritizedReplayBuffer


class HERReplayBuffer:
    """
    Hindsight Experience Replay buffer.

    Wraps a PrioritizedReplayBuffer; adds episode-level relabeling before
    transitions are committed to storage.

    Parameters
    ----------
    inner:
        The underlying storage buffer.  HER adds relabeled transitions on top
        of the original transitions.
    k:
        Number of HER goals sampled per transition.  Default 4 (as in the
        original paper).  Total transitions per episode = T * (1 + k).
    strategy:
        Goal sampling strategy.  One of "future" (default), "final", "episode".
    """

    VALID_STRATEGIES = frozenset({"future", "final", "episode"})

    def __init__(
        self,
        inner: PrioritizedReplayBuffer,
        k: int = 4,
        strategy: str = "future",
    ) -> None:
        if strategy not in self.VALID_STRATEGIES:
            raise ValueError(
                f"Unknown HER strategy {strategy!r}. "
                f"Valid options: {sorted(self.VALID_STRATEGIES)}"
            )
        self._inner    = inner
        self._k        = k
        self._strategy = strategy

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def store_episode(
        self,
        obs_list:         Sequence[np.ndarray],
        act_list:         Sequence[np.ndarray],
        rew_list:         Sequence[float],
        next_obs_list:    Sequence[np.ndarray],
        done_list:        Sequence[float],
        achieved_goals:   Sequence[np.ndarray],
        compute_reward:   Callable,
    ) -> None:
        """
        Store one full episode, with k HER-relabeled transitions per step.

        Parameters
        ----------
        obs_list:
            Observations at each step.  Shape [T, env_obs_dim + goal_dim].
            The last ``goal_dim`` elements are the desired goal (concatenated
            by the pipeline).
        act_list:
            Actions.  Shape [T, act_dim].
        rew_list:
            Original rewards from the environment.  Shape [T].
        next_obs_list:
            Next observations.  Shape [T, env_obs_dim + goal_dim].
        done_list:
            Done flags.  Shape [T].  1.0 = episode over.
        achieved_goals:
            Achieved goals **after** each transition.
            ``achieved_goals[t]`` = ``next_obs["achieved_goal"]`` at step t.
            Shape [T, goal_dim].
        compute_reward:
            ``env.compute_reward(achieved_goal, desired_goal, info) -> float``.
        """
        T = len(act_list)
        if T == 0:
            return

        goal_dim   = len(achieved_goals[0])
        obs_dim    = len(obs_list[0])
        env_obs_dim = obs_dim - goal_dim

        # --- Store original transitions (via inner's n-step path) ---
        for t in range(T):
            self._inner.push(
                obs_list[t], act_list[t], float(rew_list[t]),
                next_obs_list[t], float(done_list[t]),
            )
        self._inner.flush_episode()

        # --- Store HER-relabeled transitions (direct _commit, always 1-step) ---
        for t in range(T):
            her_goals = self._sample_goals(achieved_goals, t, T)
            env_obs_t      = obs_list[t][:env_obs_dim]
            env_next_obs_t = next_obs_list[t][:env_obs_dim]

            for g_prime in her_goals:
                g_prime = np.asarray(g_prime, dtype=np.float32)
                obs_r      = np.concatenate([env_obs_t,      g_prime])
                next_obs_r = np.concatenate([env_next_obs_t, g_prime])
                r_r        = float(compute_reward(
                    np.asarray(achieved_goals[t], dtype=np.float32),
                    g_prime,
                    {},
                ))
                # Use _commit() directly — relabeled transitions are always 1-step.
                self._inner._commit(
                    obs_r, act_list[t], r_r, next_obs_r,
                    float(done_list[t]), 1,
                )

    # ------------------------------------------------------------------
    # Buffer API (delegates to inner)
    # ------------------------------------------------------------------

    def sample(self, batch_size: int):
        """Sample a batch; delegates to the inner PrioritizedReplayBuffer."""
        return self._inner.sample(batch_size)

    def update_priorities(self, indices, errors) -> None:
        self._inner.update_priorities(indices, errors)

    def __len__(self) -> int:
        return len(self._inner)

    @property
    def _alpha(self):
        return self._inner._alpha

    @property
    def _beta(self):
        return self._inner._beta

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _sample_goals(
        self,
        achieved_goals: Sequence[np.ndarray],
        t: int,
        T: int,
    ) -> List[np.ndarray]:
        """
        Sample up to k HER goals for transition t according to strategy.

        future:  sample from future states in [t, T-1]
        final:   always use the last achieved goal
        episode: sample from any state in [0, T-1]
        """
        if self._strategy == "final":
            return [achieved_goals[-1]] * self._k

        if self._strategy == "future":
            # future indices drawn with replacement from [t, T-1]
            idxs = np.random.randint(t, T, size=self._k)
            return [achieved_goals[int(i)] for i in idxs]

        if self._strategy == "episode":
            idxs = np.random.randint(0, T, size=self._k)
            return [achieved_goals[int(i)] for i in idxs]

        raise ValueError(f"Unknown strategy: {self._strategy!r}")  # unreachable
