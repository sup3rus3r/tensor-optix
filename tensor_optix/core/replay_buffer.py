"""
Prioritized Experience Replay buffer with n-step return support.

Used by DQN and SAC agents (both Torch and TF).

PER reference: Schaul et al. 2016 — "Prioritized Experience Replay"
n-step reference: Sutton & Barto — multi-step TD targets

Design:
- alpha=0 (default): uniform sampling via simple deque — identical to classic DQN.
  Zero overhead, zero math change vs the original buffer.
- alpha>0: SumTree for O(log N) priority-weighted sampling with IS correction.
- n_step=1 (default): standard 1-step TD, no accumulation.
- n_step>1: n-step accumulation before committing to the buffer.
- alpha, beta, n_step are runtime-tunable (set_params) so SPSA can adapt them.
"""

import random
from collections import deque
from typing import Tuple

import numpy as np


class _SumTree:
    """
    Binary sum tree for O(log N) priority sampling.

    Leaves hold transition priorities. Internal nodes hold sums.
    """

    def __init__(self, capacity: int):
        self._capacity = capacity
        self._tree = np.zeros(2 * capacity, dtype=np.float64)
        self._data: list = [None] * capacity
        self._write = 0
        self._size = 0

    def _propagate(self, idx: int, delta: float) -> None:
        parent = idx // 2
        while parent >= 1:
            self._tree[parent] += delta
            parent //= 2

    def add(self, priority: float, data) -> None:
        idx = self._write + self._capacity
        old = self._tree[idx]
        self._tree[idx] = priority
        self._propagate(idx, priority - old)
        self._data[self._write] = data
        self._write = (self._write + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)

    def update(self, idx: int, priority: float) -> None:
        old = self._tree[idx]
        self._tree[idx] = priority
        self._propagate(idx, priority - old)

    def get(self, value: float) -> Tuple[int, float, object]:
        idx = 1
        while idx < self._capacity:
            left = 2 * idx
            right = left + 1
            if value <= self._tree[left]:
                idx = left
            else:
                value -= self._tree[left]
                idx = right
        leaf_idx = idx
        data_idx = leaf_idx - self._capacity
        return leaf_idx, self._tree[leaf_idx], self._data[data_idx]

    @property
    def total(self) -> float:
        return float(self._tree[1])

    @property
    def size(self) -> int:
        return self._size

    @property
    def max_priority(self) -> float:
        if self._size == 0:
            return 1.0
        leaf_vals = self._tree[self._capacity: self._capacity + self._size]
        return float(leaf_vals.max()) if len(leaf_vals) > 0 else 1.0


class PrioritizedReplayBuffer:
    """
    Replay buffer supporting both uniform and prioritized sampling, plus n-step returns.

    alpha=0 (default): standard uniform replay — simple deque + random.sample.
        Identical behaviour to the original DQN/SAC buffer. Zero overhead.
    alpha>0: SumTree-based prioritized sampling with IS correction weights.

    n_step=1 (default): standard 1-step TD target.
    n_step>1: accumulates n steps before committing, uses γⁿ discounting.

    All params are runtime-tunable via set_params() so SPSA can adapt them.

    Sampling always returns:
        obs, actions, rewards, next_obs, dones, weights, indices, n_steps
    When alpha=0, weights are all 1.0 and indices are dummy zeros (not used).
    """

    def __init__(
        self,
        capacity: int,
        alpha: float = 0.0,
        beta: float = 0.4,
        n_step: int = 1,
        gamma: float = 0.99,
        eps: float = 1e-6,
    ):
        self._capacity = capacity
        self._alpha = alpha
        self._beta = beta
        self._n_step = n_step
        self._gamma = gamma
        self._eps = eps

        # Uniform path: simple deque
        self._uniform_buf: deque = deque(maxlen=capacity)
        # PER path: SumTree (built lazily when alpha > 0)
        self._tree: _SumTree = None

        # n-step accumulation buffer
        self._n_buf: list = []

    def _ensure_tree(self) -> None:
        if self._tree is None:
            self._tree = _SumTree(self._capacity)

    def set_params(self, alpha: float = None, beta: float = None, n_step: int = None, gamma: float = None) -> None:
        """Update tunable params at runtime (called by agent.set_hyperparams)."""
        if alpha is not None:
            self._alpha = float(alpha)
            if self._alpha > 0:
                self._ensure_tree()
        if beta is not None:
            self._beta = float(beta)
        if n_step is not None:
            self._n_step = max(1, int(n_step))
        if gamma is not None:
            self._gamma = float(gamma)

    def push(self, obs, action, reward, next_obs, done) -> None:
        obs      = np.array(obs,      dtype=np.float32)
        next_obs = np.array(next_obs, dtype=np.float32)

        if self._n_step == 1:
            # Fast path: commit immediately, no accumulation
            self._commit(obs, action, float(reward), next_obs, float(done), n=1)
        else:
            self._n_buf.append((obs, action, float(reward), next_obs, float(done)))
            while len(self._n_buf) >= self._n_step:
                self._commit_n_step()
            if done:
                while self._n_buf:
                    self._commit_n_step()

    def _commit(self, obs, action, reward, next_obs, done, n: int) -> None:
        if self._alpha == 0:
            self._uniform_buf.append((obs, action, reward, next_obs, done, n))
        else:
            self._ensure_tree()
            priority = (self._tree.max_priority + self._eps) ** self._alpha
            self._tree.add(priority, (obs, action, reward, next_obs, done, n))

    def _commit_n_step(self) -> None:
        obs_0, action_0 = self._n_buf[0][0], self._n_buf[0][1]
        n = min(self._n_step, len(self._n_buf))
        G = sum((self._gamma ** k) * self._n_buf[k][2] for k in range(n))
        _, _, _, next_obs_n, done_n = self._n_buf[n - 1][:5]
        self._commit(obs_0, action_0, G, next_obs_n, done_n, n)
        self._n_buf.pop(0)

    def sample(self, batch_size: int):
        """
        Returns: obs, actions, rewards, next_obs, dones, weights, indices, n_steps
        weights = 1.0 for all when alpha=0.
        """
        if self._alpha == 0:
            return self._sample_uniform(batch_size)
        return self._sample_per(batch_size)

    def _sample_uniform(self, batch_size: int):
        batch = random.sample(self._uniform_buf, batch_size)
        obs, actions, rewards, next_obs, dones, ns = zip(*batch)
        ones = np.ones(batch_size, dtype=np.float32)
        zeros = np.zeros(batch_size, dtype=np.int64)
        return (
            np.array(obs,     dtype=np.float32),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_obs,dtype=np.float32),
            np.array(dones,   dtype=np.float32),
            ones, zeros,
            np.array(ns,      dtype=np.int32),
        )

    def _sample_per(self, batch_size: int):
        if self._tree.total <= 0 or self._tree.size < batch_size:
            raise ValueError("Not enough samples in PER buffer")

        indices  = np.zeros(batch_size, dtype=np.int64)
        priorities = np.zeros(batch_size, dtype=np.float64)
        obs_l, act_l, rew_l, nxt_l, don_l, ns_l = [], [], [], [], [], []

        segment = self._tree.total / batch_size
        for i in range(batch_size):
            val = np.random.uniform(segment * i, segment * (i + 1))
            leaf_idx, priority, data = self._tree.get(val)
            if data is None:
                val = np.random.uniform(0, self._tree.total)
                leaf_idx, priority, data = self._tree.get(val)
            obs_0, action_0, G, next_obs_n, done_n, n = data
            indices[i] = leaf_idx
            priorities[i] = priority
            obs_l.append(obs_0); act_l.append(action_0); rew_l.append(G)
            nxt_l.append(next_obs_n); don_l.append(done_n); ns_l.append(n)

        N = self._tree.size
        probs   = priorities / (self._tree.total + 1e-10)
        weights = (N * probs) ** (-self._beta)
        weights = (weights / weights.max()).astype(np.float32)

        return (
            np.array(obs_l,  dtype=np.float32),
            np.array(act_l),
            np.array(rew_l,  dtype=np.float32),
            np.array(nxt_l,  dtype=np.float32),
            np.array(don_l,  dtype=np.float32),
            weights, indices,
            np.array(ns_l,   dtype=np.int32),
        )

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        """Update priorities after TD error is known. No-op when alpha=0."""
        if self._alpha == 0 or self._tree is None:
            return
        for idx, err in zip(indices, td_errors):
            self._tree.update(int(idx), (float(abs(err)) + self._eps) ** self._alpha)

    def flush_episode(self) -> None:
        """Flush n-step buffer at episode end."""
        self._n_buf.clear()

    def __len__(self) -> int:
        if self._alpha == 0:
            return len(self._uniform_buf)
        return self._tree.size if self._tree is not None else 0
