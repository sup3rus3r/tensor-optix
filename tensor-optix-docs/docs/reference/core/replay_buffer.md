# tensor_optix.core.replay_buffer

```
Prioritized Experience Replay buffer with n-step return support.

Used by DQN and SAC agents (both Torch and TF).

PER reference: Schaul et al. 2016 - "Prioritized Experience Replay"
n-step reference: Sutton & Barto - multi-step TD targets

Design:
- alpha=0 (default): uniform sampling via simple deque - identical to classic DQN.
  Zero overhead, zero math change vs the original buffer.
- alpha>0: SumTree for O(log N) priority-weighted sampling with IS correction.
- n_step=1 (default): standard 1-step TD, no accumulation.
- n_step>1: n-step accumulation before committing to the buffer.
- alpha, beta, n_step are runtime-tunable (set_params) so SPSA can adapt them.
```

## PrioritizedReplayBuffer

```python
class PrioritizedReplayBuffer:
    """
    Replay buffer supporting both uniform and prioritized sampling, plus n-step returns.

    alpha=0 (default): standard uniform replay - simple deque + random.sample.
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
    ): ...

    def set_params(self, alpha=None, beta=None, n_step=None, gamma=None) -> None:
        """Update tunable params at runtime (called by agent.set_hyperparams)."""

    def push(self, obs, action, reward, next_obs, done) -> None: ...

    def sample(self, batch_size: int):
        """
        Returns: obs, actions, rewards, next_obs, dones, weights, indices, n_steps
        weights = 1.0 for all when alpha=0.
        """

    def update_priorities(self, indices, td_errors) -> None:
        """Update priorities after TD error is known. No-op when alpha=0."""

    def flush_episode(self) -> None:
        """Flush n-step buffer at episode end."""

    def __len__(self) -> int: ...
```

Internally backed by a binary `_SumTree` for O(log N) priority sampling when `alpha > 0`; an ordinary `deque` is used when `alpha == 0` for zero overhead on the common case.

See also [HERReplayBuffer](her_buffer.md), which wraps this buffer with goal relabeling.
