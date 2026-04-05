import numpy as np
from typing import Dict, Generator, List


def compute_gae(
    rewards: List[float],
    values: List[float],
    dones: List[bool],
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple:
    """
    Generalized Advantage Estimation (GAE-λ) over a flat rollout buffer.

    Handles episode boundaries within the buffer correctly: when dones[t] is True,
    the next-state bootstrap and future-advantage propagation are both zeroed out
    via the next_non_terminal mask. This means a single window can contain multiple
    episode fragments without needing separate per-episode buffers.

    Limitation: truncated episodes (time-limit) are treated identically to terminated
    episodes — V(s_{t+1}) is set to 0 rather than bootstrapped from the critic.
    For environments with fixed time limits (e.g. Gymnasium TimeLimit wrappers),
    this slightly underestimates value at the final step. Override compute_gae and
    pass the critic's V(next_obs) explicitly for precise handling.

    Args:
        rewards:    per-step rewards, length T
        values:     V(s_t) from critic, length T
        dones:      terminated | truncated flags, length T
        gamma:      discount factor
        gae_lambda: GAE smoothing parameter (0 = TD(0), 1 = Monte Carlo)

    Returns:
        advantages: np.ndarray shape [T], GAE-λ advantages
        returns:    np.ndarray shape [T], advantages + values (TD-λ targets for critic)
    """
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    values_arr = np.array(values, dtype=np.float32)
    last_gae = 0.0

    for t in reversed(range(T)):
        next_non_terminal = 1.0 - float(dones[t])
        next_value = values_arr[t + 1] if t < T - 1 else 0.0
        delta = rewards[t] + gamma * next_value * next_non_terminal - values_arr[t]
        last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        advantages[t] = last_gae

    returns = advantages + values_arr
    return advantages, returns


def make_minibatches(
    data: Dict[str, np.ndarray],
    minibatch_size: int,
    shuffle: bool = True,
) -> Generator[Dict[str, np.ndarray], None, None]:
    """
    Yield shuffled minibatch dicts from a flat rollout dict.

    All arrays in `data` must share the same first dimension (T steps).
    Yields ceil(T / minibatch_size) minibatches per call.

    Args:
        data:           dict of str → np.ndarray, all shape [T, ...]
        minibatch_size: number of samples per minibatch
        shuffle:        whether to shuffle indices before slicing

    Yields:
        dict with same keys, each value sliced to [minibatch_size, ...]
    """
    T = next(iter(data.values())).shape[0]
    indices = np.arange(T)
    if shuffle:
        np.random.shuffle(indices)
    for start in range(0, T, minibatch_size):
        batch_idx = indices[start : start + minibatch_size]
        yield {key: arr[batch_idx] for key, arr in data.items()}
