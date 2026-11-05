import numpy as np
from typing import Dict, Generator, List


def compute_gae(
    rewards: List[float],
    values: List[float],
    dones: List[bool],
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    last_value=0.0,
    n_envs: int = 1,
) -> tuple:
    """
    Generalized Advantage Estimation (GAE-λ) over a flat rollout buffer.

    Handles episode boundaries within the buffer correctly: when dones[t] is True,
    the next-state bootstrap and future-advantage propagation are both zeroed out
    via the next_non_terminal mask. This means a single window can contain multiple
    episode fragments without needing separate per-episode buffers.

    n_envs: number of parallel environments whose transitions are interleaved in
        the flat buffer, i.e. index t*n_envs + i holds env i's timestep t (the
        layout produced by VectorBatchPipeline). Default 1 treats the buffer as
        a single trajectory (array-index adjacency == timestep adjacency), which
        is the only layout a flat single-env pipeline ever produces. For n_envs
        > 1, the recurrence runs independently per env over its strided slice —
        entries belonging to a different env at the same timestep are never
        treated as temporally adjacent.

    last_value: V(s_T) — critic estimate of the state immediately after the window.
        Pass 0.0 (default) when the window ended at a terminal state.
        Pass the critic's value of the post-window observation when the window ended
        mid-episode (truncated by window_size, not by env termination). This corrects
        the bootstrap at the rollout boundary. TFPPOAgent and TorchPPOAgent compute
        this automatically from EpisodeData.final_obs; callers that bypass the agent
        should supply it explicitly.
        When n_envs > 1, pass a sequence of n_envs values (one bootstrap per env's
        own final observation) instead of a single scalar.

    Args:
        rewards:    per-step rewards, length T
        values:     V(s_t) from critic, length T
        dones:      terminated | truncated flags, length T
        gamma:      discount factor
        gae_lambda: GAE smoothing parameter (0 = TD(0), 1 = Monte Carlo)
        last_value: V(s_T) bootstrap for the post-window state(s)
        n_envs:     number of interleaved parallel environments (default 1)

    Returns:
        advantages: np.ndarray shape [T], GAE-λ advantages
        returns:    np.ndarray shape [T], advantages + values (TD-λ targets for critic)
    """
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    values_arr = np.array(values, dtype=np.float32)

    if n_envs > 1:
        last_values = np.broadcast_to(np.asarray(last_value, dtype=np.float32), (n_envs,))
    else:
        last_values = None

    for env_i in range(n_envs):
        env_indices = range(env_i, T, n_envs)
        last_gae = 0.0
        env_last_value = float(last_values[env_i]) if last_values is not None else float(last_value)
        for pos, t in enumerate(reversed(list(env_indices))):
            next_non_terminal = 1.0 - float(dones[t])
            is_last_step_for_env = (pos == 0)
            next_value = env_last_value if is_last_step_for_env else values_arr[t + n_envs]
            delta = rewards[t] + gamma * next_value * next_non_terminal - values_arr[t]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae

    returns = advantages + values_arr
    return advantages, returns


def resolve_gae_bootstrap(episode_data, critic_fn) -> tuple:
    """
    Compute the (last_value, n_envs) bootstrap arguments for compute_gae()
    from an EpisodeData's final_obs field and its dones.

    final_obs is None (single-env pipelines) -> n_envs=1, scalar last_value,
    bootstrapped only when the single trajectory's last step wasn't terminal.

    final_obs is a list (VectorBatchPipeline) -> n_envs=len(final_obs), one
    last_value per env, each independently bootstrapped from that env's own
    final observation (None when that env's own last step was terminal).

    critic_fn: callable obs -> float, the critic's V(obs) for one observation.
    """
    final_obs = episode_data.final_obs
    dones = episode_data.dones

    if isinstance(final_obs, (list, tuple)):
        n_envs = len(final_obs)
        T = len(dones)
        last_value = [
            critic_fn(final_obs[i]) if final_obs[i] is not None else 0.0
            for i in range(n_envs)
        ]
        return last_value, n_envs

    n_envs = 1
    last_value = 0.0
    if not dones[-1] and final_obs is not None:
        last_value = critic_fn(final_obs)
    return last_value, n_envs


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
