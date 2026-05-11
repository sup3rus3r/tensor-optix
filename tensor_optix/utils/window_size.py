from __future__ import annotations

"""
Compute the optimal BatchPipeline window_size for a given environment
and algorithm without requiring the user to guess a magic number.

Formula:
    window_size = clip(k * mean_episode_steps, MIN_STEPS, MAX_STEPS)

k multipliers:
    SAC / TD3  (off-policy)  k = 1.0  — one episode per window is sufficient;
                                         the replay buffer handles sample efficiency.
    PPO / RecurrentPPO       k = 4.0  — on-policy needs ~4x episode length for
    (on-policy)                          well-estimated GAE advantages.
                                         Matches SB3's n_steps=2048 for CartPole
                                         (max 500 steps → 4*500=2000 ≈ 2048).
    DQN / Rainbow            k = 1.0  — off-policy, same reasoning as SAC.
    default                  k = 2.0  — unknown algorithm, conservative middle ground.

mean_episode_steps is estimated from the env in this order:
  1. env.spec.max_episode_steps   (Gymnasium TimeLimit wrapper metadata)
  2. env._max_episode_steps       (unwrapped TimeLimit attribute)
  3. Short calibration rollout    (5 episodes, random actions)
  4. Fallback: 1000
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

_MIN_STEPS = 512
_MAX_STEPS = 8192

# Canonical algorithm name → k multiplier
_K: dict[str, float] = {
    "sac":          1.0,
    "td3":          1.0,
    "dqn":          1.0,
    "rainbow":      1.0,
    "rainbowdqn":   1.0,
    "ppo":          4.0,
    "recurrentppo": 4.0,
    "rppo":         4.0,
}


def optimal_window_size(
    env,
    algorithm: str = "default",
    min_steps: int = _MIN_STEPS,
    max_steps: int = _MAX_STEPS,
) -> int:
    """
    Return the optimal window_size for BatchPipeline given *env* and *algorithm*.

    Parameters
    ----------
    env:
        Gymnasium environment (needs observation_space / action_space).
    algorithm:
        Case-insensitive algorithm name: ``"SAC"``, ``"PPO"``, ``"TD3"``,
        ``"DQN"``, ``"Rainbow"``.  Unknown strings fall back to k=2.0.
    min_steps:
        Hard floor on the returned window size (default 512).
    max_steps:
        Hard ceiling on the returned window size (default 8192).

    Returns
    -------
    int
        Recommended window_size.
    """
    k = _K.get(algorithm.lower().replace(" ", "").replace("_", ""), 2.0)
    mean_ep = _estimate_mean_episode_steps(env)
    result = max(min_steps, min(max_steps, int(k * mean_ep)))
    logger.debug(
        "optimal_window_size: alg=%s k=%.1f mean_ep=%d → window=%d",
        algorithm, k, mean_ep, result,
    )
    return result


def _estimate_mean_episode_steps(env) -> int:
    """Estimate mean episode length from env metadata, then calibration rollout."""

    # 1. Gymnasium spec metadata (most reliable — set by TimeLimit wrapper)
    try:
        if hasattr(env, "spec") and env.spec is not None:
            meps = getattr(env.spec, "max_episode_steps", None)
            if meps is not None and meps > 0:
                return int(meps)
    except Exception:
        pass

    # 2. Direct TimeLimit attribute (unwrapped env)
    try:
        meps = getattr(env, "_max_episode_steps", None)
        if meps is not None and meps > 0:
            return int(meps)
    except Exception:
        pass

    # 3. Short calibration rollout — 5 episodes of random actions
    try:
        return _calibration_rollout(env, n_episodes=5)
    except Exception as exc:
        logger.debug("calibration rollout failed: %s", exc)

    # 4. Conservative fallback
    logger.warning(
        "optimal_window_size: could not estimate mean episode length for %s. "
        "Using fallback of 1000 steps. Pass window_size explicitly to override.",
        type(env).__name__,
    )
    return 1000


def _calibration_rollout(env, n_episodes: int = 5) -> int:
    """Run n_episodes of random actions; return mean steps per episode."""
    import numpy as np

    lengths = []
    for _ in range(n_episodes):
        steps = 0
        try:
            env.reset()
        except Exception:
            break
        done = False
        while not done:
            action = env.action_space.sample()
            try:
                result = env.step(action)
            except Exception:
                break
            # Gymnasium 1.0: (obs, reward, terminated, truncated, info)
            # Older API:      (obs, reward, done, info)
            if len(result) == 5:
                _, _, terminated, truncated, _ = result
                done = terminated or truncated
            else:
                _, _, done, _ = result
            steps += 1
        lengths.append(steps)

    if not lengths:
        raise RuntimeError("calibration rollout produced no episodes")
    return int(np.mean(lengths))
