"""
Environment handling for RL discovery: accept a Gymnasium environment, a
zero-argument factory, or an environment id; step both the 5-tuple
(Gymnasium) and 4-tuple (legacy gym) APIs; build fresh, seeded copies for
trials; and the two wrappers discovery can recommend (fixed observation
normalisation, action rescaling to the [-1, 1] range tanh policies emit).
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np


def is_multi_agent(env: Any) -> bool:
    return hasattr(env, "possible_agents") or (hasattr(env, "agents") and hasattr(env, "observation_spaces"))


def env_factory(env_or_factory: Any) -> Tuple[Callable[[], Any], Any, Optional[str]]:
    """
    Returns (factory, a probe instance, note). With a Gymnasium env that has a
    registered spec, fresh copies come from ``gymnasium.make(spec)``; with a
    factory they come from the factory; otherwise the one instance is reused
    (trials run sequentially and re-seed it), and the note says so.
    """
    if isinstance(env_or_factory, str):
        return (lambda: gym.make(env_or_factory)), gym.make(env_or_factory), None
    if isinstance(env_or_factory, type) or (callable(env_or_factory) and not hasattr(env_or_factory, "step")):
        inst = env_or_factory()
        return env_or_factory, inst, None
    env = env_or_factory
    spec = getattr(env, "spec", None)
    if spec is not None:
        try:
            fresh = gym.make(spec)
            fresh.close()
            return (lambda: gym.make(spec)), env, None
        except Exception:
            pass
    return (lambda: env), env, ("the environment has no registered spec and no factory was given, so every "
                                "trial reuses the same instance (re-seeded); pass a zero-argument factory for "
                                "fully independent trials")


def reset(env: Any, seed: Optional[int] = None):
    try:
        out = env.reset(seed=seed)
    except TypeError:
        if seed is not None and hasattr(env, "seed"):
            env.seed(seed)
        out = env.reset()
    if isinstance(out, tuple) and len(out) == 2 and isinstance(out[1], dict):
        return out[0]
    return out


def step(env: Any, action) -> Tuple[Any, float, bool, bool]:
    out = env.step(action)
    if len(out) == 5:
        obs, r, term, trunc, _ = out
        return obs, float(r), bool(term), bool(trunc)
    obs, r, done, info = out
    trunc = bool(info.get("TimeLimit.truncated", False)) if isinstance(info, dict) else False
    return obs, float(r), bool(done) and not trunc, trunc


def flat_obs(obs) -> Optional[np.ndarray]:
    a = np.asarray(obs, dtype=np.float64) if not isinstance(obs, dict) else None
    if a is None or a.ndim != 1:
        return None
    return a


# ---------------------------------------------------------------------------
# wrappers
# ---------------------------------------------------------------------------

def make_wrapped(factory: Callable[[], Any], wrappers: Dict[str, Any]):
    """Apply discovery's wrappers in a fixed order: action rescale, then observation normalisation."""
    env = factory()
    if wrappers.get("action_rescale"):
        p = wrappers["action_rescale"]
        env = ActionRescale(env, np.asarray(p["low"], dtype=np.float64), np.asarray(p["high"], dtype=np.float64))
    if wrappers.get("obs_normalize"):
        p = wrappers["obs_normalize"]
        env = FixedObsNormalize(env, np.asarray(p["mean"], dtype=np.float64), np.asarray(p["std"], dtype=np.float64))
    return env


class FixedObsNormalize(gym.ObservationWrapper):
    """(obs - mean) / std with statistics fixed from discovery's random rollout
    (fixed, so a policy never sees its inputs drift under it), clipped to +-10."""

    def __init__(self, env, mean: np.ndarray, std: np.ndarray):
        super().__init__(env)
        self._mean = mean.astype(np.float32)
        self._std = np.where(std > 1e-8, std, 1.0).astype(np.float32)
        n = self._mean.shape
        self.observation_space = gym.spaces.Box(-10.0, 10.0, shape=n, dtype=np.float32)

    def observation(self, obs):
        return np.clip((np.asarray(obs, dtype=np.float32) - self._mean) / self._std, -10.0, 10.0)


class ActionRescale(gym.ActionWrapper):
    """Agents with tanh-squashed policies emit actions in [-1, 1]; map them onto [low, high]."""

    def __init__(self, env, low: np.ndarray, high: np.ndarray):
        super().__init__(env)
        self._low, self._high = low, high
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=low.shape, dtype=np.float32)

    def action(self, action):
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        return (self._low + (a + 1.0) * 0.5 * (self._high - self._low)).astype(np.float32)
