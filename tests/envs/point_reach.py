"""
PointReachEnv — minimal 2D goal-conditioned environment for HER testing.

A 2-D point mass lives in [-1, 1]². The agent applies force directly.
Reward is sparse: 0 on success (within `tol` of goal), -1 otherwise.

Implements the GoalEnv convention:
    observation_space: Dict(observation, achieved_goal, desired_goal)
    compute_reward(achieved_goal, desired_goal, info) -> float

This environment is intentionally difficult without HER:
  - P(random action reaches goal) ≈ (π * tol²) / 4  ≈ 0.008 for tol=0.1
  - A random policy accumulates reward ≈ -1 per step almost always.
  - With HER, every episode produces at least one "success" in hindsight.

Not registered in Gymnasium — imported directly in tests.
"""

import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Dict


class PointReachEnv(gym.Env):
    """
    2D point-mass goal-reaching environment.

    obs["observation"]   — current position [x, y] ∈ [-1, 1]²
    obs["achieved_goal"] — current position [x, y]  (same as observation)
    obs["desired_goal"]  — target position  [gx, gy] ∈ [-1, 1]²

    Action: velocity delta [dx, dy] ∈ [-1, 1]²
    Position clamped to [-1, 1]².

    Reward: 0.0 if ||pos - goal|| < tol, else -1.0   (sparse)
    Episode terminates when success or after max_steps.
    """

    metadata = {"render_modes": []}

    def __init__(self, tol: float = 0.1, max_steps: int = 50, seed: int = 0):
        super().__init__()
        self.tol       = tol
        self.max_steps = max_steps

        self.observation_space = Dict({
            "observation":   Box(-1.0, 1.0, shape=(2,), dtype=np.float32),
            "achieved_goal": Box(-1.0, 1.0, shape=(2,), dtype=np.float32),
            "desired_goal":  Box(-1.0, 1.0, shape=(2,), dtype=np.float32),
        })
        self.action_space = Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

        self._rng   = np.random.default_rng(seed)
        self._pos   = np.zeros(2, dtype=np.float32)
        self._goal  = np.zeros(2, dtype=np.float32)
        self._step  = 0

    # ------------------------------------------------------------------
    # GoalEnv interface
    # ------------------------------------------------------------------

    def compute_reward(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        info,
    ) -> float:
        """
        Sparse reward: 0.0 on success, -1.0 otherwise.

        Works element-wise for batched inputs (shape [..., 2]):
            compute_reward(batch_achieved, batch_desired, {})
            → np.ndarray of shape [...]
        """
        achieved = np.asarray(achieved_goal, dtype=np.float64)
        desired  = np.asarray(desired_goal,  dtype=np.float64)
        d = np.linalg.norm(achieved - desired, axis=-1)
        return -(d >= self.tol).astype(np.float32)

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._pos  = self._rng.uniform(-1.0, 1.0, size=2).astype(np.float32)
        self._goal = self._rng.uniform(-1.0, 1.0, size=2).astype(np.float32)
        self._step = 0
        return self._obs(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        self._pos  = np.clip(self._pos + 0.1 * action, -1.0, 1.0).astype(np.float32)
        self._step += 1

        reward    = float(self.compute_reward(self._pos, self._goal, {}))
        success   = reward == 0.0
        truncated = self._step >= self.max_steps
        terminated = success

        return self._obs(), reward, terminated, truncated, {"is_success": success}

    def _obs(self):
        return {
            "observation":   self._pos.copy(),
            "achieved_goal": self._pos.copy(),
            "desired_goal":  self._goal.copy(),
        }
