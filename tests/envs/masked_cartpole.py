"""
MaskedCartPoleEnv — CartPole-v1 with velocity observations zeroed out.

CartPole-v1 observation: [cart_pos, cart_vel, pole_angle, pole_ang_vel]

This wrapper zeros indices 1 and 3 (the velocity dimensions), turning the
fully-observable MDP into a POMDP.  The agent can no longer infer velocity
from a single observation.

Why this creates a genuine POMDP
---------------------------------
The optimal policy for CartPole requires knowledge of both position AND
velocity.  A static feedforward policy on [cart_pos, pole_angle] alone
cannot determine whether the cart is moving left or right, so it cannot
choose the correct corrective action with certainty.

An LSTM policy can approximate the velocity by computing
    vel_t ≈ (pos_t - pos_{t-1}) / Δt
from the history of position observations, recovering the Markov structure
through recurrency.

The velocity zeroing makes feedforward PPO structurally unable to solve
the task beyond chance performance, while LSTM PPO can recover from this.
"""

import numpy as np
import gymnasium as gym


class MaskedCartPoleEnv(gym.Wrapper):
    """
    CartPole-v1 with velocity dimensions (indices 1 and 3) set to 0.

    Feedforward policies see [cart_pos, 0, pole_angle, 0].
    LSTM policies can recover velocity from the history of positions.
    """

    def __init__(self):
        super().__init__(gym.make("CartPole-v1"))

    def observation(self, obs: np.ndarray) -> np.ndarray:
        obs = obs.copy()
        obs[1] = 0.0   # cart_velocity → 0
        obs[3] = 0.0   # pole_angular_velocity → 0
        return obs

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self.observation(obs), reward, terminated, truncated, info
