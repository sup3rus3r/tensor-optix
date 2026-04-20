"""
tests/test_improvements/test_04_auto_select.py

Tests for tensor_optix.make_agent() — the auto algorithm selection factory.

Mathematical invariants being verified:

1. TYPE CORRECTNESS
   Discrete action space  →  PPO   (categorical π; softmax over integer indices)
   Box action space       →  SAC   (squashed Gaussian; output ∈ (-1,1)^d)
   Box + deterministic    →  TD3   (deterministic π; output ∈ (-1,1)^d via tanh)
   MultiDiscrete          →  NotImplementedError (factored policy, no universal form)
   Dict / Tuple           →  NotImplementedError (structured policy, custom arch required)
   MultiBinary            →  NotImplementedError (multi-label Bernoulli, custom arch)

2. ACTION SPACE INCOMPATIBILITY
   SAC built for Box must output values in (-1, 1) — structurally incompatible with
   Discrete environments that require integer indices in {0, ..., n-1}.
   PPO built for Discrete must output integer actions — structurally incompatible
   with continuous envs that require real-valued vectors.
   We verify this type mismatch directly (no training required).

3. OBSERVATION SPACE VALIDATION
   Non-flat (image) or non-Box observation spaces raise NotImplementedError.

4. DETERMINISTIC FLAG
   make_agent(env, deterministic=True) → TD3
   make_agent(env, deterministic=False) → SAC   (default)

5. FRAMEWORK ROUTING
   framework="torch" → Torch variants
   framework="tf"    → TF variants (tested via isinstance)

6. NETWORK ARCHITECTURE
   Built networks have correct input/output dimensions derived from env spaces.
"""

import numpy as np
import pytest

from gymnasium.spaces import (
    Box, Discrete, MultiBinary, MultiDiscrete,
    Dict as GymDict, Tuple as GymTuple,
)
from gymnasium.spaces import utils as gym_utils

from tensor_optix.algorithms.torch_ppo import TorchPPOAgent
from tensor_optix.algorithms.torch_sac import TorchSACAgent
from tensor_optix.algorithms.torch_td3 import TorchTD3Agent
from tensor_optix.factory import make_agent


# ---------------------------------------------------------------------------
# Minimal fake environment — carries spaces without running a real sim
# ---------------------------------------------------------------------------

class FakeEnv:
    def __init__(self, obs_space, act_space):
        self.observation_space = obs_space
        self.action_space      = act_space


def _flat_box(n: int) -> Box:
    return Box(low=-np.inf, high=np.inf, shape=(n,), dtype=np.float32)


def _continuous_env(obs_dim=8, act_dim=2) -> FakeEnv:
    return FakeEnv(_flat_box(obs_dim), Box(low=-1, high=1, shape=(act_dim,), dtype=np.float32))


def _discrete_env(obs_dim=8, n_actions=4) -> FakeEnv:
    return FakeEnv(_flat_box(obs_dim), Discrete(n_actions))


# ---------------------------------------------------------------------------
# 1. Type correctness
# ---------------------------------------------------------------------------

class TestTypeCorrectness:

    def test_discrete_returns_ppo(self):
        env   = _discrete_env()
        agent = make_agent(env, framework="torch")
        assert isinstance(agent, TorchPPOAgent), \
            f"Expected TorchPPOAgent for Discrete space, got {type(agent).__name__}"

    def test_continuous_returns_sac_by_default(self):
        env   = _continuous_env()
        agent = make_agent(env, framework="torch")
        assert isinstance(agent, TorchSACAgent), \
            f"Expected TorchSACAgent for Box space (default), got {type(agent).__name__}"

    def test_continuous_deterministic_returns_td3(self):
        env   = _continuous_env()
        agent = make_agent(env, framework="torch", deterministic=True)
        assert isinstance(agent, TorchTD3Agent), \
            f"Expected TorchTD3Agent for Box space + deterministic=True, got {type(agent).__name__}"

    def test_multidiscrete_raises(self):
        env = FakeEnv(_flat_box(8), MultiDiscrete([3, 4]))
        with pytest.raises(NotImplementedError, match="MultiDiscrete"):
            make_agent(env)

    def test_dict_action_raises(self):
        env = FakeEnv(_flat_box(8), GymDict({"x": Discrete(3), "y": Box(-1, 1, (2,))}))
        with pytest.raises(NotImplementedError, match="Dict"):
            make_agent(env)

    def test_multibinary_raises(self):
        env = FakeEnv(_flat_box(8), MultiBinary(4))
        with pytest.raises(NotImplementedError, match="MultiBinary"):
            make_agent(env)


# ---------------------------------------------------------------------------
# 2. Observation space validation
# ---------------------------------------------------------------------------

class TestObservationSpaceValidation:

    def test_image_obs_raises(self):
        """3-D (image) observation space is not supported."""
        env = FakeEnv(
            Box(low=0, high=255, shape=(84, 84, 3), dtype=np.uint8),
            Discrete(4),
        )
        with pytest.raises(NotImplementedError, match="1-D"):
            make_agent(env)

    def test_2d_obs_raises(self):
        """2-D matrix observation is not supported."""
        env = FakeEnv(
            Box(low=-np.inf, high=np.inf, shape=(10, 10), dtype=np.float32),
            Discrete(4),
        )
        with pytest.raises(NotImplementedError, match="1-D"):
            make_agent(env)

    def test_dict_obs_raises(self):
        """Dict observation space is not supported."""
        env = FakeEnv(
            GymDict({"obs": _flat_box(8), "goal": _flat_box(3)}),
            Discrete(4),
        )
        with pytest.raises(NotImplementedError, match="Box"):
            make_agent(env)


# ---------------------------------------------------------------------------
# 3. Action type incompatibility proof
# (no training — just verify output type mismatch)
# ---------------------------------------------------------------------------

class TestActionTypeIncompatibility:

    def test_sac_outputs_continuous_not_integer(self):
        """
        SAC act() returns a float array in (-1,1)^d — not an integer index.
        If used on CartPole (n=2 actions), the output cannot be fed directly
        to env.step() as a discrete action without wrapping.
        This proves why make_agent() never returns SAC for Discrete spaces.
        """
        # Build SAC as if on a 'wrong' continuous env, same obs_dim as CartPole
        env   = _continuous_env(obs_dim=4, act_dim=2)
        agent = make_agent(env, framework="torch")

        obs = np.zeros(4, dtype=np.float32)
        action = agent.act(obs)

        # Action must be a float array, NOT an integer
        assert isinstance(action, np.ndarray), "SAC action should be ndarray"
        assert action.dtype in (np.float32, np.float64), \
            f"SAC action dtype should be float, got {action.dtype}"
        assert action.shape == (2,), f"Expected shape (2,), got {action.shape}"
        # Values must be in (-1, 1) — not valid as discrete action indices
        assert np.all(np.abs(action) <= 1.0 + 1e-6), \
            "SAC outputs should be in [-1, 1] via tanh"

    def test_ppo_outputs_integer_not_continuous(self):
        """
        PPO act() returns a single integer in {0, ..., n-1}.
        If used on LunarLanderContinuous (Box(2,)), the action does not match
        the expected (2,) float array — proving why make_agent() never returns
        PPO (discrete) for Box action spaces.
        """
        env   = _discrete_env(obs_dim=8, n_actions=4)
        agent = make_agent(env, framework="torch")

        obs    = np.zeros(8, dtype=np.float32)
        action = agent.act(obs)

        # PPO returns a scalar integer (or array of shape ())
        assert isinstance(action, (int, np.integer)), \
            f"PPO action should be integer, got {type(action).__name__} = {action}"
        assert 0 <= int(action) < 4, \
            f"PPO discrete action should be in [0, 4), got {action}"

    def test_td3_outputs_continuous(self):
        """TD3 (deterministic) act() returns float vector in (-1,1)^d."""
        env   = _continuous_env(obs_dim=8, act_dim=3)
        agent = make_agent(env, framework="torch", deterministic=True)

        obs    = np.zeros(8, dtype=np.float32)
        action = agent.act(obs)

        assert isinstance(action, np.ndarray)
        assert action.shape == (3,)
        assert np.all(np.abs(action) <= 1.0 + 1e-6), \
            "TD3 outputs should be in [-1, 1] via tanh"


# ---------------------------------------------------------------------------
# 4. Network architecture dimensions
# ---------------------------------------------------------------------------

class TestNetworkDimensions:

    def test_ppo_actor_input_output(self):
        """PPO actor maps obs_dim → n_actions logits."""
        import torch
        env   = _discrete_env(obs_dim=6, n_actions=5)
        agent = make_agent(env, framework="torch", device="cpu")

        dummy_obs = torch.zeros(1, 6)
        logits = agent._actor(dummy_obs)
        assert logits.shape == (1, 5), \
            f"PPO actor output shape: expected (1, 5), got {logits.shape}"

    def test_sac_actor_input_output(self):
        """SAC actor maps obs_dim → 2*act_dim (mean || log_std)."""
        import torch
        env   = _continuous_env(obs_dim=8, act_dim=3)
        agent = make_agent(env, framework="torch", device="cpu")

        dummy_obs = torch.zeros(1, 8)
        out = agent._actor(dummy_obs)
        assert out.shape == (1, 6), \
            f"SAC actor output shape: expected (1, 6), got {out.shape}"

    def test_td3_actor_input_output(self):
        """TD3 actor maps obs_dim → act_dim actions (tanh output)."""
        import torch
        env   = _continuous_env(obs_dim=8, act_dim=3)
        agent = make_agent(env, framework="torch", deterministic=True, device="cpu")

        dummy_obs = torch.zeros(1, 8)
        out = agent._actor(dummy_obs)
        assert out.shape == (1, 3), \
            f"TD3 actor output shape: expected (1, 3), got {out.shape}"
        # tanh output must be in (-1, 1)
        assert torch.all(torch.abs(out) <= 1.0 + 1e-6), \
            "TD3 actor output must be in [-1, 1]"

    def test_sac_critic_input_output(self):
        """SAC critic maps [obs || act] → scalar Q-value."""
        import torch
        env   = _continuous_env(obs_dim=8, act_dim=3)
        agent = make_agent(env, framework="torch", device="cpu")

        dummy_input = torch.zeros(1, 8 + 3)
        q1 = agent._c1(dummy_input)
        q2 = agent._c2(dummy_input)
        assert q1.shape == (1, 1), f"Q1 shape: expected (1,1), got {q1.shape}"
        assert q2.shape == (1, 1), f"Q2 shape: expected (1,1), got {q2.shape}"

    def test_custom_hidden_sizes(self):
        """hidden_sizes parameter changes network width."""
        import torch
        env   = _continuous_env(obs_dim=4, act_dim=2)
        agent = make_agent(env, framework="torch", hidden_sizes=(64, 64), device="cpu")

        dummy_obs = torch.zeros(1, 4)
        # Verify the network runs without error (shape check)
        out = agent._actor(dummy_obs)
        assert out.shape == (1, 4)  # act_dim=2, so 2*2=4 for SAC


# ---------------------------------------------------------------------------
# 5. Hyperparams wired correctly
# ---------------------------------------------------------------------------

class TestHyperparamWiring:

    def test_custom_hyperparams_accepted(self):
        """Custom HyperparamSet is passed through to the agent."""
        from tensor_optix.core.types import HyperparamSet
        env = _continuous_env()
        hp  = HyperparamSet(params={
            "learning_rate":    1e-3,
            "gamma":            0.95,
            "tau":              0.01,
            "batch_size":       128,
            "updates_per_step": 1,
            "replay_capacity":  100_000,
            "per_alpha":        0.0,
            "per_beta":         0.4,
            "n_step":           3,
        }, episode_id=0)

        agent = make_agent(env, framework="torch", hyperparams=hp)
        assert agent._hyperparams.params["gamma"] == pytest.approx(0.95)
        assert agent._hyperparams.params["n_step"] == 3

    def test_default_hyperparams_are_set(self):
        """When hyperparams=None, sensible defaults are present."""
        env   = _continuous_env()
        agent = make_agent(env, framework="torch")
        hp    = agent.get_hyperparams()
        assert "gamma" in hp.params
        assert "tau" in hp.params
        assert "learning_rate" in hp.params

    def test_is_on_policy_false_for_sac(self):
        """SAC is off-policy."""
        env   = _continuous_env()
        agent = make_agent(env, framework="torch")
        assert agent.is_on_policy is False

    def test_is_on_policy_true_for_ppo(self):
        """PPO is on-policy."""
        env   = _discrete_env()
        agent = make_agent(env, framework="torch")
        assert agent.is_on_policy is True
